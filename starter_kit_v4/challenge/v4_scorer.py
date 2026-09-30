#!/usr/bin/env python3
"""v4 scorer: exposure quality integration, per-target scoring, best-score settlement.

Formula (2026-09-28 ruling; all constants live in config/v4_score_config.json):

    q_exp(i) = q0^-1 · mean_t[eff* · transp* · sky_quality* · lunar(i)
               / (seeing* · airmass(i,t)^0.6)]
    g(i,e)   = min( f(i)·t·q_exp(i) / (f0·t0), 1 )        # obstruction/miss/altitude -> 0
    s(i,e)   = weight(i) · g(i,e)
    c(i,e)   = s(i,e) · prog_mult(i,e)
    best(i)  = max over valid exposures e of c(i,e)
    total    = sum best(i) - P_req·#(required with max valid factor < 0.5)
               - U·(1 - Jain(r_1..r_K)) + report settlement

The starred weather components include applicable event multipliers. Site-level
weather components are integrated over the exposure interval: each overlapped
slot contributes its truth values weighted by overlap seconds, and closed slots or
daytime gaps contribute zero. Directional (HORIZON_SECTOR) event multipliers and
closures are applied per target only while they are active and cover the target.
Airmass, lunar quality, and directional-event coverage use the target position
at each integration-piece midpoint. The program band is derived per target from
site weather and that target's own time-resolved lunar quality and airmass,
without instrument efficiency (v3 convention) or directional-event multipliers.

The BestLedger keeps every raw per-target contribution with its observe-action index so
best scores are always replayable: invalidating an action window (data loss) is a flag
flip plus a rescan, never a destructive edit.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Callable, Mapping, Sequence

from .tile_geometry_simulator import (
    _angular_separation_deg,
    _moon_equatorial_deg,
    _normalized_airmass,
    _sun_equatorial_deg,
)
from .v4_fiber_map import min_altitude_during, radec_to_altaz


PROGRAMS = ("DARK", "BRIGHT", "BACKUP")

# Event types from v4_events.csv that the scorer applies per target. ALL-scope events
# are already folded into the per-slot weather truth and must not be double-counted.
DIRECTIONAL_MULTIPLIER_TYPES = ("rainy", "cloudy", "smoggy", "cold_wave", "tornado")
FORCE_CLOSE_TYPE = "rocket_launch"
ZERO_SCORE_TYPE = "terrain_obstruction"
FAULT_TYPE = "instrument_fault"


def validate_lunar_model(score_config: Mapping) -> None:
    """Validate the v3-compatible lunar quality parameters once at scenario load."""
    lunar = score_config["lunar_model"]
    if float(lunar["angular_decay_scale_deg"]) <= 0.0:
        raise ValueError("lunar angular decay scale must be positive")
    if float(lunar["altitude_exponent"]) <= 0.0:
        raise ValueError("lunar altitude exponent must be positive")
    if not 0.0 <= float(lunar["maximum_penalty"]) < 1.0:
        raise ValueError("lunar maximum penalty must lie in [0, 1)")


@lru_cache(maxsize=65536)
def _lunar_geometry(moment: datetime, latitude_deg: float, longitude_deg: float) -> tuple[float, float, float, float]:
    """Moon RA/Dec, illuminated fraction, and altitude shared across targets."""
    moon_ra, moon_dec = _moon_equatorial_deg(moment)
    sun_ra, sun_dec = _sun_equatorial_deg(moment)
    sun_moon_separation = _angular_separation_deg(sun_ra, sun_dec, moon_ra, moon_dec)
    illumination = (1.0 - math.cos(math.radians(sun_moon_separation))) / 2.0
    moon_altitude, _ = radec_to_altaz(moon_ra, moon_dec, moment, latitude_deg, longitude_deg)
    return moon_ra, moon_dec, illumination, moon_altitude


def lunar_quality_factor(
    ra_deg: float, dec_deg: float, moment: datetime, site: Mapping, score_config: Mapping
) -> float:
    """V3-style Moon penalty for a target direction at one instant."""
    lunar = score_config["lunar_model"]
    moon_ra, moon_dec, illumination, moon_altitude = _lunar_geometry(
        moment, float(site["latitude_deg"]), float(site["longitude_deg"])
    )
    if moon_altitude <= 0.0:
        return 1.0
    separation = _angular_separation_deg(ra_deg, dec_deg, moon_ra, moon_dec)
    altitude_weight = math.sin(math.radians(moon_altitude)) ** float(lunar["altitude_exponent"])
    angular_weight = math.exp(-separation / float(lunar["angular_decay_scale_deg"]))
    penalty = float(lunar["maximum_penalty"]) * illumination * altitude_weight * angular_weight
    return max(0.0, min(1.0, 1.0 - penalty))


def azimuth_inside(value: float, start: float, end: float) -> bool:
    return start <= value <= end if start <= end else value >= start or value <= end


@dataclass(frozen=True)
class SlotTruth:
    slot_id: str
    start_utc: datetime
    end_utc: datetime
    is_observable: bool
    seeing_arcsec: float
    transparency: float
    sky_quality: float
    instrument_efficiency: float


@dataclass(frozen=True)
class ScoreEvent:
    event_id: str
    event_type: str
    actual_start_utc: datetime
    actual_end_utc: datetime
    scope_type: str
    azimuth_start_deg: float | None
    azimuth_end_deg: float | None
    min_altitude_deg: float | None
    max_altitude_deg: float | None
    seeing_multiplier: float
    transparency_multiplier: float
    sky_quality_multiplier: float
    instrument_efficiency_multiplier: float
    force_close: bool = False

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.actual_start_utc < end and self.actual_end_utc > start

    def covers_altaz(self, alt_deg: float, az_deg: float) -> bool:
        if self.scope_type == "ALL":
            return True
        assert (
            self.azimuth_start_deg is not None
            and self.azimuth_end_deg is not None
            and self.min_altitude_deg is not None
            and self.max_altitude_deg is not None
        )
        return self.min_altitude_deg <= alt_deg <= self.max_altitude_deg and azimuth_inside(
            az_deg, self.azimuth_start_deg, self.azimuth_end_deg
        )


class WeatherTruth:
    """Slot-aligned site weather with independent repair state for each fault event."""

    def __init__(
        self, slots: Sequence[SlotTruth], faults: ScoreEvent | Sequence[ScoreEvent] | None
    ) -> None:
        if not slots:
            raise ValueError("weather truth must contain slots")
        self.slots = list(slots)
        self._starts = [slot.start_utc for slot in self.slots]
        if faults is None:
            self.faults = []
        elif isinstance(faults, ScoreEvent):
            self.faults = [faults]
        else:
            self.faults = sorted(faults, key=lambda event: event.actual_start_utc)
        if any(
            left.actual_end_utc > right.actual_start_utc
            for left, right in zip(self.faults, self.faults[1:])
        ):
            raise ValueError("instrument fault events must not overlap")
        self.fault_repairs: dict[str, datetime] = {}
        self.false_reports_since_correct = 0

    @property
    def fault_repair_utc(self) -> datetime | None:
        """Compatibility view for the single-fault case."""
        return self.fault_repairs.get(self.faults[0].event_id) if len(self.faults) == 1 else None

    def active_fault(self, moment: datetime) -> ScoreEvent | None:
        return next(
            (fault for fault in self.faults
             if fault.actual_start_utc <= moment < fault.actual_end_utc),
            None,
        )

    def repair_fault(self, moment: datetime) -> bool:
        fault = self.active_fault(moment)
        if fault is None or fault.event_id in self.fault_repairs:
            return False
        self.fault_repairs[fault.event_id] = moment
        return True

    def _slot_index(self, moment: datetime) -> int:
        # Slots are chronological; the start list is built once (the prototype rebuilt it
        # on every call, which dominated the per-decision cost on long seasons).
        return max(0, bisect.bisect_right(self._starts, moment) - 1)

    def segments(self, start: datetime, end: datetime):
        """Yield (seconds, SlotTruth) overlap pieces of [start, end) against the slots."""
        cursor = start
        index = self._slot_index(start)
        while cursor < end:
            if index >= len(self.slots):
                yield (end - cursor).total_seconds(), None
                break
            slot = self.slots[index]
            if slot.end_utc <= cursor:
                index += 1
                continue
            if cursor < slot.start_utc:
                gap_end = min(end, slot.start_utc)
                yield (gap_end - cursor).total_seconds(), None
                cursor = gap_end
                continue
            piece_end = min(end, slot.end_utc)
            yield (piece_end - cursor).total_seconds(), slot
            cursor = piece_end
            index += 1

    def reset_false_report_count(self) -> None:
        """Start a new false-report allowance after a correct report."""
        self.false_reports_since_correct = 0

    def report_fault(self, moment: datetime, score_config: Mapping) -> float:
        """Repair a correct report; penalize false reports after the free threshold."""
        if self.repair_fault(moment):
            self.reset_false_report_count()
            return float(score_config["reporting"]["correct_reward"])
        self.false_reports_since_correct += 1
        if self.false_reports_since_correct <= int(
            score_config["reporting"].get("false_report_free_allowance", 0)
        ):
            return 0.0
        return float(score_config["reporting"]["false_penalty"])

    def _apply_fault(self, truth: SlotTruth, moment: datetime) -> float:
        """Effective instrument efficiency for a slot, undoing a repaired fault."""
        efficiency = truth.instrument_efficiency
        fault = self.active_fault(moment)
        repair_at = self.fault_repairs.get(fault.event_id) if fault is not None else None
        if (
            fault is not None
            and repair_at is not None
            and moment >= repair_at
            and fault.instrument_efficiency_multiplier > 0
        ):
            efficiency = efficiency / fault.instrument_efficiency_multiplier
            efficiency = min(1.0, efficiency)
        return efficiency

    def band_quality(
        self,
        start: datetime,
        end: datetime,
        score_config: Mapping,
        target_altitude_at: Callable[[datetime], float],
        lunar_factor_at: Callable[[datetime], float] | None = None,
    ) -> float:
        """Program-band quality with time-resolved target lunar factor and airmass.

        The runner passes per-target values, so one exposure may split across bands.
        """
        total = 0.0
        weight = 0.0
        airmass_exponent = float(score_config["airmass_exponent"])
        cursor = start
        for seconds, truth in self.segments(start, end):
            weight += seconds
            segment_end = cursor + timedelta(seconds=seconds)
            piece = cursor
            while piece < segment_end:
                piece_end = min(segment_end, piece + timedelta(seconds=120))
                sample = piece + (piece_end - piece) / 2
                if truth is not None and truth.is_observable:
                    lunar = lunar_factor_at(sample) if lunar_factor_at else 1.0
                    airmass = _normalized_airmass(target_altitude_at(sample))
                    total += (
                        (piece_end - piece).total_seconds()
                        * truth.transparency
                        * truth.sky_quality
                        * lunar
                        / truth.seeing_arcsec
                        / airmass**airmass_exponent
                    )
                piece = piece_end
            cursor = segment_end
        if weight <= 0.0:
            return 0.0
        return (total / weight) / float(score_config["q0"])


def program_band(q_band: float, score_config: Mapping) -> str:
    bands = score_config["program"]["bands"]
    if q_band >= float(bands["DARK"]):
        return "DARK"
    if q_band >= float(bands["BRIGHT"]):
        return "BRIGHT"
    return "BACKUP"


def program_multiplier(declared: str, actual: str, score_config: Mapping) -> float:
    if declared not in PROGRAMS:
        raise ValueError(f"unknown program {declared!r}")
    if declared == actual:
        return float(score_config["program"]["multipliers"][declared])
    return float(score_config["program"]["mismatch_multiplier"])


@dataclass(frozen=True)
class TargetScore:
    target_id: str
    factor: float  # g(i,e) in [0, 1]
    quality: float  # q_exp before capping
    score: float  # c(i,e) = s(i,e) * prog_mult


def score_target_exposure(
    target: Mapping,
    segments: Sequence[tuple[float, SlotTruth | None]],
    events: Sequence[ScoreEvent],
    weather: WeatherTruth,
    duration_seconds: int,
    prog_mult: float,
    score_config: Mapping,
    exposure_start_utc: datetime,
    target_altaz_at: Callable[[datetime], tuple[float, float]],
    site: Mapping | None = None,
) -> TargetScore:
    """One target's capped contribution for one exposure.

    `segments` are the exposure's slot overlaps. Event start/end boundaries split
    each segment before integration. The target's horizon position and airmass are
    refreshed in at most 120-second pieces. The runner checks fiber assignment and
    the altitude limit before calling this function.
    """
    lunar_enabled = "lunar_model" in score_config
    if lunar_enabled and site is None:
        raise ValueError("site is required when lunar_model is configured")
    zero = TargetScore(str(target["target_id"]), 0.0, 0.0, 0.0)
    total = 0.0
    weight_sum = 0.0
    airmass_exponent = float(score_config["airmass_exponent"])
    cursor = exposure_start_utc
    for seconds, truth in segments:
        weight_sum += seconds
        segment_end = cursor + timedelta(seconds=seconds)
        cuts = {cursor, segment_end}
        for event in events:
            if cursor < event.actual_start_utc < segment_end:
                cuts.add(event.actual_start_utc)
            if cursor < event.actual_end_utc < segment_end:
                cuts.add(event.actual_end_utc)
        boundaries = sorted(cuts)
        for left, right in zip(boundaries, boundaries[1:]):
            piece = left
            while piece < right:
                piece_end = min(right, piece + timedelta(seconds=120))
                sample = piece + (piece_end - piece) / 2
                alt, az = target_altaz_at(sample)
                active = [
                    event for event in events
                    if event.actual_start_utc <= sample < event.actual_end_utc
                    and event.covers_altaz(alt, az)
                ]
                if truth is not None and truth.is_observable and not any(
                    event.force_close or event.event_type in (ZERO_SCORE_TYPE, FORCE_CLOSE_TYPE)
                    for event in active
                ):
                    seeing = truth.seeing_arcsec
                    transparency = truth.transparency
                    sky = truth.sky_quality
                    efficiency = weather._apply_fault(truth, sample)
                    for event in active:
                        if event.event_type in DIRECTIONAL_MULTIPLIER_TYPES:
                            seeing *= event.seeing_multiplier
                            transparency *= event.transparency_multiplier
                            sky *= event.sky_quality_multiplier
                            efficiency *= event.instrument_efficiency_multiplier
                    lunar = lunar_quality_factor(
                        float(target["ra_deg"]), float(target["dec_deg"]), sample, site, score_config
                    ) if lunar_enabled else 1.0
                    airmass = _normalized_airmass(alt)
                    total += (
                        (piece_end - piece).total_seconds()
                        * efficiency
                        * transparency
                        * sky
                        * lunar
                        / seeing
                        / airmass**airmass_exponent
                    )
                piece = piece_end
        cursor = segment_end
    if weight_sum <= 0.0:
        return zero
    quality = (total / weight_sum) / float(score_config["q0"])
    factor = min(
        float(target["feature_flux"])
        * duration_seconds
        * quality
        / (float(score_config["flux_zero_point"]) * int(score_config["exposure_zero_point_seconds"])),
        1.0,
    )
    score = float(target["science_weight"]) * factor * prog_mult
    return TargetScore(str(target["target_id"]), factor, quality, score)


@dataclass
class LedgerEntry:
    action_index: int
    target_id: str
    factor: float
    score: float
    valid: bool = True
    invalidated_by: str = ""


class BestLedger:
    """Raw per-target contributions with replayable best-score settlement."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, action_index: int, target_id: str, factor: float, score: float) -> None:
        self.entries.append(LedgerEntry(action_index, target_id, factor, score))

    def invalidate_window(self, first: int, last_exclusive: int, event_id: str) -> int:
        count = 0
        for entry in self.entries:
            if entry.valid and first <= entry.action_index < last_exclusive:
                entry.valid = False
                entry.invalidated_by = event_id
                count += 1
        return count

    def best(self) -> dict[str, tuple[float, float]]:
        """target_id -> (factor, score) of its best valid contribution (max score)."""
        best: dict[str, tuple[float, float]] = {}
        for entry in self.entries:
            if not entry.valid:
                continue
            current = best.get(entry.target_id)
            if current is None or entry.score > current[1]:
                best[entry.target_id] = (entry.factor, entry.score)
        return best

    def max_factors(self) -> dict[str, float]:
        """Largest valid exposure factor for each target, independent of score bonuses."""
        factors: dict[str, float] = {}
        for entry in self.entries:
            if entry.valid:
                factors[entry.target_id] = max(factors.get(entry.target_id, 0.0), entry.factor)
        return factors


def required_penalty(
    targets: Sequence[Mapping], max_factors: Mapping[str, float], score_config: Mapping
) -> tuple[int, float]:
    threshold = float(score_config["required"]["observed_factor_threshold"])
    missing = sum(
        1
        for target in targets
        if target["required"] and max_factors.get(str(target["target_id"]), 0.0) < threshold
    )
    return missing, missing * float(score_config["required"]["penalty_per_missing"])


def uniformity_penalty(
    targets: Sequence[Mapping], max_factors: Mapping[str, float], score_config: Mapping
) -> tuple[dict[str, float], float]:
    """Jain-index penalty over RA bands: r_k = observed fraction (factor >= 0.5) per band."""
    width = float(score_config["uniformity"]["ra_band_width_deg"])
    threshold = float(score_config["uniformity"]["observed_factor_threshold"])
    band_total: dict[int, int] = {}
    band_observed: dict[int, int] = {}
    for target in targets:
        band = int(float(target["ra_deg"]) // width)
        band_total[band] = band_total.get(band, 0) + 1
        if max_factors.get(str(target["target_id"]), 0.0) >= threshold:
            band_observed[band] = band_observed.get(band, 0) + 1
    ratios = {
        f"ra_{band * width:.0f}_{(band + 1) * width:.0f}": band_observed.get(band, 0) / count
        for band, count in sorted(band_total.items())
    }
    values = list(ratios.values())
    if not values:
        return ratios, 0.0
    squares = sum(value**2 for value in values)
    # All bands empty: Jain is undefined; treat as maximally uneven (penalty = U).
    jain = 0.0 if squares == 0.0 else sum(values) ** 2 / (len(values) * squares)
    return ratios, float(score_config["uniformity"]["weight"]) * (1.0 - jain)


def settle_reports(reports: Sequence[datetime], weather: WeatherTruth, score_config: Mapping) -> float:
    """Instrument-fault reporting settlement (+reward / false-report penalty, v3-style)."""
    return sum(weather.report_fault(moment, score_config) for moment in reports)
