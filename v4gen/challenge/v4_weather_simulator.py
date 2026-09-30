#!/usr/bin/env python3
"""Generate the v4 weather and event subsystem: full truth plus agent-side publications.

Truth line (organizer/scorer): a seeded night calendar and 900 s slots for the v4 site
and date range, per-slot weather truth (v3 background model: seasonal + AR(1) night/slot
quality, background dome closure), directional weather system events, scheduled rocket
launches, global earthquakes with an exponentially magnitude-scaled shock that decays
exponentially per night, permanent terrain-obstruction azimuth sectors, and v3-semantics
instrument faults (hidden efficiency hit, revealed only through the efficiency jitter in
scores). nova/reddening anomalies from v3 are dropped.

Publication line (agent): per-slot bulletins and periodic coarse forecasts as JSONL.
Every notice carries only the coarse event kind plus an 8-point compass direction (or
"ALL"); no numeric weather quantities, severities, or exact scopes are ever published.
Earthquakes and instrument faults never appear in forecasts; instrument faults never
appear in any publication.

Runtime code uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import (
    NIGHT_COLUMNS,
    SLOT_COLUMNS,
    WEATHER_COLUMNS,
    SEED_DERIVATION_KEY,
    SEED_DERIVATIONS,
    format_utc,
    sha256_file,
    stream_seed,
    write_exact_csv,
    write_text_lf,
)
from .observing_calendar import Night, Slot
from .v4_catalog_generator import _sun_altitude_deg, _twilight_crossing


SCHEMA_VERSION = "v4-weather-v2"

WEATHER_CONDITIONS = ("rainy", "cloudy", "smoggy", "cold_wave", "tornado")
EVENT_TYPES = WEATHER_CONDITIONS + ("rocket_launch", "earthquake", "terrain_obstruction", "instrument_fault")
FORECASTABLE_TYPES = WEATHER_CONDITIONS + ("rocket_launch",)
# Instrument faults are hidden entirely; earthquakes may be announced as current
# conditions but never forecast.
BULLETIN_HIDDEN_TYPES = ("instrument_fault",)

# Coarse publication vocabulary: weather phrases map onto weather system events rather
# than constituting their own event types.
COARSE_KIND = {
    "rainy": "rain",
    "cloudy": "overcast",
    "smoggy": "haze",
    "cold_wave": "cold_snap",
    "tornado": "storm",
    "rocket_launch": "rocket_launch",
    "earthquake": "earthquake",
    "terrain_obstruction": "terrain_obstruction",
    "instrument_fault": "instrument_fault",
}

DIRECTION_CODES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
ALL_DIRECTION = "ALL"

V4_EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "actual_start_utc",
    "actual_end_utc",
    "scope_type",
    "azimuth_start_deg",
    "azimuth_end_deg",
    "min_altitude_deg",
    "max_altitude_deg",
    "magnitude",
    "severity",
    "force_close",
    "zero_score",
    "seeing_multiplier",
    "transparency_multiplier",
    "sky_quality_multiplier",
    "instrument_efficiency_multiplier",
]

V4_EARTHQUAKE_EFFECT_COLUMNS = [
    "event_id",
    "night_id",
    "magnitude",
    "degradation",
    "seeing_multiplier",
    "transparency_multiplier",
    "sky_quality_multiplier",
    "instrument_efficiency_multiplier",
]

SCOPE_TYPES = ("ALL", "HORIZON_SECTOR")

V4_STRESS_EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "trigger_mode",
    "trigger_ref",
    "window_start_fraction",
    "window_end_fraction",
    "window_max_fraction",
    "alt_offset_rad",
    "az_offset_rad",
]

STRESS_EVENT_TYPES = ("data_loss", "pointing_offset")

# Publication contract for the data-loss runtime resync (MP-055). The generator only
# defines the schema; the v4 runner/scorer fills the values when the loss triggers.
# Numbering convention (confirmed 2026-09-27): action indices are 0-based; the
# invalidated window is the half-open interval [floor(i·N), floor(j·N)) of the N
# **observe** actions executed so far (wait actions produce no losable data and are
# excluded from the numbering space); window width = floor(j·N) - floor(i·N), which
# floor rounding keeps within ceil(x·N).
STATE_RESYNC_SCHEMA = {
    "schema_version": "v4-state-resync-v1",
    "record_type": "state_resync",
    "semantics": (
        "Sent by the runner when a data_loss stress event triggers. The message itself "
        "is the notification; no separate bulletin is issued. Observe actions in the "
        "window are invalidated without refunding their time (wait actions are not part "
        "of the window's numbering space); the original action records are kept "
        "organizer-side; best scores are recomputed by replay with the window removed."
    ),
    "fields": {
        "record_type": "the string \"state_resync\"",
        "issued_at_utc": "UTC timestamp of the trigger slot",
        "trigger_event_id": "event_id of the data_loss row in v4_stress_events.csv",
        "invalidated_window": {
            "action_count_at_trigger": "N, observe actions executed so far (wait actions excluded)",
            "action_index_start": "floor(window_start_fraction * N), 0-based over observe actions, inclusive",
            "action_index_end_exclusive": "floor(window_end_fraction * N), exclusive",
            "window_start_fraction": "i from the truth row",
            "window_end_fraction": "j from the truth row",
            "window_max_fraction": "x from the truth row",
        },
        "observed_target_ids": ["target_id", "..."],
        "best_scores": [{"target_id": "target_id", "best_score": "float"}],
    },
    "notes": [
        "No numeric weather quantities are involved; this contract concerns action history only.",
        "The bulletin stream carries no separate data_loss notice: this message is the notice.",
    ],
}


def azimuth_to_direction(azimuth_deg: float) -> str:
    """Map an azimuth (0=north, 90=east) to the nearest 8-point compass code."""
    return DIRECTION_CODES[int((azimuth_deg % 360.0 + 22.5) // 45.0) % 8]


def _sector_center_azimuth(start_deg: float, end_deg: float) -> float:
    width = (end_deg - start_deg) % 360.0
    return (start_deg + width / 2.0) % 360.0


def event_direction(event: "V4Event") -> str:
    if event.scope_type == ALL_DIRECTION:
        return ALL_DIRECTION
    return azimuth_to_direction(_sector_center_azimuth(event.azimuth_start_deg, event.azimuth_end_deg))


@dataclass(frozen=True)
class V4Event:
    event_id: str
    event_type: str
    actual_start_utc: datetime
    actual_end_utc: datetime
    scope_type: str  # "ALL" or "HORIZON_SECTOR"
    azimuth_start_deg: float | None
    azimuth_end_deg: float | None
    min_altitude_deg: float | None
    max_altitude_deg: float | None
    magnitude: float | None
    severity: float
    force_close: bool
    zero_score: bool
    seeing_multiplier: float
    transparency_multiplier: float
    sky_quality_multiplier: float
    instrument_efficiency_multiplier: float

    def overlaps(self, moment_start: datetime, moment_end: datetime) -> bool:
        return self.actual_start_utc < moment_end and self.actual_end_utc > moment_start

    def csv_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actual_start_utc": format_utc(self.actual_start_utc),
            "actual_end_utc": format_utc(self.actual_end_utc),
            "scope_type": self.scope_type,
            "azimuth_start_deg": "" if self.azimuth_start_deg is None else f"{self.azimuth_start_deg:.6f}",
            "azimuth_end_deg": "" if self.azimuth_end_deg is None else f"{self.azimuth_end_deg:.6f}",
            "min_altitude_deg": "" if self.min_altitude_deg is None else f"{self.min_altitude_deg:.6f}",
            "max_altitude_deg": "" if self.max_altitude_deg is None else f"{self.max_altitude_deg:.6f}",
            "magnitude": "" if self.magnitude is None else f"{self.magnitude:.3f}",
            "severity": f"{self.severity:.6f}",
            "force_close": str(self.force_close).lower(),
            "zero_score": str(self.zero_score).lower(),
            "seeing_multiplier": f"{self.seeing_multiplier:.6f}",
            "transparency_multiplier": f"{self.transparency_multiplier:.6f}",
            "sky_quality_multiplier": f"{self.sky_quality_multiplier:.6f}",
            "instrument_efficiency_multiplier": f"{self.instrument_efficiency_multiplier:.6f}",
        }


@dataclass(frozen=True)
class EarthquakeEffect:
    event_id: str
    night_id: str
    magnitude: float
    degradation: float
    seeing_multiplier: float
    transparency_multiplier: float
    sky_quality_multiplier: float
    instrument_efficiency_multiplier: float

    def csv_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "night_id": self.night_id,
            "magnitude": f"{self.magnitude:.3f}",
            "degradation": f"{self.degradation:.6f}",
            "seeing_multiplier": f"{self.seeing_multiplier:.6f}",
            "transparency_multiplier": f"{self.transparency_multiplier:.6f}",
            "sky_quality_multiplier": f"{self.sky_quality_multiplier:.6f}",
            "instrument_efficiency_multiplier": f"{self.instrument_efficiency_multiplier:.6f}",
        }


def earthquake_initial_degradation(magnitude: float, config: Mapping) -> float:
    """Shock-night degradation: exponential in magnitude, capped below total loss."""
    quake = config["earthquake"]
    raw = math.exp(
        float(quake["impact_coefficient"])
        * (magnitude - float(quake["reference_magnitude"]))
    )
    return min(float(quake["max_degradation"]), raw)


def earthquake_night_degradation(
    initial_degradation: float, nights_since_shock: int, config: Mapping
) -> float:
    """Exponential per-night decay of the post-shock degradation."""
    if nights_since_shock < 0:
        return 0.0
    return initial_degradation * math.exp(
        -nights_since_shock / float(config["earthquake"]["decay_nights"])
    )


def _earthquake_multipliers(degradation: float, config: Mapping) -> tuple[float, float, float, float]:
    quake = config["earthquake"]
    return (
        1.0 + float(quake["seeing_impact_coefficient"]) * degradation,
        1.0 - float(quake["transparency_impact_coefficient"]) * degradation,
        1.0 - float(quake["sky_quality_impact_coefficient"]) * degradation,
        1.0 - degradation,
    )


# --- config ----------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping) -> None:
    required = {
        "schema_version", "seed", "site", "survey", "quality", "background_closure",
        "weather_events", "rocket_launch", "earthquake", "terrain_obstruction",
        "instrument_fault", "publication", "output",
    }
    if not required <= set(config) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid v4 weather config keys or schema_version")
    _validate_seed(config)
    survey = config["survey"]
    start = date.fromisoformat(str(survey["start_date"]))
    end = date.fromisoformat(str(survey["end_date"]))
    if end <= start:
        raise ValueError("survey end_date must follow start_date")
    if int(survey["slot_seconds"]) < 60 or 86400 % int(survey["slot_seconds"]):
        raise ValueError("slot_seconds must be >= 60 and divide one day")
    if not -90.0 < float(survey["sun_altitude_limit_deg"]) < 0.0:
        raise ValueError("sun altitude limit must lie in (-90, 0)")
    conditions = config["weather_events"]["conditions"]
    if set(conditions) != set(WEATHER_CONDITIONS):
        raise ValueError("weather_events.conditions must define exactly the five weather conditions")
    for name, definition in conditions.items():
        if int(definition["count"]) < 0:
            raise ValueError(f"{name}: count must be non-negative")
        if not set(definition["scope_weights"]) <= set(SCOPE_TYPES):
            raise ValueError(f"{name}: scope_weights may only use ALL / HORIZON_SECTOR")
        lo, hi = map(int, definition["duration_slots"])
        if lo < 1 or hi < lo:
            raise ValueError(f"{name}: invalid duration_slots")
    quake = config["earthquake"]
    mag_lo, mag_hi = map(float, quake["magnitude_range"])
    if not 0.0 < mag_lo <= mag_hi:
        raise ValueError("invalid earthquake magnitude range")
    if float(quake["impact_coefficient"]) <= 0.0:
        raise ValueError("earthquake impact coefficient must be positive (impact grows with magnitude)")
    if not 0.0 < float(quake["max_degradation"]) < 1.0:
        raise ValueError("earthquake max degradation must lie in (0, 1)")
    if float(quake["decay_nights"]) <= 0.0:
        raise ValueError("earthquake decay_nights must be positive")
    for name in ("seeing_impact_coefficient", "transparency_impact_coefficient", "sky_quality_impact_coefficient"):
        if float(quake[name]) < 0.0:
            raise ValueError(f"earthquake {name} must be non-negative")
    terrain = config["terrain_obstruction"]
    lo, hi = map(int, terrain["sector_count"])
    if lo < 1 or hi < lo:
        raise ValueError("invalid terrain sector_count range")
    rocket = config["rocket_launch"]
    if int(rocket["count"]) < 0:
        raise ValueError("rocket launch count must be non-negative")
    if "force_close" in rocket:
        raise ValueError("rocket_launch.force_close is obsolete; rocket launches always close their sector")
    if int(config["instrument_fault"]["count"]) < 0:
        raise ValueError("instrument_fault.count must be non-negative")
    if int(config["publication"]["forecast_interval_days"]) < 1:
        raise ValueError("forecast interval must be positive")
    if int(config["publication"]["forecast_horizon_days"]) < 1:
        raise ValueError("forecast horizon must be positive")
    stress = config.get("stress_tests")
    if stress is not None:
        if not isinstance(stress["enabled"], bool):
            raise ValueError("stress_tests.enabled must be a boolean")
        data_loss = stress["data_loss"]
        if data_loss["trigger"] not in ("after_earthquake", "fixed_slot", "fixed_date"):
            raise ValueError("data_loss trigger must be after_earthquake / fixed_slot / fixed_date")
        if data_loss["trigger"] == "fixed_slot" and "slot_id" not in data_loss:
            raise ValueError("fixed_slot trigger requires data_loss.slot_id")
        if data_loss["trigger"] == "fixed_date" and "date" not in data_loss:
            raise ValueError("fixed_date trigger requires data_loss.date")
        if not 0.0 < float(data_loss["window_max_fraction"]) <= 1.0:
            raise ValueError("window_max_fraction must lie in (0, 1]")
        offset = stress["pointing_offset"]
        for key in ("max_abs_alt_rad", "max_abs_az_rad"):
            if not 0.0 < float(offset[key]) < 0.1:
                raise ValueError(f"pointing_offset {key} must lie in (0, 0.1) rad")


def _validate_seed(config: Mapping) -> None:
    """Seeds are non-negative integers; ``seed_derivation`` (optional) must be a known mode.

    Without ``seed_derivation`` the legacy ``seed + offset`` streams are used, which keeps the
    author's reference artifacts byte-identical. Competition cards must use ``sha256-v1``
    (enforced by ``v4_config_check.cross_validate_generator_configs(require_hashed_seeds=True)``).
    """
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    mode = config.get(SEED_DERIVATION_KEY)
    if mode is not None and mode not in SEED_DERIVATIONS:
        raise ValueError(f"unsupported {SEED_DERIVATION_KEY} {mode!r}")


# --- v4 night calendar -------------------------------------------------------------


def build_nights(config: Mapping) -> tuple[list[Night], list[Slot]]:
    """Night calendar for the v4 site: sun below the twilight limit, 900 s slots.

    Same construction as observing_calendar.build_calendar but driven by the v4 weather
    config (start/end dates instead of a day count) so this subsystem stays self-contained.
    """
    site = config["site"]
    survey = config["survey"]
    latitude = float(site["latitude_deg"])
    longitude = float(site["longitude_deg"])
    threshold = float(survey["sun_altitude_limit_deg"])
    slot_seconds = int(survey["slot_seconds"])
    local_zone = timezone(timedelta(hours=float(site["utc_offset_hours"])))
    first = date.fromisoformat(str(survey["start_date"]))
    last = date.fromisoformat(str(survey["end_date"]))
    nights: list[Night] = []
    slots: list[Slot] = []
    cursor = first
    while cursor < last:
        start = datetime.combine(cursor, datetime.min.time(), tzinfo=local_zone).replace(hour=12)
        start_utc = start.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(days=1)
        step = timedelta(seconds=300)
        crossings: list[tuple[datetime, bool]] = []
        previous = start_utc
        previous_below = _sun_altitude_deg(previous, latitude, longitude) <= threshold
        probe = previous + step
        while probe <= end_utc:
            below = _sun_altitude_deg(probe, latitude, longitude) <= threshold
            if below != previous_below:
                crossings.append(
                    (_twilight_crossing(previous, probe, previous_below, threshold, latitude, longitude, 1), below)
                )
            previous = probe
            previous_below = below
            probe += step
        dusk = next((moment for moment, entered in crossings if entered), None)
        dawn = next(
            (moment for moment, entered in crossings if not entered and dusk and moment > dusk),
            None,
        )
        if dusk is None or dawn is None:
            raise ValueError(f"no complete observing night for {cursor.isoformat()} at v4 site")
        observing_start = _ceil_epoch(dusk, slot_seconds)
        observing_end = _floor_epoch(dawn, slot_seconds)
        if observing_end <= observing_start:
            raise ValueError(f"no full slots for {cursor.isoformat()}")
        count = int((observing_end - observing_start).total_seconds()) // slot_seconds
        night_id = f"N{cursor.strftime('%Y%m%d')}"
        nights.append(Night(night_id, cursor, dusk, dawn, observing_start, observing_end, count))
        for index in range(count):
            slots.append(
                Slot(
                    slot_id=f"{night_id}-S{index + 1:03d}",
                    night_id=night_id,
                    timestamp_utc=observing_start + timedelta(seconds=index * slot_seconds),
                    duration_seconds=slot_seconds,
                )
            )
        cursor += timedelta(days=1)
    return nights, slots


def _ceil_epoch(moment: datetime, quantum: int) -> datetime:
    seconds = int(moment.timestamp())
    aligned = ((seconds + quantum - 1) // quantum) * quantum
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


def _floor_epoch(moment: datetime, quantum: int) -> datetime:
    seconds = int(moment.timestamp())
    return datetime.fromtimestamp((seconds // quantum) * quantum, tz=timezone.utc)


# --- event generation ---------------------------------------------------------------


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    threshold = rng.random() * sum(float(value) for value in weights.values())
    total = 0.0
    for key, value in weights.items():
        total += float(value)
        if threshold <= total:
            return key
    return next(reversed(weights))


def _scaled_multiplier(configured: float, severity: float) -> float:
    return 1.0 + severity * (configured - 1.0)


def _random_sector(rng: random.Random, width_range, altitude_range) -> tuple[float, float, float, float]:
    start = rng.uniform(0.0, 360.0)
    width = rng.uniform(float(width_range[0]), float(width_range[1]))
    return (
        start,
        (start + width) % 360.0,
        0.0,
        rng.uniform(float(altitude_range[0]), float(altitude_range[1])),
    )


def _end_after_observing_slots(slots: Sequence[Slot], start_index: int, count: int) -> datetime:
    """End after `count` published observing slots, skipping daytime gaps."""
    if count < 1 or not 0 <= start_index < len(slots):
        raise ValueError("invalid observing-slot event span")
    return slots[min(start_index + count, len(slots)) - 1].end_utc


def generate_events(
    config: Mapping, nights: Sequence[Night], slots: Sequence[Slot]
) -> tuple[list[V4Event], list[EarthquakeEffect]]:
    """All truth-layer events. RNG stream per event family keeps families independent."""
    events: list[V4Event] = []
    effects: list[EarthquakeEffect] = []
    sequence = 0

    def next_id() -> str:
        nonlocal sequence
        sequence += 1
        return f"V4EV{sequence:04d}"

    first_start = slots[0].timestamp_utc
    last_end = slots[-1].end_utc

    # Weather system events: v3 placement model, scopes restricted to ALL / HORIZON_SECTOR.
    rng = random.Random(stream_seed(config, "v4.weather.events", 2000))
    weather_cfg = config["weather_events"]
    for condition in WEATHER_CONDITIONS:
        definition = weather_cfg["conditions"][condition]
        for occurrence in range(int(definition["count"])):
            nominal = int((occurrence + 1) * len(slots) / (int(definition["count"]) + 1))
            start_index = max(0, min(len(slots) - 1, nominal + rng.randint(-80, 80)))
            start_slot = slots[start_index]
            duration_slots = rng.randint(*map(int, definition["duration_slots"]))
            start = start_slot.timestamp_utc
            end = _end_after_observing_slots(slots, start_index, duration_slots)
            severity = rng.uniform(0.55, 1.0)
            scope_type = _weighted_choice(rng, definition["scope_weights"])
            if scope_type == "HORIZON_SECTOR":
                az_start, az_end, min_alt, max_alt = _random_sector(
                    rng,
                    weather_cfg["sector_width_deg_range"],
                    weather_cfg["sector_altitude_limit_deg_range"],
                )
            else:
                az_start = az_end = min_alt = max_alt = None
            events.append(
                V4Event(
                    next_id(), condition, start, end, scope_type,
                    az_start, az_end, min_alt, max_alt, None, severity,
                    bool(definition["force_close"]), False,
                    _scaled_multiplier(float(definition["seeing_multiplier"]), severity),
                    _scaled_multiplier(float(definition["transparency_multiplier"]), severity),
                    _scaled_multiplier(float(definition["sky_quality_multiplier"]), severity),
                    1.0,
                )
            )

    # Rocket launches: schedule-driven, short, directional, forecastable.
    # A launch closure ends no later than the final slot of its starting night.
    rng = random.Random(stream_seed(config, "v4.rocket_launch", 2100))
    rocket = config["rocket_launch"]
    night_end_by_id = {night.night_id: night.observing_end_utc for night in nights}
    for occurrence in range(int(rocket["count"])):
        nominal = int((occurrence + 1) * len(slots) / (int(rocket["count"]) + 1))
        start_index = max(0, min(len(slots) - 1, nominal + rng.randint(-120, 120)))
        start_slot = slots[start_index]
        duration_slots = rng.randint(*map(int, rocket["duration_slots"]))
        end = min(
            _end_after_observing_slots(slots, start_index, duration_slots),
            night_end_by_id[start_slot.night_id],
        )
        az_start, az_end, min_alt, max_alt = _random_sector(
            rng, rocket["azimuth_sector_width_deg"], (0.0, 90.0)
        )
        max_alt = rng.uniform(
            float(rocket["max_altitude_deg"][0]), float(rocket["max_altitude_deg"][1])
        )
        events.append(
            V4Event(
                next_id(), "rocket_launch", start_slot.timestamp_utc,
                end,
                "HORIZON_SECTOR", az_start, az_end, min_alt, max_alt, None,
                1.0, True, False, 1.0, 1.0, 1.0, 1.0,
            )
        )

    # Earthquakes: global, unforecastable; shock degradation exponential in magnitude,
    # then exponential decay per night. The event row is the instant shock; the per-night
    # effect table carries the decaying truth.
    rng = random.Random(stream_seed(config, "v4.earthquake", 2200))
    quake = config["earthquake"]
    night_index_by_id = {night.night_id: index for index, night in enumerate(nights)}
    for occurrence in range(int(quake["count"])):
        nominal = int((occurrence + 1) * len(slots) / (int(quake["count"]) + 1))
        start_slot = slots[max(0, min(len(slots) - 1, nominal + rng.randint(-200, 200)))]
        magnitude = rng.uniform(float(quake["magnitude_range"][0]), float(quake["magnitude_range"][1]))
        event_id = next_id()
        events.append(
            V4Event(
                event_id, "earthquake", start_slot.timestamp_utc,
                start_slot.timestamp_utc + timedelta(seconds=start_slot.duration_seconds),
                "ALL", None, None, None, None, magnitude, 1.0, False, False,
                1.0, 1.0, 1.0, 1.0,
            )
        )
        initial = earthquake_initial_degradation(magnitude, config)
        shock_night = night_index_by_id[start_slot.night_id]
        negligible = float(quake["negligible_degradation"])
        for night_index in range(shock_night, len(nights)):
            degradation = earthquake_night_degradation(initial, night_index - shock_night, config)
            if degradation < negligible:
                break
            seeing, transparency, sky, efficiency = _earthquake_multipliers(degradation, config)
            effects.append(
                EarthquakeEffect(
                    event_id, nights[night_index].night_id, magnitude, degradation,
                    seeing, transparency, sky, efficiency,
                )
            )

    # Terrain obstruction: permanent azimuth sectors, present from the first slot.
    rng = random.Random(stream_seed(config, "v4.terrain", 2300))
    terrain = config["terrain_obstruction"]
    sector_count = rng.randint(*map(int, terrain["sector_count"]))
    for _ in range(sector_count):
        start = rng.uniform(0.0, 360.0)
        width = rng.uniform(float(terrain["width_deg_range"][0]), float(terrain["width_deg_range"][1]))
        max_alt = rng.uniform(
            float(terrain["max_altitude_deg_range"][0]), float(terrain["max_altitude_deg_range"][1])
        )
        events.append(
            V4Event(
                next_id(), "terrain_obstruction", first_start, last_end,
                "HORIZON_SECTOR", start, (start + width) % 360.0, 0.0, max_alt,
                None, 1.0, False, True, 1.0, 1.0, 1.0, 1.0,
            )
        )

    # Instrument faults form a chronological, non-overlapping event sequence. Each
    # event persists until the next one begins (or survey end); the scorer keeps
    # independent repair state for each event.
    rng = random.Random(stream_seed(config, "v4.instrument_fault", 2400))
    fault = config["instrument_fault"]
    fault_count = int(fault["count"])
    if fault_count > len(slots):
        raise ValueError("instrument_fault.count exceeds the number of observing slots")
    fault_proposals = []
    for occurrence in range(fault_count):
        nominal = int((occurrence + 1) * len(slots) / (fault_count + 1))
        start_index = max(0, min(len(slots) - 1, nominal + rng.randint(-200, 200)))
        multiplier = rng.uniform(
            float(fault["instrument_efficiency_multiplier_range"][0]),
            float(fault["instrument_efficiency_multiplier_range"][1]),
        )
        fault_proposals.append((start_index, occurrence, multiplier))
    fault_proposals.sort()
    fault_starts = []
    for position, (proposed_index, _, multiplier) in enumerate(fault_proposals):
        earliest = fault_starts[-1][0] + 1 if fault_starts else 0
        latest = len(slots) - (fault_count - position)
        fault_starts.append((max(earliest, min(proposed_index, latest)), multiplier))
    for position, (start_index, multiplier) in enumerate(fault_starts):
        start_slot = slots[start_index]
        end = slots[fault_starts[position + 1][0]].timestamp_utc if position + 1 < fault_count else last_end
        events.append(
            V4Event(
                next_id(), "instrument_fault", start_slot.timestamp_utc, end,
                "ALL", None, None, None, None, None, 1.0, False, False,
                1.0, 1.0, 1.0, multiplier,
            )
        )

    return sorted(events, key=lambda item: (item.actual_start_utc, item.event_id)), effects


# --- stress-test events (MP-055) -----------------------------------------------------


def generate_stress_events(
    config: Mapping, events: Sequence[V4Event]
) -> list[dict[str, object]]:
    """Truth rows for the two gated stress events; empty when the switch is off.

    When `stress_tests.enabled` is false this consumes no randomness and produces
    nothing, keeping all artifacts byte-identical to the non-stress baseline. Each
    family draws from its own RNG stream so the two events stay independent.

    Both events are fully hidden from the agent: data_loss is disclosed at trigger
    time by the state_resync message itself (never a bulletin), and pointing_offset
    is never announced at all (2026-09-27 ruling) — the agent can only infer it from
    the systematic directional miss pattern of its assigned targets.

    Window convention (confirmed 2026-09-27): i, j are drawn uniformly on [0, 1],
    sorted, and redrawn while j - i > x; the invalidated action window is the half-open
    interval [floor(i·N), floor(j·N)) of the N actions executed at trigger time.
    """
    stress = config.get("stress_tests")
    if not stress or not stress["enabled"]:
        return []
    rows: list[dict[str, object]] = []

    data_loss = stress["data_loss"]
    trigger = str(data_loss["trigger"])
    trigger_ref: str | None = None
    if trigger == "after_earthquake":
        quakes = [event for event in events if event.event_type == "earthquake"]
        if quakes:
            trigger_ref = quakes[0].event_id
        else:
            warnings.warn(
                "stress_tests.data_loss trigger is after_earthquake but the survey has "
                "no earthquake events; skipping data_loss generation"
            )
    elif trigger == "fixed_slot":
        trigger_ref = str(data_loss["slot_id"])
    else:
        trigger_ref = str(data_loss["date"])
    if trigger_ref is not None:
        rng = random.Random(stream_seed(config, "v4.stress.data_loss", 3100))
        limit = float(data_loss["window_max_fraction"])
        while True:
            start_fraction, end_fraction = sorted(rng.uniform(0.0, 1.0) for _ in range(2))
            if end_fraction - start_fraction <= limit:
                break
        rows.append(
            {
                "event_id": "V4ST0001",
                "event_type": "data_loss",
                "trigger_mode": trigger,
                "trigger_ref": trigger_ref,
                "window_start_fraction": f"{start_fraction:.6f}",
                "window_end_fraction": f"{end_fraction:.6f}",
                "window_max_fraction": f"{limit:.6f}",
                "alt_offset_rad": "",
                "az_offset_rad": "",
            }
        )

    offset = stress["pointing_offset"]
    rng = random.Random(stream_seed(config, "v4.stress.pointing_offset", 3200))
    alt_offset = rng.uniform(-float(offset["max_abs_alt_rad"]), float(offset["max_abs_alt_rad"]))
    az_offset = rng.uniform(-float(offset["max_abs_az_rad"]), float(offset["max_abs_az_rad"]))
    rows.append(
        {
            "event_id": "V4ST0002",
            "event_type": "pointing_offset",
            "trigger_mode": "at_survey_start",
            "trigger_ref": "",
            "window_start_fraction": "",
            "window_end_fraction": "",
            "window_max_fraction": "",
            "alt_offset_rad": f"{alt_offset:.8f}",
            "az_offset_rad": f"{az_offset:.8f}",
        }
    )
    return rows


# --- per-slot weather truth -----------------------------------------------------------


def _clip(value: float, model: Mapping) -> float:
    return max(float(model["minimum"]), min(float(model["maximum"]), value))


def generate_weather(
    config: Mapping,
    nights: Sequence[Night],
    slots: Sequence[Slot],
    events: Sequence[V4Event],
    effects: Sequence[EarthquakeEffect],
) -> list[dict[str, object]]:
    """v3 background quality model with the v4 global events folded into the truth.

    Directional (HORIZON_SECTOR) events stay in the event table and are applied by the
    scorer per pointing; this per-slot truth carries only global effects.
    """
    rng = random.Random(stream_seed(config, "v4.weather.slots", 1000))
    quality = config["quality"]
    fields = ("seeing_arcsec", "transparency", "sky_quality")
    night_state = {field: float(quality[field]["nominal"]) for field in fields}
    efficiency_model = quality["instrument_efficiency"]
    jitter = (float(efficiency_model["jitter_minimum"]), float(efficiency_model["jitter_maximum"]))
    by_night: defaultdict[str, list[Slot]] = defaultdict(list)
    for slot in slots:
        by_night[slot.night_id].append(slot)
    effects_by_night: defaultdict[str, list[EarthquakeEffect]] = defaultdict(list)
    for effect in effects:
        effects_by_night[effect.night_id].append(effect)
    quake_start_by_id = {
        event.event_id: event.actual_start_utc
        for event in events if event.event_type == "earthquake"
    }
    global_events = [event for event in events if event.scope_type == "ALL"]

    rows: list[dict[str, object]] = []
    for night in nights:
        night_slots = by_night[night.night_id]
        day_number = night.night_date.timetuple().tm_yday
        phase = 2 * math.pi * (day_number - int(quality["seasonal_phase_day"])) / 365.2425
        for field in fields:
            model = quality[field]
            direction = 1.0 if field == "seeing_arcsec" else -1.0
            mean = float(model["nominal"]) + direction * float(model["seasonal_amplitude"]) * math.sin(phase)
            phi = float(quality["night_correlation"])
            night_state[field] = _clip(
                mean + phi * (night_state[field] - mean)
                + rng.gauss(0.0, float(model["night_sigma"]) * math.sqrt(1 - phi**2)),
                model,
            )
        slot_state = dict(night_state)
        background_open = True
        closure = config["background_closure"]
        start_probability = float(closure["start_probability_per_open_slot"]) + float(
            closure["seasonal_probability_amplitude"]
        ) * max(0.0, math.sin(phase))
        quake_effects = effects_by_night.get(night.night_id, [])
        for slot in night_slots:
            phi = float(quality["slot_correlation"])
            for field in fields:
                model = quality[field]
                slot_state[field] = _clip(
                    night_state[field] + phi * (slot_state[field] - night_state[field])
                    + rng.gauss(0.0, float(model["slot_sigma"]) * math.sqrt(1 - phi**2)),
                    model,
                )
            if background_open and rng.random() < start_probability:
                background_open = False
            elif not background_open and rng.random() < float(closure["reopen_probability_per_closed_slot"]):
                background_open = True
            active = [
                event for event in global_events
                if event.event_type != "earthquake"
                and event.overlaps(slot.timestamp_utc, slot.end_utc)
            ]
            observable = background_open and not any(event.force_close for event in active)
            if not observable:
                rows.append(
                    {
                        "slot_id": slot.slot_id,
                        "night_id": slot.night_id,
                        "timestamp_utc": format_utc(slot.timestamp_utc),
                        "duration_seconds": slot.duration_seconds,
                        "is_observable": "false",
                        "seeing_arcsec": "",
                        "transparency": "",
                        "sky_quality": "",
                        "instrument_efficiency": "",
                    }
                )
                continue
            values = dict(slot_state)
            efficiency = rng.uniform(*jitter)
            for event in active:
                values["seeing_arcsec"] *= event.seeing_multiplier
                values["transparency"] *= event.transparency_multiplier
                values["sky_quality"] *= event.sky_quality_multiplier
                efficiency *= event.instrument_efficiency_multiplier
            for effect in quake_effects:
                if slot.timestamp_utc < quake_start_by_id[effect.event_id]:
                    continue
                values["seeing_arcsec"] *= effect.seeing_multiplier
                values["transparency"] *= effect.transparency_multiplier
                values["sky_quality"] *= effect.sky_quality_multiplier
                efficiency *= effect.instrument_efficiency_multiplier
            rows.append(
                {
                    "slot_id": slot.slot_id,
                    "night_id": slot.night_id,
                    "timestamp_utc": format_utc(slot.timestamp_utc),
                    "duration_seconds": slot.duration_seconds,
                    "is_observable": "true",
                    "seeing_arcsec": f"{_clip(values['seeing_arcsec'], quality['seeing_arcsec']):.6f}",
                    "transparency": f"{_clip(values['transparency'], quality['transparency']):.6f}",
                    "sky_quality": f"{_clip(values['sky_quality'], quality['sky_quality']):.6f}",
                    "instrument_efficiency": f"{_clip(efficiency, quality['instrument_efficiency']):.6f}",
                }
            )
    return rows


# --- agent-side publications -----------------------------------------------------------


def _notice(event: V4Event) -> dict[str, object]:
    return {"event_kind": COARSE_KIND[event.event_type], "direction": event_direction(event)}


def generate_bulletins(
    config: Mapping,
    nights: Sequence[Night],
    slots: Sequence[Slot],
    events: Sequence[V4Event],
    effects: Sequence[EarthquakeEffect],
) -> list[str]:
    """One bulletin per decision point (slot): active coarse event kinds + directions.

    The first slot's bulletin additionally carries the permanent terrain-obstruction
    sectors. Earthquakes appear while their degradation remains non-negligible.
    Instrument faults never appear.
    """
    publishable = [event for event in events if event.event_type not in BULLETIN_HIDDEN_TYPES]
    terrain = [event for event in publishable if event.event_type == "terrain_obstruction"]
    timed = [event for event in publishable if event.event_type != "terrain_obstruction"]
    effects_by_night: defaultdict[str, list[EarthquakeEffect]] = defaultdict(list)
    for effect in effects:
        effects_by_night[effect.night_id].append(effect)
    quake_events = {event.event_id: event for event in timed if event.event_type == "earthquake"}

    lines: list[str] = []
    for slot in slots:
        notices: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()

        def add(kind: str, direction: str) -> None:
            if (kind, direction) not in seen:
                seen.add((kind, direction))
                notices.append({"event_kind": kind, "direction": direction})

        # "initial" marks the survey's opening bulletin: the one-time terrain
        # obstruction announcement rides only on this record, later bulletins
        # never repeat it.
        initial = slot.slot_id == slots[0].slot_id
        if initial:
            for event in terrain:
                add(COARSE_KIND["terrain_obstruction"], event_direction(event))
        for event in timed:
            if event.event_type == "earthquake":
                continue  # handled through the decaying effect table below
            if event.overlaps(slot.timestamp_utc, slot.end_utc):
                add(COARSE_KIND[event.event_type], event_direction(event))
        for effect in effects_by_night.get(slot.night_id, []):
            event = quake_events[effect.event_id]
            if slot.timestamp_utc >= event.actual_start_utc:
                add(COARSE_KIND["earthquake"], event_direction(event))
        record = {
            "record_type": "bulletin",
            "slot_id": slot.slot_id,
            "night_id": slot.night_id,
            "issued_at_utc": format_utc(slot.timestamp_utc),
            "initial": initial,
            "notices": notices,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def generate_forecasts(
    config: Mapping, nights: Sequence[Night], events: Sequence[V4Event]
) -> list[str]:
    """Periodic coarse forecasts: forecastable events only, night-granularity timing."""
    publication = config["publication"]
    interval = int(publication["forecast_interval_days"])
    horizon = int(publication["forecast_horizon_days"])
    forecastable = [event for event in events if event.event_type in FORECASTABLE_TYPES]
    night_by_date = {night.night_date: night for night in nights}
    lines: list[str] = []
    issue_date = nights[0].night_date
    last_date = nights[-1].night_date
    while issue_date <= last_date:
        night = night_by_date.get(issue_date)
        if night is not None:
            issued = night.observing_start_utc
            coverage_end = issued + timedelta(days=horizon)
            notices = []
            for event in sorted(forecastable, key=lambda item: (item.actual_start_utc, item.event_id)):
                if not event.overlaps(issued, coverage_end):
                    continue
                # Only nights inside this forecast's coverage window are named: a long event
                # must not reveal nights beyond the horizon (integration hardening R6).
                touched = sorted(
                    {
                        candidate.night_date.isoformat()
                        for candidate in nights
                        if candidate.observing_start_utc < coverage_end
                        and candidate.observing_end_utc > issued
                        and event.overlaps(candidate.observing_start_utc, candidate.observing_end_utc)
                    }
                )
                if not touched:
                    continue
                notices.append(
                    {
                        "event_kind": COARSE_KIND[event.event_type],
                        "direction": event_direction(event),
                        "nights": touched,
                    }
                )
            record = {
                "record_type": "forecast",
                "issued_at_utc": format_utc(issued),
                "coverage_start_utc": format_utc(issued),
                "coverage_end_utc": format_utc(coverage_end),
                "notices": notices,
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        issue_date += timedelta(days=interval)
    return lines


# --- assembly ---------------------------------------------------------------------------


def generate(config_path: Path, output_dir: Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    if output_dir is None:
        output_dir = (config_path.parent / str(config["output"]["directory"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nights, slots = build_nights(config)
    events, effects = generate_events(config, nights, slots)
    weather_rows = generate_weather(config, nights, slots, events, effects)
    stress_rows = generate_stress_events(config, events)
    bulletin_lines = generate_bulletins(config, nights, slots, events, effects)
    forecast_lines = generate_forecasts(config, nights, events)

    paths = {
        "night_calendar": output_dir / "v4_night_calendar.csv",
        "slots": output_dir / "v4_slots.csv",
        "weather_truth": output_dir / "v4_weather_truth.csv",
        "events": output_dir / "v4_events.csv",
        "earthquake_effects": output_dir / "v4_earthquake_effects.csv",
        "bulletins": output_dir / "v4_bulletins.jsonl",
        "forecasts": output_dir / "v4_forecasts.jsonl",
    }
    if stress_rows:
        paths["stress_events"] = output_dir / "v4_stress_events.csv"
        paths["state_resync_schema"] = output_dir / "v4_state_resync_schema.json"
    write_exact_csv(paths["night_calendar"], NIGHT_COLUMNS, (night.csv_row() for night in nights))
    write_exact_csv(paths["slots"], SLOT_COLUMNS, (slot.csv_row() for slot in slots))
    write_exact_csv(paths["weather_truth"], WEATHER_COLUMNS, weather_rows)
    write_exact_csv(paths["events"], V4_EVENT_COLUMNS, (event.csv_row() for event in events))
    write_exact_csv(
        paths["earthquake_effects"],
        V4_EARTHQUAKE_EFFECT_COLUMNS,
        (effect.csv_row() for effect in effects),
    )
    write_text_lf(paths["bulletins"], "\n".join(bulletin_lines) + "\n")
    write_text_lf(paths["forecasts"], "\n".join(forecast_lines) + "\n")
    if stress_rows:
        write_exact_csv(paths["stress_events"], V4_STRESS_EVENT_COLUMNS, stress_rows)
        write_text_lf(
            paths["state_resync_schema"],
            json.dumps(STATE_RESYNC_SCHEMA, indent=2, sort_keys=True) + "\n",
        )

    type_counts: dict[str, int] = {}
    for event in events:
        type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
    summary = {
        "schema_version": "v4-weather-summary-v1",
        "generator": "challenge/v4_weather_simulator.py",
        "survey": {
            "start_date": str(config["survey"]["start_date"]),
            "end_date": str(config["survey"]["end_date"]),
            "night_count": len(nights),
            "slot_count": len(slots),
            "slot_seconds": int(config["survey"]["slot_seconds"]),
        },
        "events": {
            "total": len(events),
            "by_type": dict(sorted(type_counts.items())),
            "earthquakes": [
                {
                    "event_id": event.event_id,
                    "magnitude": round(float(event.magnitude), 3),
                    "initial_degradation": round(
                        earthquake_initial_degradation(float(event.magnitude), config), 6
                    ),
                    "affected_nights": sum(1 for item in effects if item.event_id == event.event_id),
                }
                for event in events
                if event.event_type == "earthquake"
            ],
            "terrain_sectors": [
                {
                    "event_id": event.event_id,
                    "azimuth_start_deg": round(float(event.azimuth_start_deg), 3),
                    "azimuth_end_deg": round(float(event.azimuth_end_deg), 3),
                    "max_altitude_deg": round(float(event.max_altitude_deg), 3),
                    "direction": event_direction(event),
                }
                for event in events
                if event.event_type == "terrain_obstruction"
            ],
            "rocket_launch_count": type_counts.get("rocket_launch", 0),
        },
        "weather": {
            "closed_slot_count": sum(1 for row in weather_rows if row["is_observable"] == "false"),
        },
        "publications": {
            "bulletin_count": len(bulletin_lines),
            "forecast_count": len(forecast_lines),
            "contract": (
                "notices carry only coarse event_kind plus an 8-point compass direction "
                "(or ALL); no numeric weather quantities; earthquakes and instrument "
                "faults are never forecast; instrument faults are never published"
            ),
        },
        "sha256": {"config": sha256_file(config_path)}
        | {name: sha256_file(path) for name, path in paths.items()},
    }
    if stress_rows:
        summary["stress_tests"] = {
            "enabled": True,
            "events": stress_rows,
            "state_resync_schema": "v4_state_resync_schema.json",
            "window_convention": (
                "0-based indices over observe actions only (wait excluded); "
                "invalidated window [floor(i*N), floor(j*N)); "
                "width floor(j*N)-floor(i*N) <= ceil(x*N)"
            ),
        }
    write_text_lf(
        output_dir / "v4_weather_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="v4 weather config JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the output directory named in the config",
    )
    args = parser.parse_args()
    summary = generate(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
