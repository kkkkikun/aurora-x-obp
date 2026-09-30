#!/usr/bin/env python3
"""v4 configuration cross-checks (platform hardening, not part of the v4 prototype).

Two layers:

* ``cross_validate_generator_configs`` checks that the configs used to *build* one task
  card agree with each other (site, survey dates, twilight limit, altitude limit, stress
  switch) before anything is generated. Competition cards call it with
  ``require_hashed_seeds=True`` so every RNG stream is seeded through sha256.
* ``validate_scenario`` checks a loaded scenario (what the runner actually scores) for
  internal consistency: site agreement between scenario and fibre config, sane fibre /
  exposure / score parameters, a clean catalogue, chronological non-overlapping slots
  and chronologically ordered publications (the runner's delivery cursor relies on it).

Pure standard library.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .contracts import SEED_DERIVATION_HASHED, SEED_DERIVATION_KEY

SITE_KEYS = ("latitude_deg", "longitude_deg", "utc_offset_hours")
PROGRAMS = ("DARK", "BRIGHT", "BACKUP")
SCORE_SCHEMA_VERSION = "v4-score-v1"
DEFAULT_FALSE_REPORT_FREE_ALLOWANCE = 0
DEFAULT_MAX_CONSECUTIVE_REPORTS = 32


def _site_tuple(site: Mapping, label: str) -> tuple[float, float, float]:
    try:
        return tuple(float(site[key]) for key in SITE_KEYS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: site must define {', '.join(SITE_KEYS)}") from exc


def _same_site(sites: Sequence[tuple[str, Mapping]]) -> None:
    reference_label, reference = sites[0]
    expected = _site_tuple(reference, reference_label)
    for label, site in sites[1:]:
        if _site_tuple(site, label) != expected:
            raise ValueError(f"site mismatch between {reference_label} and {label}")


def cross_validate_generator_configs(
    catalog: Mapping,
    weather: Mapping,
    scenario: Mapping | None = None,
    fiber: Mapping | None = None,
    *,
    require_hashed_seeds: bool = False,
) -> None:
    """Raise ValueError unless the generator configs of one card agree with each other."""
    sites = [("catalog", catalog["site"]), ("weather", weather["site"])]
    if scenario is not None:
        sites.append(("scenario", scenario["site"]))
    if fiber is not None:
        sites.append(("fiber", fiber["site"]))
    _same_site(sites)

    observability = catalog["observability"]
    survey = weather["survey"]
    for catalog_key, weather_key in (("start_date", "start_date"), ("end_date", "end_date")):
        if str(observability[catalog_key]) != str(survey[weather_key]):
            raise ValueError(
                f"catalog observability.{catalog_key} differs from weather survey.{weather_key}"
            )
    if float(observability["sun_altitude_limit_deg"]) != float(survey["sun_altitude_limit_deg"]):
        raise ValueError("catalog and weather sun_altitude_limit_deg differ")
    if scenario is not None:
        if float(observability["minimum_altitude_deg"]) != float(scenario["minimum_altitude_deg"]):
            raise ValueError("catalog observability.minimum_altitude_deg differs from the scenario")
        stress_on = bool(scenario.get("stress", {}).get("enabled", False))
        weather_stress = weather.get("stress_tests") or {}
        if stress_on != bool(weather_stress.get("enabled", False)):
            raise ValueError("scenario stress.enabled differs from weather stress_tests.enabled")
    if fiber is not None:
        validate_fiber_config(fiber)
    if require_hashed_seeds:
        for label, config in (("catalog", catalog), ("weather", weather)):
            if config.get(SEED_DERIVATION_KEY) != SEED_DERIVATION_HASHED:
                raise ValueError(f"{label} config must set {SEED_DERIVATION_KEY}={SEED_DERIVATION_HASHED!r}")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_fiber_config(fiber: Mapping) -> None:
    field = fiber["field"]
    n_fibers = int(field["n_fibers"])
    side = math.isqrt(n_fibers)
    if n_fibers < 1 or side * side != n_fibers:
        raise ValueError("fiber n_fibers must be a positive perfect square")
    if _finite(field["fiber_area_deg2"], "fiber_area_deg2") <= 0.0:
        raise ValueError("fiber_area_deg2 must be positive")
    if _finite(field["gap_deg"], "gap_deg") < 0.0:
        raise ValueError("gap_deg must be non-negative")
    exposure = fiber["exposure"]
    low = int(exposure["min_duration_seconds"])
    high = int(exposure["max_duration_seconds"])
    if not 1 <= low <= high:
        raise ValueError("exposure bounds must satisfy 1 <= min_duration_seconds <= max_duration_seconds")


def validate_score_config(score: Mapping) -> None:
    if score.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported score schema_version")
    for key in ("q0", "flux_zero_point", "exposure_zero_point_seconds"):
        if _finite(score[key], key) <= 0.0:
            raise ValueError(f"score {key} must be positive")
    if _finite(score["airmass_exponent"], "airmass_exponent") < 0.0:
        raise ValueError("score airmass_exponent must be non-negative")
    program = score["program"]
    bands = program["bands"]
    if not float(bands["DARK"]) > float(bands["BRIGHT"]) >= 0.0:
        raise ValueError("program bands must satisfy DARK > BRIGHT >= 0")
    if set(program["multipliers"]) != set(PROGRAMS):
        raise ValueError("program multipliers must define exactly DARK/BRIGHT/BACKUP")
    for name in PROGRAMS:
        if _finite(program["multipliers"][name], f"multiplier {name}") <= 0.0:
            raise ValueError("program multipliers must be positive")
    if _finite(program["mismatch_multiplier"], "mismatch_multiplier") < 0.0:
        raise ValueError("mismatch_multiplier must be non-negative")
    if _finite(score["required"]["penalty_per_missing"], "penalty_per_missing") < 0.0:
        raise ValueError("required penalty must be non-negative")
    for section in ("required", "uniformity"):
        if not 0.0 < _finite(score[section]["observed_factor_threshold"], "threshold") <= 1.0:
            raise ValueError(f"{section}.observed_factor_threshold must lie in (0, 1]")
    if _finite(score["uniformity"]["weight"], "uniformity weight") < 0.0:
        raise ValueError("uniformity weight must be non-negative")
    if _finite(score["uniformity"]["ra_band_width_deg"], "ra band width") <= 0.0:
        raise ValueError("uniformity ra_band_width_deg must be positive")
    reporting = score["reporting"]
    _finite(reporting["correct_reward"], "correct_reward")
    _finite(reporting["false_penalty"], "false_penalty")
    free_allowance = reporting.get("false_report_free_allowance", DEFAULT_FALSE_REPORT_FREE_ALLOWANCE)
    if isinstance(free_allowance, bool) or not isinstance(free_allowance, int) or free_allowance < 0:
        raise ValueError("reporting.false_report_free_allowance must be a non-negative integer")
    max_reports = reporting.get("max_consecutive_reports", DEFAULT_MAX_CONSECUTIVE_REPORTS)
    if isinstance(max_reports, bool) or not isinstance(max_reports, int) or max_reports < 1:
        raise ValueError("reporting.max_consecutive_reports must be a positive integer")


def validate_targets(targets: Sequence[Mapping]) -> None:
    if not targets:
        raise ValueError("scenario has no targets")
    seen: set[str] = set()
    for target in targets:
        target_id = str(target["target_id"])
        if not target_id or target_id in seen:
            raise ValueError(f"duplicate or empty target_id {target_id!r}")
        seen.add(target_id)
        if not 0.0 <= _finite(target["ra_deg"], "ra_deg") < 360.0:
            raise ValueError(f"target {target_id}: ra_deg outside [0, 360)")
        if not -90.0 <= _finite(target["dec_deg"], "dec_deg") <= 90.0:
            raise ValueError(f"target {target_id}: dec_deg outside [-90, 90]")
        if _finite(target["feature_flux"], "feature_flux") < 0.0:
            raise ValueError(f"target {target_id}: negative feature_flux")
        if _finite(target["science_weight"], "science_weight") < 0.0:
            raise ValueError(f"target {target_id}: negative science_weight")


def _chronological(records: Sequence[Mapping], parse, label: str) -> None:
    previous = None
    for record in records:
        moment = parse(record["issued_at_utc"])
        if previous is not None and moment < previous:
            raise ValueError(f"{label} are not in chronological order")
        previous = moment


def validate_scenario(scenario) -> None:
    """Consistency checks on a loaded ``v4_runner.Scenario``; raises ValueError."""
    from .v4_runner import _parse_utc  # local import: v4_runner imports this module

    config = scenario.config
    _same_site([("scenario", scenario.site), ("fiber config", scenario.fiber_config["site"])])
    if not 0.0 <= float(config["minimum_altitude_deg"]) < 90.0:
        raise ValueError("minimum_altitude_deg must lie in [0, 90)")
    validate_fiber_config(scenario.fiber_config)
    validate_score_config(scenario.score_config)
    validate_targets(scenario.targets)
    if not scenario.slots:
        raise ValueError("scenario has no slots")
    for left, right in zip(scenario.slots, scenario.slots[1:]):
        if right.start_utc < left.end_utc:
            raise ValueError(f"slots overlap or are out of order at {right.slot_id}")
    for slot in scenario.slots:
        if slot.end_utc <= slot.start_utc:
            raise ValueError(f"slot {slot.slot_id} has non-positive duration")
    for event in scenario.events:
        if event.actual_end_utc < event.actual_start_utc:
            raise ValueError(f"event {event.event_id} ends before it starts")
        if event.scope_type == "HORIZON_SECTOR" and None in (
            event.azimuth_start_deg, event.azimuth_end_deg,
            event.min_altitude_deg, event.max_altitude_deg,
        ):
            raise ValueError(f"directional event {event.event_id} lacks its sector")
    _chronological(scenario.bulletins, _parse_utc, "bulletins")
    _chronological(scenario.forecasts, _parse_utc, "forecasts")
    if scenario.stress_enabled and not scenario.stress_rows:
        raise ValueError("stress is enabled but the stress events table is empty")
