"""Tests for the v4 scorer and runner (MP-057)."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from challenge import v4_scorer as vs
from challenge import v4_runner as vr
from challenge.v4_fiber_map import FiberGrid, min_altitude_during, radec_to_altaz
from challenge.tile_geometry_simulator import Tile, geometry_sample


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "v4"
DEFAULT_SCENARIO = REFERENCE / "v4_scenario_default.json"
STRESS_SCENARIO = REFERENCE / "v4_scenario_stress.json"


@pytest.fixture(scope="module", autouse=True)
def _reference_scenarios(v4_reference_dir):
    """Point the scenario paths at the generated reference products (see conftest)."""
    global DEFAULT_SCENARIO, STRESS_SCENARIO
    DEFAULT_SCENARIO = v4_reference_dir / "v4_scenario_default.json"
    STRESS_SCENARIO = v4_reference_dir / "v4_scenario_stress.json"


T0 = datetime(2026, 10, 1, 23, 45, 0, tzinfo=timezone.utc)

SCORE_CONFIG = {
    "q0": 0.68,
    "flux_zero_point": 0.5,
    "exposure_zero_point_seconds": 900,
    "airmass_exponent": 0.6,
    "program": {
        "bands": {"DARK": 0.65, "BRIGHT": 0.40},
        "multipliers": {"DARK": 1.20, "BRIGHT": 1.12, "BACKUP": 1.06},
        "mismatch_multiplier": 1.0,
    },
    "required": {"penalty_per_missing": 50, "observed_factor_threshold": 0.5},
    "uniformity": {"weight": 200.0, "ra_band_width_deg": 10.0, "observed_factor_threshold": 0.5},
    "reporting": {"correct_reward": 100, "false_penalty": -150,
                  "false_report_free_allowance": 2, "max_consecutive_reports": 32},
}

SITE = {"latitude_deg": -24.6157, "longitude_deg": -70.3976}
LUNAR_MODEL = {"angular_decay_scale_deg": 35.0, "altitude_exponent": 1.0, "maximum_penalty": 0.75}


def _slot(minutes: int, duration: int = 900, observable: bool = True, **values) -> vs.SlotTruth:
    start = T0 + timedelta(seconds=minutes * 60)
    return vs.SlotTruth(
        f"S{minutes:03d}", start, start + timedelta(seconds=duration), observable,
        values.get("seeing", 1.0), values.get("transp", 0.9), values.get("sky", 1.0),
        values.get("eff", 1.0),
    )


def _event(event_type, start_min, end_min, az=(0.0, 0.0), alt=(0.0, 90.0), scope="HORIZON_SECTOR", **mults) -> vs.ScoreEvent:
    return vs.ScoreEvent(
        "EV1", event_type, T0 + timedelta(minutes=start_min), T0 + timedelta(minutes=end_min),
        scope, az[0] if scope != "ALL" else None, az[1] if scope != "ALL" else None,
        alt[0] if scope != "ALL" else None, alt[1] if scope != "ALL" else None,
        mults.get("seeing", 1.0), mults.get("transp", 1.0), mults.get("sky", 1.0),
        mults.get("eff", 1.0),
    )


def _target(tid="T1", flux=0.5, weight=1.0, ra=60.0, dec=-40.0, required=False):
    return {
        "target_id": tid, "ra_deg": ra, "dec_deg": dec, "target_class": "ELG",
        "feature_flux": flux, "science_weight": weight, "required": required,
    }


# --- formula units -----------------------------------------------------------------------


def test_segments_weighting_and_closure():
    slots = [_slot(0), _slot(15, seeing=2.0), _slot(30, observable=False), _slot(45)]
    weather = vs.WeatherTruth(slots, None)
    segments = list(weather.segments(T0, T0 + timedelta(minutes=60)))
    assert len(segments) == 4
    assert all(seconds == 900 for seconds, _ in segments)
    assert segments[2][1].is_observable is False
    # Band quality: transparency/(sky*seeing) weighted; the closed slot contributes 0.
    q = weather.band_quality(T0, T0 + timedelta(minutes=60), SCORE_CONFIG, lambda _: 90.0)
    expected = ((0.9 / 1.0) + (0.9 / 2.0) + 0.0 + (0.9 / 1.0)) / 4 / SCORE_CONFIG["q0"]
    assert q == pytest.approx(expected, rel=1e-9)


def test_daytime_gap_has_no_weather():
    weather = vs.WeatherTruth([_slot(0), _slot(720)], None)
    start = T0 + timedelta(hours=1)
    end = start + timedelta(minutes=15)
    assert list(weather.segments(start, end)) == [(900.0, None)]
    crossing = list(weather.segments(T0 + timedelta(hours=11, minutes=55),
                                     T0 + timedelta(hours=12, minutes=5)))
    assert crossing == [(300.0, None), (300.0, weather.slots[1])]


def test_program_band_and_multiplier():
    assert vs.program_band(0.70, SCORE_CONFIG) == "DARK"
    assert vs.program_band(0.50, SCORE_CONFIG) == "BRIGHT"
    assert vs.program_band(0.10, SCORE_CONFIG) == "BACKUP"
    assert vs.program_band(0.65, SCORE_CONFIG) == "DARK"
    assert vs.program_band(0.40, SCORE_CONFIG) == "BRIGHT"
    assert vs.program_multiplier("DARK", "DARK", SCORE_CONFIG) == 1.20
    assert vs.program_multiplier("BRIGHT", "BRIGHT", SCORE_CONFIG) == 1.12
    assert vs.program_multiplier("BACKUP", "BACKUP", SCORE_CONFIG) == 1.06
    # Mismatch degrades to 1.0 (no extra penalty beyond losing the bonus).
    assert vs.program_multiplier("DARK", "BACKUP", SCORE_CONFIG) == 1.0
    with pytest.raises(ValueError):
        vs.program_multiplier("LUX", "DARK", SCORE_CONFIG)


def _score(flux=0.5, events=(), slots=None, mult=1.0, weight=1.0, alt=60.0, az=180.0,
           duration=900, trajectory=None):
    slots = slots or [_slot(0)]
    weather = vs.WeatherTruth(slots, None)
    segments = list(weather.segments(T0, T0 + timedelta(seconds=duration)))
    # The scorer expects the caller to pass only events overlapping the exposure.
    end = T0 + timedelta(seconds=duration)
    active = [event for event in events if event.overlaps(T0, end)]
    trajectory = trajectory or (lambda _: (alt, az))
    return vs.score_target_exposure(
        _target(flux=flux, weight=weight), segments, active, weather,
        duration, mult, SCORE_CONFIG, T0, trajectory,
    )


def test_factor_cap_and_weight():
    bright = _score(flux=5.0)
    assert bright.factor == 1.0  # capped
    assert bright.score == pytest.approx(1.0 * 1.0 * 1.0)
    faint = _score(flux=0.1)
    assert faint.factor < 1.0
    weighted = _score(flux=5.0, weight=1.7, mult=1.2)
    assert weighted.score == pytest.approx(1.7 * 1.0 * 1.2)


def test_sky_quality_higher_is_better():
    clear = _score(flux=0.1, slots=[_slot(0, sky=1.2)])
    poor = _score(flux=0.1, slots=[_slot(0, sky=0.8)])
    assert clear.quality > poor.quality
    cloudy = _event("cloudy", 0, 15, az=(170.0, 190.0), sky=0.5)
    assert _score(flux=0.1, events=[cloudy]).quality == pytest.approx(_score(flux=0.1).quality * 0.5)


def test_target_lunar_factor_matches_published_starter_model():
    moment = datetime(2026, 10, 24, 0, tzinfo=timezone.utc)
    moon_ra, moon_dec, illumination, moon_alt = vs._lunar_geometry(
        moment, SITE["latitude_deg"], SITE["longitude_deg"]
    )
    assert illumination > 0.9 and moon_alt > 40.0
    config = {**SCORE_CONFIG, "lunar_model": LUNAR_MODEL}
    vs.validate_lunar_model(config)
    for ra, dec in ((moon_ra, moon_dec), ((moon_ra + 90.0) % 360.0, moon_dec)):
        tile = Tile("X", ra, dec, 900, "R", "DARK", moment, moment + timedelta(days=1))
        old_factor = geometry_sample(tile, moment, {"lunar_model": LUNAR_MODEL}, {"site": SITE})[
            "lunar_quality_factor"
        ]
        assert vs.lunar_quality_factor(ra, dec, moment, SITE, config) == pytest.approx(
            old_factor, abs=1e-12
        )
    near = vs.lunar_quality_factor(moon_ra, moon_dec, moment, SITE, config)
    far = vs.lunar_quality_factor((moon_ra + 90.0) % 360.0, moon_dec, moment, SITE, config)
    assert near < far < 1.0
    disabled = {**config, "lunar_model": {**LUNAR_MODEL, "maximum_penalty": 0.0}}
    assert vs.lunar_quality_factor(moon_ra, moon_dec, moment, SITE, disabled) == 1.0


def test_lunar_factor_applies_per_target_and_program_center():
    moment = datetime(2026, 10, 24, 0, tzinfo=timezone.utc)
    moon_ra, moon_dec, _, _ = vs._lunar_geometry(moment, SITE["latitude_deg"], SITE["longitude_deg"])
    config = {**SCORE_CONFIG, "lunar_model": LUNAR_MODEL}
    weather = vs.WeatherTruth([vs.SlotTruth("S", moment, moment + timedelta(seconds=900), True,
                                            1.0, 0.9, 1.0, 1.0)], None)
    segments = list(weather.segments(moment, moment + timedelta(seconds=900)))

    def scored(ra):
        target = _target(flux=0.1, ra=ra, dec=moon_dec)
        return vs.score_target_exposure(
            target, segments, [], weather, 900, 1.0,
            config, moment, lambda _: (60.0, 180.0), site=SITE,
        )

    assert scored(moon_ra).quality < scored((moon_ra + 90.0) % 360.0).quality
    near_band = weather.band_quality(
        moment, moment + timedelta(seconds=900), config, lambda _: 90.0,
        lambda when: vs.lunar_quality_factor(moon_ra, moon_dec, when, SITE, config),
    )
    far_band = weather.band_quality(
        moment, moment + timedelta(seconds=900), config, lambda _: 90.0,
        lambda when: vs.lunar_quality_factor((moon_ra + 90.0) % 360.0, moon_dec, when, SITE, config),
    )
    assert near_band < far_band


def test_directional_event_applies_per_target():
    cloudy = _event("cloudy", 0, 60, az=(170.0, 190.0), alt=(50.0, 70.0), seeing=2.0, transp=0.5)
    with_event = _score(events=[cloudy])
    without = _score()
    assert with_event.quality < without.quality
    # Target outside the sector is unaffected.
    elsewhere = _score(events=[cloudy], az=10.0)
    assert elsewhere.quality == pytest.approx(without.quality)


def test_terrain_obstruction_zero_score():
    obstruction = _event("terrain_obstruction", -60, 600, az=(170.0, 190.0), alt=(0.0, 40.0))
    # Target above the obstruction altitude cap is fine.
    assert _score(events=[obstruction], alt=60.0).factor > 0.0
    # Target inside the sector scores zero.
    assert _score(events=[obstruction], alt=30.0, az=180.0).factor == 0.0


def test_rocket_launch_force_close():
    rocket = _event("rocket_launch", 0, 5, az=(170.0, 190.0), alt=(0.0, 90.0))
    # Five closed minutes lose five minutes of signal, not the whole exposure.
    assert _score(flux=0.1, events=[rocket]).factor == pytest.approx(_score(flux=0.1).factor * 2 / 3)
    whole = _event("rocket_launch", 0, 15, az=(170.0, 190.0), alt=(0.0, 90.0))
    assert _score(events=[whole]).factor == 0.0
    # Outside the launch window the target scores normally.
    gone = _event("rocket_launch", -60, -30, az=(170.0, 190.0), alt=(0.0, 90.0))
    assert _score(events=[gone]).factor > 0.0


def test_directional_weather_uses_only_its_active_minutes():
    cloudy = _event("cloudy", 5, 10, az=(170.0, 190.0), alt=(0.0, 90.0), seeing=2.0)
    assert _score(events=[cloudy]).quality == pytest.approx(_score().quality * (5 / 6))


def test_directional_weather_follows_target_position_during_exposure():
    cloudy = _event("cloudy", 0, 15, az=(0.0, 90.0), alt=(0.0, 90.0), seeing=2.0)
    trajectory = lambda moment: (60.0, 45.0 if moment < T0 + timedelta(minutes=8) else 180.0)
    assert _score(events=[cloudy], trajectory=trajectory).quality == pytest.approx(
        _score().quality * 11 / 15
    )


def test_azimuth_wrap_sector():
    assert vs.azimuth_inside(359.0, 350.0, 10.0)
    assert vs.azimuth_inside(5.0, 350.0, 10.0)
    assert not vs.azimuth_inside(20.0, 350.0, 10.0)
    wrapped = _event("cloudy", 0, 60, az=(350.0, 10.0), seeing=2.0)
    inside = _score(events=[wrapped], az=359.0)
    outside = _score(events=[wrapped], az=180.0)
    assert inside.quality < outside.quality


def test_bad_weather_truth_degrades_quality():
    calm = _slot(0)
    poor = _slot(0, seeing=2.5, sky=0.6, transp=0.5, eff=0.48)
    assert _score(slots=[poor]).quality < _score(slots=[calm]).quality


def test_best_ledger_max_and_invalidation():
    ledger = vs.BestLedger()
    ledger.record(0, "T1", 0.4, 0.4)
    ledger.record(1, "T1", 0.8, 0.9)
    ledger.record(1, "T2", 0.5, 0.5)
    assert ledger.best() == {"T1": (0.8, 0.9), "T2": (0.5, 0.5)}
    invalidated = ledger.invalidate_window(1, 2, "V4ST0001")
    assert invalidated == 2
    assert ledger.best() == {"T1": (0.4, 0.4)}  # T2 fully invalidated, T1 falls back
    assert ledger.max_factors() == {"T1": 0.4}


def test_required_penalty():
    targets = [_target("A", required=True), _target("B", required=True), _target("C")]
    factors = {"A": 0.9}
    missing, cost = vs.required_penalty(targets, factors, SCORE_CONFIG)
    assert (missing, cost) == (1, 50.0)  # B below threshold (unobserved); C not required


def test_completion_uses_any_valid_factor_not_best_scoring_exposure():
    ledger = vs.BestLedger()
    ledger.record(0, "T1", 0.51, 0.51)
    ledger.record(1, "T1", 0.49, 0.588)
    assert ledger.best()["T1"] == (0.49, 0.588)
    assert vs.required_penalty([_target("T1", required=True)], ledger.max_factors(), SCORE_CONFIG) == (0, 0.0)


def test_uniformity_penalty_jain():
    targets = [_target(f"T{i}", ra=5.0 + 10.0 * (i % 2)) for i in range(4)]
    full = {t["target_id"]: 0.9 for t in targets}
    ratios, cost = vs.uniformity_penalty(targets, full, SCORE_CONFIG)
    assert cost == pytest.approx(0.0)
    _, cost_none = vs.uniformity_penalty(targets, {}, SCORE_CONFIG)
    assert cost_none == pytest.approx(200.0)
    # Band 0 fully covered, band 1 empty: Jain = 0.5.
    half = {targets[0]["target_id"]: 0.9, targets[2]["target_id"]: 0.9}
    _, cost_half = vs.uniformity_penalty(targets, half, SCORE_CONFIG)
    assert cost_half == pytest.approx(100.0)


def test_report_settlement_and_repair():
    fault = vs.ScoreEvent(
        "EV9", "instrument_fault", T0 - timedelta(days=1), T0 + timedelta(days=30),
        "ALL", None, None, None, None, 1.0, 1.0, 1.0, 0.6,
    )
    weather = vs.WeatherTruth([_slot(0, eff=0.6)], fault)
    first = vs.settle_reports([T0], weather, SCORE_CONFIG)
    assert first == 100.0
    # After repair the fault multiplier is undone in the site factor.
    assert weather._apply_fault(weather.slots[0], T0 + timedelta(minutes=1)) == pytest.approx(1.0)
    assert vs.settle_reports([T0], weather, SCORE_CONFIG) == 0.0  # first false report is free
    no_fault = vs.WeatherTruth([_slot(0)], None)
    assert vs.settle_reports([T0, T0], no_fault, SCORE_CONFIG) == 0.0
    assert vs.settle_reports([T0], no_fault, SCORE_CONFIG) == -150.0


def test_report_before_fault_is_false_and_later_report_repairs():
    fault = _event("instrument_fault", 10, 120, scope="ALL", eff=0.6)
    weather = vs.WeatherTruth([_slot(0, eff=0.6)], fault)
    assert weather.report_fault(T0, SCORE_CONFIG) == 0.0
    assert weather.fault_repair_utc is None
    after_start = T0 + timedelta(minutes=10)
    assert weather.report_fault(after_start, SCORE_CONFIG) == 100.0
    assert weather._apply_fault(weather.slots[0], after_start) == pytest.approx(1.0)
    assert weather.report_fault(after_start, SCORE_CONFIG) == 0.0  # correct report reset the false streak


def test_runner_report_repairs_before_next_exposure(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    fault = next(event for event in scenario.events if event.event_type == "instrument_fault")
    slot = next(item for item in scenario.slots if item.start_utc == fault.actual_start_utc)
    assert slot.is_observable
    short = replace(scenario, slots=[slot], survey_start=slot.start_utc,
                    survey_end=slot.end_utc, bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    grid = FiberGrid.from_config(scenario.fiber_config)
    lat = float(scenario.site["latitude_deg"])
    lon = float(scenario.site["longitude_deg"])
    target = next(t for t in scenario.targets if t["feature_flux"] < 0.2 and
                  45.0 < radec_to_altaz(
                      t["ra_deg"], t["dec_deg"], slot.start_utc, lat, lon
                  )[0] < 65.0)
    alt, az = radec_to_altaz(target["ra_deg"], target["dec_deg"], slot.start_utc, lat, lon)
    d_alt, d_az = grid.fiber_center_offset(5)
    cmd_alt = alt - d_alt
    cmd_az = (az - d_az / math.cos(math.radians(cmd_alt))) % 360.0
    observe = {"action": "observe", "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
               "assignments": {5: target["target_id"]}, "duration_seconds": 900, "program": "BACKUP"}
    before = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([observe]), tmp_path / "fault")
    after = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([{"action": "report"}, observe]),
                            tmp_path / "repaired")
    assert before["components"]["sum_best_scores"] > 0.0
    assert after["components"]["sum_best_scores"] > before["components"]["sum_best_scores"]
    assert after["components"]["report_settlement"] == 100.0


def test_report_result_is_announced_immediately_and_repeated_report_is_false(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    fault = next(event for event in scenario.events if event.event_type == "instrument_fault")
    slot = next(item for item in scenario.slots if item.start_utc == fault.actual_start_utc)
    short = replace(scenario, slots=[slot], survey_start=slot.start_utc,
                    survey_end=slot.end_utc, bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    actions = iter([{"action": "report"}, {"action": "report"}])
    snapshots = []

    def factory(_context):
        def agent(snapshot):
            snapshots.append(snapshot)
            return next(actions, None)
        return agent

    out = tmp_path / "report_feedback"
    result = vr.run_scenario(DEFAULT_SCENARIO, factory, out)
    assert len(snapshots) == 3
    assert all(snapshot["now_utc"] == snapshots[0]["now_utc"] for snapshot in snapshots)
    assert snapshots[1]["last_result"] == {
        "action": "report", "correct": True, "repaired": True, "score_delta": 100.0,
    }
    assert snapshots[2]["last_result"] == {
        "action": "report", "correct": False, "repaired": False, "score_delta": 0.0,
    }
    assert [snapshot["new_messages"][0]["correct"] for snapshot in snapshots[1:]] == [True, False]
    assert all(snapshot["new_messages"][0]["record_type"] == "report_result"
               for snapshot in snapshots[1:])
    assert all(snapshot["latest_bulletin"] is None for snapshot in snapshots)
    assert result["components"]["report_settlement"] == 100.0
    messages = [json.loads(line) for line in (out / "messages.jsonl").read_text().splitlines()]
    assert messages == [snapshot["new_messages"][0] for snapshot in snapshots[1:]]


def test_runner_limits_target_altitude_without_a_field_center_cutoff(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    slot = next(item for item in scenario.slots if item.is_observable)
    short = replace(scenario, slots=[slot], survey_start=slot.start_utc,
                    survey_end=slot.end_utc, bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    grid = FiberGrid.from_config(scenario.fiber_config)
    lat = float(scenario.site["latitude_deg"])
    lon = float(scenario.site["longitude_deg"])
    target = next(
        t for t in scenario.targets
        if 30.1 < radec_to_altaz(t["ra_deg"], t["dec_deg"], slot.start_utc, lat, lon)[0] < 30.3
        and min_altitude_during(t["ra_deg"], t["dec_deg"], slot.start_utc,
                                slot.end_utc, scenario.config, 120) >= 30.0
    )
    def action(target, fiber_id):
        alt, az = radec_to_altaz(target["ra_deg"], target["dec_deg"], slot.start_utc, lat, lon)
        d_alt, d_az = grid.fiber_center_offset(fiber_id)
        cmd_alt = alt - d_alt
        cmd_az = (az - d_az / math.cos(math.radians(cmd_alt))) % 360.0
        return {"action": "observe", "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
                "assignments": {fiber_id: target["target_id"]},
                "duration_seconds": 900, "program": "BACKUP"}

    high_center = action(target, 5)
    low_center = action(target, 9)
    assert high_center["pointing"]["alt_deg"] > 30.0
    assert low_center["pointing"]["alt_deg"] < 30.0
    high = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([high_center]), tmp_path / "high_center")
    low = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([low_center]), tmp_path / "low_center")
    assert high["counts"]["observations"] == low["counts"]["observations"] == 1
    assert high["components"]["sum_best_scores"] == low["components"]["sum_best_scores"]

    below_target = next(
        t for t in scenario.targets
        if 29.6 < radec_to_altaz(t["ra_deg"], t["dec_deg"], slot.start_utc, lat, lon)[0] < 29.9
    )
    below_action = action(below_target, 5)
    assert below_action["pointing"]["alt_deg"] > 30.0
    below = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([below_action]), tmp_path / "low_target")
    assert below["counts"]["observations"] == 0


# --- runner-level -------------------------------------------------------------------------


def _fixed_observe_actions(scenario_path: Path, count: int, duration: int = 900):
    """Deterministic valid observe actions built from the scenario's real targets.

    Each action tracks the same anchor target (pointing recomputed at that action's
    start time), so every action has on-glass assignments regardless of sky rotation.
    """
    scenario = vr.load_scenario(scenario_path)
    grid = FiberGrid.from_config(scenario.fiber_config)
    lat = float(scenario.site["latitude_deg"])
    lon = float(scenario.site["longitude_deg"])
    start = scenario.survey_start
    anchor = next(
        t for t in scenario.targets
        if radec_to_altaz(t["ra_deg"], t["dec_deg"], start, lat, lon)[0] > 50.0
    )
    actions = []
    for index in range(count):
        moment = start + timedelta(seconds=index * duration)
        anchor_alt, anchor_az = radec_to_altaz(anchor["ra_deg"], anchor["dec_deg"], moment, lat, lon)
        cmd_alt = round(anchor_alt + grid.pitch_deg / 2.0, 4)
        cmd_az = round(anchor_az + (grid.pitch_deg / 2.0) / math.cos(math.radians(anchor_alt)), 4)
        cos_alt = math.cos(math.radians(cmd_alt))
        half = grid.fov_side_deg / 2.0
        assignments: dict[int, str] = {}
        for target in scenario.targets:
            if len(assignments) >= 12:
                break
            t_alt, t_az = radec_to_altaz(target["ra_deg"], target["dec_deg"], moment, lat, lon)
            d_alt = t_alt - cmd_alt
            d_az = ((t_az - cmd_az + 180.0) % 360.0 - 180.0) * cos_alt
            if abs(d_alt) > half or abs(d_az) > half:
                continue
            fiber_id, region = grid.classify_offset(d_alt, d_az)
            if region != "glass" or fiber_id in assignments:
                continue
            assignments[fiber_id] = target["target_id"]
        actions.append(
            {
                "action": "observe",
                "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
                "assignments": assignments,
                "duration_seconds": duration,
                "program": "BACKUP",
            }
        )
    return actions


def _scripted_factory(actions):
    iterator = iter(actions)

    def factory(context):
        def agent(snapshot):
            try:
                return next(iterator)
            except StopIteration:
                return None

        return agent

    return factory


@pytest.fixture(scope="module")
def small_default_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("default_run")
    actions = _fixed_observe_actions(DEFAULT_SCENARIO, 3)
    report = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory(actions), out)
    return out, report


def test_runner_outputs_and_determinism(small_default_run, tmp_path):
    out, report = small_default_run
    assert report["counts"]["observe_actions"] == 3
    assert report["counts"]["observations"] > 0
    # Replaying the same action sequence reproduces every artifact byte-for-byte.
    out2 = tmp_path / "replay"
    vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory(_fixed_observe_actions(DEFAULT_SCENARIO, 3)), out2)
    for name in ("decisions.csv", "observations.csv", "messages.jsonl", "score_report.json"):
        assert (out / name).read_bytes() == (out2 / name).read_bytes(), name


def test_wait_action(tmp_path):
    actions = [{"action": "wait", "duration_seconds": 900}] + _fixed_observe_actions(DEFAULT_SCENARIO, 1)
    out = tmp_path / "wait_run"
    report = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory(actions), out)
    with (out / "decisions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["action"] == "wait"
    assert rows[0]["alt_deg"] == ""
    # The wait advanced time by exactly its duration before the observe started.
    first_end = datetime.fromisoformat(rows[0]["end_utc"].replace("Z", "+00:00"))
    second_start = datetime.fromisoformat(rows[1]["start_utc"].replace("Z", "+00:00"))
    assert first_end == second_start
    assert rows[1]["action"] == "observe"
    assert report["counts"]["observe_actions"] == 1


def test_runner_context_exposes_only_public_scenario_fields(tmp_path):
    def factory(context):
        assert set(context["scenario"]) == {"name", "minimum_altitude_deg"}
        assert "products" not in context["scenario"]
        assert "footprint_csv" not in context
        assert context["footprint"]
        assert context["targets"]
        return lambda snapshot: None

    vr.run_scenario(DEFAULT_SCENARIO, factory, tmp_path / "public_context")


def test_false_report_allowance_persists_across_observe_and_wait(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    slot = scenario.slots[0]
    score_config = {**scenario.score_config, "reporting": {
        **scenario.score_config["reporting"],
        "false_report_free_allowance": 2, "max_consecutive_reports": 32,
    }}
    short = replace(scenario, score_config=score_config, slots=[slot], events=[],
                    survey_start=slot.start_utc, survey_end=slot.end_utc,
                    bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    actions = [
        {"action": "report"},
        {"action": "observe", "pointing": {"alt_deg": 60, "az_deg": 0},
         "assignments": {}, "duration_seconds": 60, "program": "BACKUP"},
        {"action": "report"},
        {"action": "wait", "duration_seconds": 60},
        {"action": "report"},
    ]
    report = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory(actions), tmp_path / "reports")
    assert report["termination"]["reason"] == "agent_finished"
    assert report["counts"]["decisions"] == 5
    assert report["components"]["report_settlement"] == -150.0
    messages = [json.loads(line) for line in (tmp_path / "reports" / "messages.jsonl").read_text().splitlines()]
    report_messages = [item for item in messages if item["record_type"] == "report_result"]
    assert [item["score_delta"] for item in report_messages] == [0.0, 0.0, -150.0]


def test_correct_report_counts_toward_total_report_cap(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    fault = next(event for event in scenario.events if event.event_type == "instrument_fault")
    slot = next(item for item in scenario.slots if item.start_utc == fault.actual_start_utc)
    score_config = {**scenario.score_config, "reporting": {
        **scenario.score_config["reporting"],
        "false_report_free_allowance": 0, "max_consecutive_reports": 1,
    }}
    short = replace(scenario, score_config=score_config, slots=[slot],
                    survey_start=slot.start_utc, survey_end=slot.end_utc,
                    bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    report = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([
        {"action": "report"}, {"action": "report"},
    ]), tmp_path / "correct_then_over_cap")
    assert report["termination"]["reason"] == "agent_error"
    assert report["counts"]["decisions"] == 1
    assert report["components"]["report_settlement"] == 100.0


def test_legacy_score_config_keeps_immediate_penalty_and_32_report_cap(tmp_path, monkeypatch):
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    slot = scenario.slots[0]
    reporting = dict(scenario.score_config["reporting"])
    del reporting["false_report_free_allowance"]
    del reporting["max_consecutive_reports"]
    score_config = {**scenario.score_config, "reporting": reporting}
    short = replace(scenario, score_config=score_config, slots=[slot], events=[],
                    survey_start=slot.start_utc, survey_end=slot.end_utc,
                    bulletins=[], forecasts=[])
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    actions = [{"action": "report"}] * 33
    report = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory(actions), tmp_path / "legacy_reports")
    assert report["termination"]["reason"] == "agent_error"
    assert report["counts"]["decisions"] == 32
    assert report["components"]["report_settlement"] == -4800.0


def test_data_loss_trigger_resync(tmp_path):
    """Six observes before the first earthquake ends, then one after: the runtime trigger
    invalidates window [floor(i*N), floor(j*N)) = [4, 5) and republishes best scores."""
    scenario = vr.load_scenario(STRESS_SCENARIO)
    quake = next(event for event in scenario.events if event.event_type == "earthquake")
    trigger = quake.actual_end_utc
    # Short 60 s exposures keep the fixed assignments on-glass across all six actions.
    observes = _fixed_observe_actions(STRESS_SCENARIO, 6, duration=60)

    def factory(context):
        state = {"done_observes": 0, "final": False}

        def agent(snapshot):
            now = datetime.fromisoformat(snapshot["now_utc"].replace("Z", "+00:00"))
            if state["done_observes"] < len(observes):
                action = observes[state["done_observes"]]
                state["done_observes"] += 1
                return action
            if now < trigger + timedelta(seconds=900):
                return {"action": "wait", "duration_seconds": 3600}
            if not state["final"]:
                state["final"] = True
                return observes[0]
            return None

        return agent

    out = tmp_path / "loss_run"
    report = vr.run_scenario(STRESS_SCENARIO, factory, out)
    assert report["counts"]["observe_actions"] == 7
    [note] = report["invalidations"]
    assert note["event_id"] == "V4ST0001"
    assert note["window"] == [4, 5]  # floor(0.828565*6), floor(0.877466*6)
    with (out / "decisions.csv").open("r", encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    invalidated = [row for row in decisions if row["valid"] == "false"]
    assert len(invalidated) == 1
    assert invalidated[0]["observe_index"] == "4"
    assert invalidated[0]["invalidated_by"] == "V4ST0001"
    with (out / "observations.csv").open("r", encoding="utf-8", newline="") as handle:
        observations = list(csv.DictReader(handle))
    voided = [row for row in observations if row["valid"] == "false"]
    assert voided and all(row["observe_index"] == "4" for row in voided)

    messages = [json.loads(line) for line in (out / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    resyncs = [item for item in messages if item["record_type"] == "state_resync"]
    assert len(resyncs) == 1
    resync = resyncs[0]
    next_decision = next(
        row for row in decisions
        if datetime.fromisoformat(row["start_utc"].replace("Z", "+00:00")) >= trigger
    )
    assert resync["issued_at_utc"] == next_decision["start_utc"]
    window = resync["invalidated_window"]
    assert window["action_count_at_trigger"] == 6
    assert window["action_index_start"] == 4
    assert window["action_index_end_exclusive"] == 5
    assert window["window_start_fraction"] == pytest.approx(0.828565)
    # Recompute expected best scores from the surviving (valid) observation rows.
    expected_best: dict[str, float] = {}
    for row in observations:
        if row["valid"] == "true":
            tid = row["target_id"]
            expected_best[tid] = max(expected_best.get(tid, 0.0), float(row["score"]))
    got = {item["target_id"]: item["best_score"] for item in resync["best_scores"]}
    assert got == {tid: round(score, 6) for tid, score in sorted(expected_best.items())}
    assert resync["observed_target_ids"] == sorted(expected_best)


def test_pointing_offset_moves_hits(tmp_path):
    """Same fixed action on both scenarios: the stress offset shifts the true field."""
    scenario = vr.load_scenario(DEFAULT_SCENARIO)
    grid = FiberGrid.from_config(scenario.fiber_config)
    lat = float(scenario.site["latitude_deg"])
    lon = float(scenario.site["longitude_deg"])
    start = scenario.survey_start
    anchor = next(
        t for t in scenario.targets
        if radec_to_altaz(t["ra_deg"], t["dec_deg"], start, lat, lon)[0] > 50.0
    )
    anchor_alt, anchor_az = radec_to_altaz(anchor["ra_deg"], anchor["dec_deg"], start, lat, lon)
    # Put the anchor near the top edge of fiber 5. The stress offset shifts
    # it into a neighboring cell, so assigning fiber 5 no longer hits it.
    center_alt, center_az = grid.fiber_center_offset(5)
    edge_alt = center_alt + grid.fiber_side_deg / 2.0 - 0.01
    cmd_alt = round(anchor_alt - edge_alt, 4)
    cmd_az = round(anchor_az - center_az / math.cos(math.radians(cmd_alt)), 4)
    action = {
        "action": "observe",
        "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
        "assignments": {5: anchor["target_id"]},
        "duration_seconds": 900,
        "program": "BACKUP",
    }
    report_default = vr.run_scenario(DEFAULT_SCENARIO, _scripted_factory([dict(action)]), tmp_path / "off")
    report_stress = vr.run_scenario(STRESS_SCENARIO, _scripted_factory([dict(action)]), tmp_path / "on")
    assert report_default["counts"]["observations"] == 1  # on glass without offset
    assert report_stress["counts"]["observations"] == 0  # offset changed its cell
    assert report_stress["organizer_only"]["pointing_offset_deg"]["alt"] == pytest.approx(-0.031697, abs=1e-4)
