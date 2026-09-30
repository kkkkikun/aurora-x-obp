"""Platform hardening of the vendored v4 modules (see challenge/CHANGES-v4.md)."""
from __future__ import annotations

import copy
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from challenge import v4_catalog_generator as v4c
from challenge import v4_config_check as cc
from challenge import v4_runner as vr
from challenge import v4_weather_simulator as v4w
from challenge.contracts import derive_stream_seed
from challenge.v4_fiber_map import FiberGrid, radec_to_altaz

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "v4"


def _load(name: str) -> dict:
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def _write(path: Path, config: dict) -> Path:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _small_weather(seed=777, **extra) -> dict:
    config = _load("v4_weather_config.json")
    config["seed"] = seed
    config["survey"]["start_date"] = "2026-11-02"
    config["survey"]["end_date"] = "2026-11-30"
    config.update(extra)
    return config


def _small_catalog(seed=424242, **extra) -> dict:
    config = _load("v4_catalog_config.json")
    config["seed"] = seed
    config["footprint"]["total_area_deg2"] = 800.0
    config["footprint"]["n_components"] = 2
    config["footprint"]["component_area_weights"] = [0.6, 0.4]
    config["footprint"]["vertices_per_component"] = 24
    config["targets"]["total_count"] = 300
    config["observability"]["start_date"] = "2026-11-02"
    config["observability"]["end_date"] = "2026-11-30"
    config.update(extra)
    return config


# --- seeds --------------------------------------------------------------------------------


def test_hashed_seed_streams_are_opt_in_and_change_every_stream(tmp_path):
    legacy = v4w.generate(_write(tmp_path / "legacy.json", _small_weather()), tmp_path / "legacy")
    hashed = v4w.generate(
        _write(tmp_path / "hashed.json", _small_weather(seed_derivation="sha256-v1")), tmp_path / "hashed"
    )
    for name in ("v4_weather_truth.csv", "v4_events.csv"):
        assert (tmp_path / "legacy" / name).read_bytes() != (tmp_path / "hashed" / name).read_bytes()
    # The calendar is ephemeris only: identical under either seed mode.
    assert (tmp_path / "legacy" / "v4_slots.csv").read_bytes() == (tmp_path / "hashed" / "v4_slots.csv").read_bytes()
    assert "seed" not in legacy and "seed" not in hashed
    assert "seed" not in json.loads((tmp_path / "hashed" / "v4_weather_summary.json").read_text())


def test_hashed_stream_seed_is_the_contracts_derivation():
    config = {"seed": 123456789, "seed_derivation": "sha256-v1"}
    from challenge.contracts import stream_seed

    assert stream_seed(config, "v4.weather.slots", 1000) == derive_stream_seed(123456789, "v4.weather.slots")
    assert stream_seed({"seed": 5}, "v4.weather.slots", 1000) == 1005  # legacy stays byte-compatible


def test_catalog_summary_carries_no_seed_or_config_hash(tmp_path):
    config = _small_catalog(seed_derivation="sha256-v1", seed=2**127 + 12345)
    summary = v4c.generate_catalog(_write(tmp_path / "catalog.json", config), tmp_path / "out")
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    for document in (summary, written):
        assert "seed" not in document
        assert "config" not in document["sha256"]
    assert str(config["seed"]) not in (tmp_path / "out" / "summary.json").read_text()


@pytest.mark.parametrize("bad", [-1, True, 1.5, "7"])
def test_seed_must_be_a_non_negative_integer(tmp_path, bad):
    with pytest.raises(ValueError, match="seed"):
        v4w.validate_config(_small_weather(seed=bad))
    with pytest.raises(ValueError, match="seed"):
        v4c.validate_config(_small_catalog(seed=bad))


def test_unknown_seed_derivation_is_rejected():
    with pytest.raises(ValueError, match="seed_derivation"):
        v4w.validate_config(_small_weather(seed_derivation="md5"))
    with pytest.raises(ValueError, match="seed_derivation"):
        v4c.validate_config(_small_catalog(seed_derivation="md5"))


# --- forecast horizon ---------------------------------------------------------------------


def test_forecast_nights_stay_inside_the_coverage_window(tmp_path):
    config = _small_weather()
    # Long events so that several of them straddle a forecast horizon.
    for condition in config["weather_events"]["conditions"].values():
        condition["duration_slots"] = [150, 250]
    config["publication"]["forecast_horizon_days"] = 3
    v4w.generate(_write(tmp_path / "long.json", config), tmp_path / "out")
    nights = {row["night_date"]: row for row in csv.DictReader((tmp_path / "out" / "v4_night_calendar.csv").open())}
    notices = 0
    for line in (tmp_path / "out" / "v4_forecasts.jsonl").read_text().splitlines():
        record = json.loads(line)
        start, end = record["coverage_start_utc"], record["coverage_end_utc"]
        for notice in record["notices"]:
            notices += 1
            assert notice["nights"]
            for night in notice["nights"]:
                row = nights[night]
                assert row["observing_start_utc"] < end and row["observing_end_utc"] > start
    assert notices


# --- config cross-validation --------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
def test_false_report_free_allowance_must_be_a_non_negative_integer(bad):
    score = _load("v4_score_config.json")
    score["reporting"]["false_report_free_allowance"] = bad
    with pytest.raises(ValueError, match="false_report_free_allowance"):
        cc.validate_score_config(score)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "32"])
def test_total_report_cap_must_be_a_positive_integer(bad):
    score = _load("v4_score_config.json")
    score["reporting"]["max_consecutive_reports"] = bad
    with pytest.raises(ValueError, match="max_consecutive_reports"):
        cc.validate_score_config(score)


def test_legacy_score_config_without_false_report_limit_is_accepted():
    score = _load("v4_score_config.json")
    del score["reporting"]["false_report_free_allowance"]
    del score["reporting"]["max_consecutive_reports"]
    cc.validate_score_config(score)


def test_legacy_card_load_publishes_the_default_false_report_limit(v4_reference_dir, tmp_path, monkeypatch):
    score = _load("v4_score_config.json")
    del score["reporting"]["false_report_free_allowance"]
    del score["reporting"]["max_consecutive_reports"]
    legacy_score_path = _write(tmp_path / "legacy_score.json", score)
    resolve = vr._resolve
    monkeypatch.setattr(
        vr, "_resolve",
        lambda base, value: legacy_score_path if value == "v4_score_config.json" else resolve(base, value),
    )
    scenario = vr.load_scenario(v4_reference_dir / "v4_scenario_default.json")
    assert scenario.score_config["reporting"]["false_report_free_allowance"] == 0
    assert scenario.score_config["reporting"]["max_consecutive_reports"] == 32


def test_generator_configs_must_agree():
    catalog, weather = _small_catalog(), _small_weather()
    scenario, fiber = _load("v4_scenario_default.json"), _load("v4_fiber_config.json")
    cc.cross_validate_generator_configs(catalog, weather, scenario, fiber)

    moved = copy.deepcopy(weather)
    moved["site"]["latitude_deg"] = -30.0
    with pytest.raises(ValueError, match="site mismatch"):
        cc.cross_validate_generator_configs(catalog, moved, scenario, fiber)
    late = copy.deepcopy(weather)
    late["survey"]["end_date"] = "2026-12-30"
    with pytest.raises(ValueError, match="end_date"):
        cc.cross_validate_generator_configs(catalog, late)
    stressed = copy.deepcopy(weather)
    stressed["stress_tests"]["enabled"] = True
    with pytest.raises(ValueError, match="stress"):
        cc.cross_validate_generator_configs(catalog, stressed, scenario)
    lower = copy.deepcopy(scenario)
    lower["minimum_altitude_deg"] = 25.0
    with pytest.raises(ValueError, match="minimum_altitude"):
        cc.cross_validate_generator_configs(catalog, weather, lower)
    with pytest.raises(ValueError, match="seed_derivation"):
        cc.cross_validate_generator_configs(catalog, weather, require_hashed_seeds=True)
    cc.cross_validate_generator_configs(
        dict(catalog, seed_derivation="sha256-v1"), dict(weather, seed_derivation="sha256-v1"),
        require_hashed_seeds=True,
    )


def test_scenario_load_rejects_inconsistent_configs(v4_reference_dir, tmp_path):
    import shutil

    root = tmp_path / "bad"
    shutil.copytree(v4_reference_dir, root, ignore=shutil.ignore_patterns("stress", "run_*"))
    fiber = json.loads((root / "v4_fiber_config.json").read_text())
    fiber["site"]["longitude_deg"] = 10.0
    _write(root / "v4_fiber_config.json", fiber)
    with pytest.raises(ValueError, match="site mismatch"):
        vr.load_scenario(root / "v4_scenario_default.json")

    fiber["site"]["longitude_deg"] = -70.3976
    fiber["exposure"]["min_duration_seconds"] = 7200
    _write(root / "v4_fiber_config.json", fiber)
    with pytest.raises(ValueError, match="exposure bounds"):
        vr.load_scenario(root / "v4_scenario_default.json")

    fiber["exposure"]["min_duration_seconds"] = 60
    _write(root / "v4_fiber_config.json", fiber)
    bulletins = (root / "products" / "v4_bulletins.jsonl").read_text().splitlines()
    (root / "products" / "v4_bulletins.jsonl").write_text("\n".join([bulletins[1], bulletins[0]] + bulletins[2:]) + "\n")
    with pytest.raises(ValueError, match="chronological"):
        vr.load_scenario(root / "v4_scenario_default.json")


# --- runner: invalid actions, termination, wait-until -----------------------------------


@pytest.fixture(scope="module")
def default_scenario(v4_reference_dir):
    return v4_reference_dir / "v4_scenario_default.json"


def _anchor_observe(scenario_path: Path, at: datetime | None = None, duration: int = 900) -> dict:
    """One valid observe at the given time with a real target on fibre 5."""
    scenario = vr.load_scenario(scenario_path)
    grid = FiberGrid.from_config(scenario.fiber_config)
    lat, lon = float(scenario.site["latitude_deg"]), float(scenario.site["longitude_deg"])
    start = at or scenario.survey_start
    target = next(
        t for t in scenario.targets
        if 50.0 < radec_to_altaz(t["ra_deg"], t["dec_deg"], start, lat, lon)[0] < 70.0
    )
    alt, az = radec_to_altaz(target["ra_deg"], target["dec_deg"], start, lat, lon)
    d_alt, d_az = grid.fiber_center_offset(5)
    cmd_alt = alt - d_alt
    return {
        "action": "observe",
        "pointing": {"alt_deg": cmd_alt, "az_deg": (az - d_az / math.cos(math.radians(cmd_alt))) % 360.0},
        "assignments": {"5": target["target_id"]},
        "duration_seconds": duration,
        "program": "BACKUP",
    }


def _scripted(actions, seen=None):
    def factory(context):
        iterator = iter(actions)

        def agent(snapshot):
            if seen is not None:
                seen.append(snapshot)
            item = next(iterator, None)
            if isinstance(item, Exception):
                raise item
            return item

        return agent

    return factory


def _bad_actions(observe: dict) -> list[tuple[object, str]]:
    def mutate(**changes):
        action = copy.deepcopy(observe)
        for key, value in changes.items():
            if value is KeyError:
                action.pop(key)
            else:
                action[key] = value
        return action

    return [
        ("not an object", "JSON object"),
        ({"action": "teleport"}, "unknown action"),
        ({"action": "wait", "duration_seconds": 30}, "wait duration_seconds"),
        ({"action": "wait", "duration_seconds": 900.5}, "whole number"),
        ({"action": "wait"}, "exactly one"),
        ({"action": "wait", "until_utc": "2026-10-01T00:00:00Z"}, "later than now"),
        ({"action": "wait", "until_utc": "tomorrow"}, "timestamp"),
        ({"action": "report", "fault": True}, "no fields"),
        (mutate(program="LUX"), "program"),
        (mutate(duration_seconds=10), "duration_seconds"),
        (mutate(duration_seconds=True), "finite number"),
        (mutate(pointing={"alt_deg": 95.0, "az_deg": 10.0}), "alt_deg"),
        (mutate(pointing={"alt_deg": 45.0, "az_deg": 360.0}), "az_deg"),
        (mutate(pointing={"alt_deg": float("nan"), "az_deg": 1.0}), "finite"),
        (mutate(assignments={"16": "V4T000001"}), "fiber_id 16"),
        (mutate(assignments={"x": "V4T000001"}), "invalid fiber_id"),
        (mutate(assignments={"5": "V4T000001", "05": "V4T000002"}), "assigned twice"),
        (mutate(assignments={"1": "V4T000001", "2": "V4T000001"}), "more than one fiber"),
        (mutate(assignments={"1": "NOPE"}), "unknown target_id"),
        (mutate(assignments=KeyError), "observe needs"),
        (mutate(note="hi"), "observe needs"),
    ]


def test_invalid_actions_end_as_agent_error_and_settle_on_valid_history(default_scenario, tmp_path):
    observe = _anchor_observe(default_scenario)
    baseline = vr.run_scenario(default_scenario, _scripted([observe]), tmp_path / "baseline")
    assert baseline["termination"]["reason"] == "agent_finished"
    assert baseline["components"]["sum_best_scores"] > 0
    for index, (bad, needle) in enumerate(_bad_actions(observe)):
        report = vr.run_scenario(default_scenario, _scripted([observe, bad]), tmp_path / f"bad{index}")
        assert report["termination"]["reason"] == "agent_error", bad
        assert needle in report["termination"]["detail"], (bad, report["termination"])
        assert report["total"] == baseline["total"], bad
        assert report["counts"]["decisions"] == 1


def test_agent_termination_and_finish_are_recorded(default_scenario, tmp_path):
    observe = _anchor_observe(default_scenario)
    stop = vr.AgentTermination("global_wallclock_expired", "900 s budget used")
    report = vr.run_scenario(default_scenario, _scripted([observe, stop]), tmp_path / "wall")
    assert report["termination"] == {"reason": "global_wallclock_expired", "detail": "900 s budget used"}
    assert report["counts"]["observe_actions"] == 1
    written = json.loads((tmp_path / "wall" / "score_report.json").read_text())
    assert written["termination"]["reason"] == "global_wallclock_expired"

    finished = vr.run_scenario(default_scenario, _scripted([observe, {"action": "finish"}]), tmp_path / "fin")
    assert finished["termination"]["reason"] == "agent_finished"
    assert finished["total"] == report["total"]

    def failing_factory(context):
        raise vr.AgentTermination("agent_error", "no initialize ack")

    empty = vr.run_scenario(default_scenario, failing_factory, tmp_path / "init")
    assert empty["termination"]["reason"] == "agent_error"
    assert empty["counts"]["decisions"] == 0
    with pytest.raises(ValueError):
        vr.AgentTermination("crashed")


def test_survey_complete_after_the_last_slot(default_scenario, tmp_path, monkeypatch):
    from dataclasses import replace

    scenario = vr.load_scenario(default_scenario)
    short = replace(scenario, slots=scenario.slots[:2], survey_end=scenario.slots[1].end_utc)
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    report = vr.run_scenario(default_scenario, _scripted([{"action": "wait", "duration_seconds": 3600}]), tmp_path / "s")
    assert report["termination"]["reason"] == "survey_complete"


@pytest.mark.parametrize("remaining_seconds", [30, 300])
def test_observe_stops_at_night_end_and_scores_actual_duration(
    default_scenario, tmp_path, remaining_seconds
):
    scenario = vr.load_scenario(default_scenario)
    final_slots = [
        slot for index, slot in enumerate(scenario.slots[:-1])
        if slot.slot_id.split("-S", 1)[0] != scenario.slots[index + 1].slot_id.split("-S", 1)[0]
    ]
    final_slot = next(slot for slot in final_slots if slot.is_observable)
    start = final_slot.end_utc - timedelta(seconds=remaining_seconds)
    until = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    observe = _anchor_observe(default_scenario, start, 900)
    prefix = [{"action": "wait", "until_utc": until}]

    clipped_dir = tmp_path / "clipped"
    seen: list[dict] = []
    clipped = vr.run_scenario(default_scenario, _scripted(prefix + [observe], seen), clipped_dir)
    rows = list(csv.DictReader((clipped_dir / "decisions.csv").open()))
    assert rows[-1]["action"] == "observe"
    assert rows[-1]["end_utc"] == final_slot.end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert rows[-1]["duration_seconds"] == str(remaining_seconds)
    assert seen[-1]["now_utc"] == rows[-1]["end_utc"]
    assert clipped["components"]["sum_best_scores"] > 0

    if remaining_seconds < 60:
        return  # a legal request can be cut below the minimum request duration

    exact_dir = tmp_path / "exact"
    exact = vr.run_scenario(
        default_scenario,
        _scripted(prefix + [dict(observe, duration_seconds=remaining_seconds)]),
        exact_dir,
    )
    assert clipped["total"] == exact["total"]
    for name in ("decisions.csv", "observations.csv", "score_report.json"):
        assert (clipped_dir / name).read_bytes() == (exact_dir / name).read_bytes()


def test_observe_still_crosses_internal_slot_boundary(default_scenario, tmp_path):
    scenario = vr.load_scenario(default_scenario)
    start = scenario.slots[0].end_utc - timedelta(seconds=300)
    observe = _anchor_observe(default_scenario, start, 900)
    out = tmp_path / "cross_slot"
    vr.run_scenario(
        default_scenario,
        _scripted([{"action": "wait", "until_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ")}, observe]),
        out,
    )
    rows = list(csv.DictReader((out / "decisions.csv").open()))
    assert rows[-1]["duration_seconds"] == "900"
    assert rows[-1]["end_utc"] == (start + timedelta(seconds=900)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_wait_until_expands_without_round_trips_and_carries_messages(default_scenario, tmp_path):
    scenario = vr.load_scenario(default_scenario)
    second_night = next(slot for slot in scenario.slots if slot.slot_id.endswith("-S001") and slot.start_utc > scenario.survey_start)
    until = second_night.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    seen: list[dict] = []
    out = tmp_path / "until"
    report = vr.run_scenario(default_scenario, _scripted([{"action": "wait", "until_utc": until}], seen), out)
    assert len(seen) == 2  # one request before the wait, one after it: no round trips in between
    assert seen[1]["now_utc"] == until
    delivered = seen[1]["new_messages"]
    bulletin_ids = [item["slot_id"] for item in delivered if item["record_type"] == "bulletin"]
    night_one = scenario.slots[0].slot_id.split("-")[0]
    first_night_slots = [slot.slot_id for slot in scenario.slots if slot.slot_id.startswith(night_one + "-")]
    # Every bulletin published during the wait (rest of night 1 + the opening of night 2) arrives once.
    assert bulletin_ids == first_night_slots[1:] + [second_night.slot_id]
    rows = list(csv.DictReader((out / "decisions.csv").open()))
    assert rows and all(row["action"] == "wait" for row in rows)
    assert all(int(row["duration_seconds"]) <= 3600 for row in rows)
    assert rows[0]["start_utc"] == seen[0]["now_utc"] and rows[-1]["end_utc"] == until
    assert report["termination"]["reason"] == "agent_finished"


def test_wait_until_beyond_the_survey_ends_the_survey(default_scenario, tmp_path, monkeypatch):
    from dataclasses import replace

    scenario = vr.load_scenario(default_scenario)
    short = replace(scenario, slots=scenario.slots[:40], survey_end=scenario.slots[39].end_utc)
    monkeypatch.setattr(vr, "load_scenario", lambda _: short)
    seen: list[dict] = []
    report = vr.run_scenario(default_scenario, _scripted([{"action": "wait", "until_utc": "2027-06-01T00:00:00Z"}], seen), tmp_path / "far")
    assert len(seen) == 1
    assert report["termination"]["reason"] == "survey_complete"


def test_written_report_hides_stress_truth(v4_reference_dir, tmp_path):
    stress = v4_reference_dir / "v4_scenario_stress.json"
    report = vr.run_scenario(stress, _scripted([]), tmp_path / "stress")
    assert "organizer_only" in report
    text = (tmp_path / "stress" / "score_report.json").read_text()
    assert "pointing_offset" not in text and "organizer_only" not in text and "seed" not in text
