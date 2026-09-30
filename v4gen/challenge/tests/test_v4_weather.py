"""Tests for the v4 weather and event subsystem (challenge/v4_weather_simulator.py)."""
from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from challenge import v4_weather_simulator as v4w
from challenge import observing_calendar
from challenge.contracts import NIGHT_COLUMNS, SLOT_COLUMNS, WEATHER_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "v4"
DEFAULT_CONFIG_PATH = REFERENCE / "v4_weather_config.json"

ARTIFACTS = (
    "v4_night_calendar.csv",
    "v4_slots.csv",
    "v4_weather_truth.csv",
    "v4_events.csv",
    "v4_earthquake_effects.csv",
    "v4_bulletins.jsonl",
    "v4_forecasts.jsonl",
    "v4_weather_summary.json",
)


def _small_config() -> dict:
    """Short survey for fast tests: two weeks, same site and event mix."""
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["seed"] = 777
    config["survey"]["start_date"] = "2026-11-02"
    config["survey"]["end_date"] = "2026-11-16"
    return config


def _run(tmp_path: Path, config: dict, name: str) -> tuple[Path, dict]:
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output_dir = tmp_path / name
    summary = v4w.generate(config_path, output_dir)
    return output_dir, summary


def test_byte_determinism(tmp_path):
    first_dir, _ = _run(tmp_path, _small_config(), "first")
    second_dir, _ = _run(tmp_path, _small_config(), "second")
    for name in ARTIFACTS:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes(), name


def test_default_calendar_matches_published_starter_math():
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    starter_config = {
        "survey": {
            "start_date": config["survey"]["start_date"],
            "days": (date.fromisoformat(config["survey"]["end_date"])
                     - date.fromisoformat(config["survey"]["start_date"])).days,
            "slot_seconds": config["survey"]["slot_seconds"],
        },
        "site": {**config["site"], "sun_altitude_limit_deg": config["survey"]["sun_altitude_limit_deg"]},
        "solver": {"coarse_step_seconds": 300, "crossing_tolerance_seconds": 1},
    }
    starter_nights, starter_slots = observing_calendar.build_calendar(starter_config)
    v4_nights, v4_slots = v4w.build_nights(config)
    assert [(night.solar_dusk_utc, night.solar_dawn_utc) for night in v4_nights] == [
        (night.solar_dusk_utc, night.solar_dawn_utc) for night in starter_nights
    ]
    assert [slot.timestamp_utc for slot in v4_slots] == [slot.timestamp_utc for slot in starter_slots]


def test_csv_column_contracts(tmp_path):
    output_dir, _ = _run(tmp_path, _small_config(), "run")
    expectations = {
        "v4_night_calendar.csv": NIGHT_COLUMNS,
        "v4_slots.csv": SLOT_COLUMNS,
        "v4_weather_truth.csv": WEATHER_COLUMNS,
        "v4_events.csv": v4w.V4_EVENT_COLUMNS,
        "v4_earthquake_effects.csv": v4w.V4_EARTHQUAKE_EFFECT_COLUMNS,
    }
    for name, columns in expectations.items():
        with (output_dir / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(columns), name


def test_earthquake_magnitude_impact_monotonic_and_exponential():
    config = _small_config()
    quake = config["earthquake"]
    # Below the cap, degradation is exactly exponential (log-linear) in magnitude.
    magnitudes = [4.5, 4.8, 5.1, 5.4, 5.7]
    degradations = [v4w.earthquake_initial_degradation(m, config) for m in magnitudes]
    assert all(a < b for a, b in zip(degradations, degradations[1:]))
    coefficient = float(quake["impact_coefficient"])
    for m, d in zip(magnitudes, degradations):
        expected = math.exp(coefficient * (m - float(quake["reference_magnitude"])))
        assert d == pytest.approx(expected, rel=1e-12)
    # Above the reference magnitude the cap binds and stays bounded below 1.
    assert v4w.earthquake_initial_degradation(8.0, config) == float(quake["max_degradation"])
    # Exponential per-night decay: fixed ratio between consecutive nights.
    initial = v4w.earthquake_initial_degradation(5.5, config)
    tau = float(quake["decay_nights"])
    values = [v4w.earthquake_night_degradation(initial, k, config) for k in range(5)]
    assert values[0] == pytest.approx(initial)
    for a, b in zip(values, values[1:]):
        assert b / a == pytest.approx(math.exp(-1.0 / tau), rel=1e-12)
    assert v4w.earthquake_night_degradation(initial, -1, config) == 0.0


def test_earthquake_truth_curves(tmp_path):
    output_dir, summary = _run(tmp_path, _small_config(), "run")
    config = _small_config()
    with (output_dir / "v4_earthquake_effects.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected earthquake effect rows"
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)
    quake = config["earthquake"]
    for event_id, event_rows in by_event.items():
        degradations = [float(row["degradation"]) for row in event_rows]
        # Strict decay night over night within each earthquake.
        assert all(a > b for a, b in zip(degradations, degradations[1:]))
        assert all(d >= float(quake["negligible_degradation"]) for d in degradations)
        magnitude = float(event_rows[0]["magnitude"])
        # The CSV stores magnitude at 3 decimals, so allow for that rounding.
        assert degradations[0] == pytest.approx(
            v4w.earthquake_initial_degradation(magnitude, config), rel=1e-3
        )
        # Efficiency multiplier carries the full degradation.
        for row in event_rows:
            assert float(row["instrument_efficiency_multiplier"]) == pytest.approx(
                1.0 - float(row["degradation"]), abs=1e-6
            )


def test_earthquake_starts_at_event_slot_not_before(tmp_path):
    config = _small_config()
    with_quake, _ = _run(tmp_path, config, "with_quake")
    no_quake_config = _small_config()
    no_quake_config["earthquake"]["count"] = 0
    without_quake, _ = _run(tmp_path, no_quake_config, "without_quake")
    with (with_quake / "v4_events.csv").open(newline="", encoding="utf-8") as handle:
        quake = next(row for row in csv.DictReader(handle) if row["event_type"] == "earthquake")
    with (with_quake / "v4_weather_truth.csv").open(newline="", encoding="utf-8") as handle:
        affected = list(csv.DictReader(handle))
    with (without_quake / "v4_weather_truth.csv").open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    shock_start = quake["actual_start_utc"]
    shock_night = next(row["night_id"] for row in affected if row["timestamp_utc"] == shock_start)
    before = [i for i, row in enumerate(affected)
              if row["night_id"] == shock_night and row["timestamp_utc"] < shock_start]
    assert before
    assert all(affected[i] == baseline[i] for i in before)
    # The shock night can be entirely closed. Its persisted instrument damage
    # must still appear in a later observable slot.
    assert any(affected[i] != baseline[i] for i, row in enumerate(affected)
               if row["timestamp_utc"] >= shock_start and row["is_observable"] == "true")
    # Earthquake damage acts on the instrument, not on atmospheric seeing,
    # transparency, or sky brightness. All three still follow the same seeded
    # weather realization on both sides of the shock.
    for current, reference in zip(affected, baseline):
        for field in ("seeing_arcsec", "transparency", "sky_quality"):
            assert current[field] == reference[field]
    bulletins = [json.loads(line) for line in (with_quake / "v4_bulletins.jsonl").read_text().splitlines()]
    assert all(not any(note["event_kind"] == "earthquake" for note in bulletins[i]["notices"])
               for i in before)


def test_terrain_obstruction_constant_and_zero_score(tmp_path):
    output_dir, _ = _run(tmp_path, _small_config(), "run")
    with (output_dir / "v4_events.csv").open("r", encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    terrain = [row for row in events if row["event_type"] == "terrain_obstruction"]
    assert terrain, "expected at least one terrain obstruction sector"
    with (output_dir / "v4_slots.csv").open("r", encoding="utf-8", newline="") as handle:
        slots = list(csv.DictReader(handle))
    survey_start = slots[0]["timestamp_utc"]
    survey_end = slots[-1]["timestamp_utc"]
    for row in terrain:
        # Constant from the first slot to the end of the survey, marked zero-score.
        assert row["actual_start_utc"] == survey_start
        assert row["actual_end_utc"] >= survey_end
        assert row["zero_score"] == "true"
        assert row["scope_type"] == "HORIZON_SECTOR"
        assert 0.0 <= float(row["azimuth_start_deg"]) < 360.0
        assert 0.0 <= float(row["azimuth_end_deg"]) < 360.0
        assert 30.0 < float(row["max_altitude_deg"]) <= 90.0
    # Non-terrain events never carry the zero-score flag.
    for row in events:
        if row["event_type"] != "terrain_obstruction":
            assert row["zero_score"] == "false"
    # The initial bulletin announces each terrain sector with a coarse direction only.
    first = json.loads((output_dir / "v4_bulletins.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["initial"] is True
    terrain_notices = [n for n in first["notices"] if n["event_kind"] == "terrain_obstruction"]
    assert len(terrain_notices) == len(terrain)
    for row, notice in zip(sorted(terrain, key=lambda r: r["event_id"]), terrain_notices):
        assert set(notice) == {"event_kind", "direction"}
        center = v4w._sector_center_azimuth(float(row["azimuth_start_deg"]), float(row["azimuth_end_deg"]))
        assert notice["direction"] == v4w.azimuth_to_direction(center)


def test_forecast_filtering(tmp_path):
    output_dir, _ = _run(tmp_path, _small_config(), "run")
    raw_bulletins = (output_dir / "v4_bulletins.jsonl").read_text(encoding="utf-8")
    raw_forecasts = (output_dir / "v4_forecasts.jsonl").read_text(encoding="utf-8")
    # Instrument faults are never published anywhere; earthquakes are never forecast.
    assert "instrument_fault" not in raw_bulletins
    assert "instrument_fault" not in raw_forecasts
    assert "earthquake" not in raw_forecasts
    # Forecasts contain only forecastable coarse kinds, with kind + direction + nights only.
    allowed = {v4w.COARSE_KIND[name] for name in v4w.FORECASTABLE_TYPES}
    saw_notice = False
    for line in raw_forecasts.splitlines():
        record = json.loads(line)
        assert record["record_type"] == "forecast"
        for notice in record["notices"]:
            saw_notice = True
            assert set(notice) == {"event_kind", "direction", "nights"}
            assert notice["event_kind"] in allowed
            assert notice["direction"] in v4w.DIRECTION_CODES or notice["direction"] == "ALL"
    assert saw_notice
    # Bulletins publish only kind + direction.
    for line in raw_bulletins.splitlines():
        record = json.loads(line)
        assert record["record_type"] == "bulletin"
        for notice in record["notices"]:
            assert set(notice) == {"event_kind", "direction"}


def test_rocket_launches_are_forecast(tmp_path):
    output_dir, summary = _run(tmp_path, _small_config(), "run")
    assert summary["events"]["rocket_launch_count"] > 0
    kinds = set()
    for line in (output_dir / "v4_forecasts.jsonl").read_text(encoding="utf-8").splitlines():
        for notice in json.loads(line)["notices"]:
            kinds.add(notice["event_kind"])
    assert "rocket_launch" in kinds


def test_direction_mapping():
    cases = {
        0.0: "N", 22.4: "N", 22.5: "NE", 45.0: "NE", 67.4: "NE", 67.5: "E",
        90.0: "E", 135.0: "SE", 180.0: "S", 225.0: "SW", 270.0: "W", 315.0: "NW",
        337.4: "NW", 337.5: "N", 359.9: "N",
    }
    for azimuth, expected in cases.items():
        assert v4w.azimuth_to_direction(azimuth) == expected, azimuth


def test_weather_truth_quality_fields(tmp_path):
    output_dir, summary = _run(tmp_path, _small_config(), "run")
    with (output_dir / "v4_weather_truth.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["survey"]["slot_count"]
    closed = 0
    for row in rows:
        if row["is_observable"] == "true":
            assert 0.5 <= float(row["seeing_arcsec"]) <= 4.0
            assert 0.08 <= float(row["transparency"]) <= 1.0
            assert 0.05 <= float(row["sky_quality"]) <= 1.5
            assert 0.1 <= float(row["instrument_efficiency"]) <= 1.0
        else:
            closed += 1
            assert row["seeing_arcsec"] == ""
    assert closed == summary["weather"]["closed_slot_count"] > 0


def test_night_calendar_matches_survey(tmp_path):
    output_dir, summary = _run(tmp_path, _small_config(), "run")
    with (output_dir / "v4_night_calendar.csv").open("r", encoding="utf-8", newline="") as handle:
        nights = list(csv.DictReader(handle))
    assert len(nights) == 14  # Nov 2 .. Nov 15 inclusive
    assert summary["survey"]["night_count"] == 14
    with (output_dir / "v4_slots.csv").open("r", encoding="utf-8", newline="") as handle:
        slots = list(csv.DictReader(handle))
    assert all(int(slot["duration_seconds"]) == 900 for slot in slots)
    per_night: dict[str, int] = {}
    for slot in slots:
        per_night[slot["night_id"]] = per_night.get(slot["night_id"], 0) + 1
    for night in nights:
        assert int(night["slot_count"]) == per_night[night["night_id"]]


def test_weather_crosses_daylight_but_rocket_stops_at_night_end():
    config = _small_config()
    nights, slots = v4w.build_nights(config)
    last_first_night = next(i for i, slot in enumerate(slots) if slot.night_id != nights[0].night_id) - 1
    end = v4w._end_after_observing_slots(slots, last_first_night, 3)
    assert end == slots[last_first_night + 2].end_utc
    assert end - slots[last_first_night].timestamp_utc > timedelta(hours=12)
    for definition in config["weather_events"]["conditions"].values():
        definition["count"] = 0
    config["weather_events"]["conditions"]["cloudy"]["count"] = 1
    config["weather_events"]["conditions"]["cloudy"]["duration_slots"] = [80, 80]
    config["rocket_launch"]["count"] = 1
    config["rocket_launch"]["duration_slots"] = [60, 60]
    config["earthquake"]["count"] = 0
    config["instrument_fault"]["count"] = 0
    events, _ = v4w.generate_events(config, nights, slots)
    [cloudy] = [item for item in events if item.event_type == "cloudy"]
    assert sum(cloudy.overlaps(slot.timestamp_utc, slot.end_utc) for slot in slots) == 80

    [rocket] = [item for item in events if item.event_type == "rocket_launch"]
    rocket_start_slot = next(slot for slot in slots if slot.timestamp_utc == rocket.actual_start_utc)
    rocket_night = next(night for night in nights if night.night_id == rocket_start_slot.night_id)
    remaining_slots = sum(
        slot.night_id == rocket_night.night_id and slot.timestamp_utc >= rocket.actual_start_utc
        for slot in slots
    )
    assert remaining_slots < 60
    assert rocket.actual_end_utc == rocket_night.observing_end_utc
    assert sum(rocket.overlaps(slot.timestamp_utc, slot.end_utc) for slot in slots) == remaining_slots
    forecasts = [json.loads(line) for line in v4w.generate_forecasts(config, nights, events)]
    rocket_notices = [notice for record in forecasts for notice in record["notices"]
                      if notice["event_kind"] == "rocket_launch"]
    assert rocket_notices
    assert all(notice["nights"] == [rocket_night.night_date.isoformat()]
               for notice in rocket_notices)


def test_weather_truth_has_no_global_lunar_component(tmp_path):
    config = _small_config()
    baseline, _ = _run(tmp_path, config, "baseline")
    legacy = _small_config()
    legacy["lunar"] = {"lunar_strength": 0.35}
    with_legacy_key, _ = _run(tmp_path, legacy, "legacy")
    assert (baseline / "v4_weather_truth.csv").read_bytes() == (
        with_legacy_key / "v4_weather_truth.csv"
    ).read_bytes()
