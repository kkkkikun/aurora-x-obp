"""Tests for the gated v4 stress-test events (MP-055): data loss and pointing offset."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from challenge import v4_weather_simulator as v4w


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "v4"
DEFAULT_CONFIG_PATH = REFERENCE / "v4_weather_config.json"
STRESS_CONFIG_PATH = REFERENCE / "v4_weather_stress_config.json"

BASELINE_ARTIFACTS = (
    "v4_night_calendar.csv",
    "v4_slots.csv",
    "v4_weather_truth.csv",
    "v4_events.csv",
    "v4_earthquake_effects.csv",
    "v4_bulletins.jsonl",
    "v4_forecasts.jsonl",
    "v4_weather_summary.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict, name: str) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _stress_config(enabled: bool = True) -> dict:
    config = _load(STRESS_CONFIG_PATH)
    config["stress_tests"]["enabled"] = enabled
    return config


@pytest.fixture(scope="module")
def stress_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("stress")
    config_path = _write_config(out, _stress_config(True), "config")
    summary = v4w.generate(config_path, out / "products")
    return out / "products", summary, _stress_config(True)


def _read_stress_events(products: Path) -> list[dict[str, str]]:
    with (products / "v4_stress_events.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# (a) switch off => byte-identical to the same current weather model without stress keys
def test_disabled_matches_baseline(tmp_path):
    legacy = _load(DEFAULT_CONFIG_PATH)
    legacy.pop("stress_tests")
    out = tmp_path / "legacy"
    v4w.generate(_write_config(tmp_path, legacy, "legacy"), out)
    for name in BASELINE_ARTIFACTS:
        assert (out / name).is_file(), name
    assert not (out / "v4_stress_events.csv").exists()
    assert not (out / "v4_state_resync_schema.json").exists()

    # Adding an inert stress section changes only the embedded config hash.
    out2 = tmp_path / "disabled"
    v4w.generate(_write_config(tmp_path, _load(DEFAULT_CONFIG_PATH), "disabled"), out2)
    for name in BASELINE_ARTIFACTS:
        if name == "v4_weather_summary.json":
            continue
        assert (out2 / name).read_bytes() == (out / name).read_bytes(), name
    old = _load(out / "v4_weather_summary.json")
    new = _load(out2 / "v4_weather_summary.json")
    old["sha256"].pop("config")
    new["sha256"].pop("config")
    assert old == new


# (b) switch on => byte-level determinism
def test_enabled_byte_determinism(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    v4w.generate(_write_config(tmp_path, _stress_config(True), "a"), first)
    v4w.generate(_write_config(tmp_path, _stress_config(True), "b"), second)
    for path in sorted(first.iterdir()):
        assert (second / path.name).read_bytes() == path.read_bytes(), path.name


def _data_loss_row(stress_run):
    products, _, _ = stress_run
    rows = [row for row in _read_stress_events(products) if row["event_type"] == "data_loss"]
    assert len(rows) == 1
    return rows[0]


def _offset_row(stress_run):
    products, _, _ = stress_run
    rows = [row for row in _read_stress_events(products) if row["event_type"] == "pointing_offset"]
    assert len(rows) == 1
    return rows[0]


# (c) i/j window constraints
def test_data_loss_window_constraints(stress_run):
    _, _, config = stress_run
    row = _data_loss_row(stress_run)
    x = float(config["stress_tests"]["data_loss"]["window_max_fraction"])
    i = float(row["window_start_fraction"])
    j = float(row["window_end_fraction"])
    assert 0.0 <= i <= j <= 1.0
    assert j - i <= x + 1e-9
    assert float(row["window_max_fraction"]) == pytest.approx(x)
    # Rounding convention: the half-open floor window [floor(i*N), floor(j*N)) satisfies
    # width <= floor((j-i)*N) + 1 <= ceil(x*N) — floor rounding can widen by at most 1.
    for n in (1, 7, 100, 1000):
        width = math.floor(j * n) - math.floor(i * n)
        assert width <= math.floor((j - i) * n) + 1
        assert width <= math.ceil(x * n)


# (d) pointing offsets within the configured bounds
def test_pointing_offset_within_bounds(stress_run):
    _, _, config = stress_run
    row = _offset_row(stress_run)
    bounds = config["stress_tests"]["pointing_offset"]
    assert abs(float(row["alt_offset_rad"])) <= float(bounds["max_abs_alt_rad"])
    assert abs(float(row["az_offset_rad"])) <= float(bounds["max_abs_az_rad"])
    assert row["trigger_mode"] == "at_survey_start"


# (e) pointing_offset is never published anywhere (2026-09-27 ruling: no announcement)
def test_pointing_offset_never_published(stress_run):
    products, _, _ = stress_run
    bulletins = (products / "v4_bulletins.jsonl").read_text(encoding="utf-8")
    forecasts = (products / "v4_forecasts.jsonl").read_text(encoding="utf-8")
    assert bulletins.count("pointing_offset") == 0
    assert forecasts.count("pointing_offset") == 0
    # The truth rows still carry the offsets themselves (organizer-side only).
    row = _offset_row(stress_run)
    assert float(row["alt_offset_rad"]) != 0.0 or float(row["az_offset_rad"]) != 0.0
    assert "announcement_slot_id" not in row


# (f) after_earthquake follows the first earthquake; skipped with a warning when none
def test_after_earthquake_trigger(stress_run):
    products, _, _ = stress_run
    row = _data_loss_row(stress_run)
    assert row["trigger_mode"] == "after_earthquake"
    with (products / "v4_events.csv").open("r", encoding="utf-8", newline="") as handle:
        quakes = [r for r in csv.DictReader(handle) if r["event_type"] == "earthquake"]
    assert quakes, "stress sample config should contain earthquakes"
    assert row["trigger_ref"] == quakes[0]["event_id"]


def test_after_earthquake_without_quake_skips_with_warning(tmp_path):
    config = _stress_config(True)
    config["earthquake"]["count"] = 0
    out = tmp_path / "noquake"
    with pytest.warns(UserWarning, match="no earthquake"):
        v4w.generate(_write_config(tmp_path, config, "noquake"), out)
    rows = _read_stress_events(out)
    assert [row["event_type"] for row in rows] == ["pointing_offset"]


def test_fixed_slot_trigger(tmp_path):
    config = _stress_config(True)
    config["stress_tests"]["data_loss"]["trigger"] = "fixed_slot"
    config["stress_tests"]["data_loss"]["slot_id"] = "N20261101-S010"
    out = tmp_path / "fixed"
    v4w.generate(_write_config(tmp_path, config, "fixed"), out)
    rows = [row for row in _read_stress_events(out) if row["event_type"] == "data_loss"]
    assert len(rows) == 1
    assert rows[0]["trigger_mode"] == "fixed_slot"
    assert rows[0]["trigger_ref"] == "N20261101-S010"


# (g) stress events never appear in forecasts; no data_loss/pointing_offset bulletin either
def test_stress_events_absent_from_publications(stress_run):
    products, _, _ = stress_run
    forecasts = (products / "v4_forecasts.jsonl").read_text(encoding="utf-8")
    assert "pointing_offset" not in forecasts
    assert "data_loss" not in forecasts
    assert "state_resync" not in forecasts
    bulletins = (products / "v4_bulletins.jsonl").read_text(encoding="utf-8")
    assert "pointing_offset" not in bulletins
    assert "data_loss" not in bulletins
    assert "state_resync" not in bulletins


def test_state_resync_schema_written(stress_run):
    products, _, _ = stress_run
    schema = json.loads((products / "v4_state_resync_schema.json").read_text(encoding="utf-8"))
    assert schema["record_type"] == "state_resync"
    window = schema["fields"]["invalidated_window"]
    assert "floor(window_start_fraction * N)" in window["action_index_start"]
    assert "floor(window_end_fraction * N)" in window["action_index_end_exclusive"]


def test_stress_csv_column_contract(stress_run):
    products, _, _ = stress_run
    with (products / "v4_stress_events.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == v4w.V4_STRESS_EVENT_COLUMNS
