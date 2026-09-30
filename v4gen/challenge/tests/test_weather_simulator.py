from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.observing_calendar import load_slots
from challenge.tile_geometry_simulator import TileGeometrySimulator
from challenge.weather_simulator import (
    WeatherEvent,
    WeatherSimulator,
    generate,
    generate_events,
    generate_weather,
    load_config,
    load_events,
    load_forecasts,
    load_weather,
)


CONFIG = ROOT / "reference" / "config"
OUTPUT = ROOT / "reference" / "outputs" / "reference"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WeatherSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = TileGeometrySimulator.from_files(
            OUTPUT / "tiles.csv",
            CONFIG / "tile_config.json",
            CONFIG / "calendar_config.json",
            OUTPUT / "night_calendar.csv",
            OUTPUT / "slots.csv",
        )
        cls.weather = load_weather(OUTPUT / "weather.csv")
        cls.forecasts = load_forecasts(OUTPUT / "weather_forecasts.csv")
        cls.events = load_events(OUTPUT / "weather_events.csv")
        cls.config = load_config(CONFIG / "weather_config.json")

    def test_weather_rows_exactly_match_shared_slots(self) -> None:
        slots = load_slots(OUTPUT / "slots.csv")
        self.assertEqual(len(slots), len(self.weather))
        for expected, actual in zip(slots, self.weather):
            self.assertEqual(
                (expected.slot_id, expected.night_id, expected.timestamp_utc, expected.duration_seconds),
                (actual.slot_id, actual.night_id, actual.timestamp_utc, actual.duration_seconds),
            )
            values = (actual.seeing_arcsec, actual.transparency, actual.sky_quality, actual.instrument_efficiency)
            self.assertEqual(actual.is_observable, all(value is not None for value in values))

    def test_generation_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            generate(
                CONFIG / "weather_config.json",
                OUTPUT / "night_calendar.csv",
                OUTPUT / "slots.csv",
                OUTPUT / "tiles.csv",
                output,
            )
            for name in ("weather.csv", "weather_forecasts.csv", "weather_events.csv"):
                self.assertEqual(digest(output / name), digest(OUTPUT / name))

    def test_directional_force_close_does_not_close_control_tile(self) -> None:
        base = next(item for item in self.weather if item.is_observable)
        target = next(iter(self.geometry.tiles.values()))
        control = max(
            self.geometry.tiles.values(),
            key=lambda item: abs(item.ra_deg - target.ra_deg),
        )
        event = WeatherEvent(
            "TEST", "rocket_launch", base.timestamp_utc,
            base.end_utc, "SKY_CAP_ICRS",
            {"ra_deg": target.ra_deg, "dec_deg": target.dec_deg, "radius_deg": 1.0},
            1.0, True, 1.0, 1.0, 1.0, 1.0,
        )
        runtime = WeatherSimulator(self.weather, [], [event], self.config, self.geometry)
        affected = runtime.get_effective_conditions(base.slot_id, target.tile_id)
        unaffected = runtime.get_effective_conditions(base.slot_id, control.tile_id)
        self.assertFalse(affected["is_observable"])
        self.assertTrue(unaffected["is_observable"])
        self.assertEqual(affected["active_event_ids"], ["TEST"])
        self.assertEqual(unaffected["active_event_ids"], [])

    def test_forecast_query_never_publishes_future_revision(self) -> None:
        runtime = WeatherSimulator(self.weather, self.forecasts, self.events, self.config, self.geometry)
        first_issue = min(item.issued_at_utc for item in self.forecasts)
        self.assertEqual(runtime.get_weather_forecast(first_issue - timedelta(seconds=1)), [])
        published = runtime.get_weather_forecast(first_issue)
        self.assertTrue(published)
        self.assertTrue(all(item["issued_at_utc"] <= first_issue.isoformat().replace("+00:00", "Z") for item in published))

    def test_baseline_efficiency_jitter_is_bounded_and_frozen(self) -> None:
        model = self.config["quality"]["instrument_efficiency"]
        low, high = float(model["jitter_minimum"]), float(model["jitter_maximum"])
        # Isolate the baseline from ALL-scope event multipliers by generating without events.
        calm_config = json.loads(json.dumps(self.config))
        for definition in calm_config["events"].values():
            definition["count"] = 0
        slots = load_slots(OUTPUT / "slots.csv")
        calm = generate_weather(calm_config, slots, [])
        observable = [item.instrument_efficiency for item in calm if item.is_observable]
        self.assertTrue(observable)
        self.assertTrue(all(low <= value <= high for value in observable))
        self.assertGreater(len(set(observable)), 1, "jitter should vary per slot")
        reloaded = load_weather(OUTPUT / "weather.csv")
        self.assertEqual(
            [item.instrument_efficiency for item in reloaded],
            [item.instrument_efficiency for item in self.weather],
            "generated efficiency is frozen in weather.csv",
        )

    def test_at_most_one_instrument_fault_active_at_any_time(self) -> None:
        faults = [event for event in self.events if event.condition == "instrument_fault"]
        self.assertTrue(faults, "reference scenario should ship fault events")
        for index, first in enumerate(faults):
            for second in faults[index + 1:]:
                self.assertFalse(
                    first.actual_start_utc < second.actual_end_utc and first.actual_end_utc > second.actual_start_utc,
                    f"{first.event_id} overlaps {second.event_id}",
                )

    def test_shipped_faults_are_region_scoped(self) -> None:
        self.assertEqual(set(self.config["events"]["instrument_fault"]["scope_weights"]), {"REGION_SET"})
        faults = [event for event in self.events if event.condition == "instrument_fault"]
        self.assertTrue(faults)
        self.assertTrue(all(event.spatial_scope_type == "REGION_SET" for event in faults))

    def test_shipped_fault_persists_until_survey_end(self) -> None:
        faults = [event for event in self.events if event.condition == "instrument_fault"]
        self.assertEqual(len(faults), 1)
        last_slot = load_slots(OUTPUT / "slots.csv")[-1]
        self.assertEqual(faults[0].actual_end_utc, last_slot.end_utc)

    def test_persistent_event_config_validation(self) -> None:
        import copy
        import json
        import tempfile

        # Legacy shape (no persists_until_survey_end): duration_slots still applies.
        legacy = copy.deepcopy(self.config)
        legacy["events"]["instrument_fault"] = {
            "count": 2, "duration_slots": [4, 8], "scope_weights": {"REGION_SET": 1.0},
            "force_close": False, "seeing_multiplier": 1.0, "transparency_multiplier": 1.0,
            "sky_quality_multiplier": 1.0, "instrument_efficiency_multiplier": 0.5,
        }
        slots = load_slots(OUTPUT / "slots.csv")
        events = generate_events(legacy, slots, [])
        legacy_faults = [event for event in events if event.condition == "instrument_fault"]
        self.assertEqual(len(legacy_faults), 2)
        for event in legacy_faults:
            duration_slots = (event.actual_end_utc - event.actual_start_utc).total_seconds() / 900
            self.assertTrue(4 <= duration_slots <= 8, "absent persists_until_survey_end, duration_slots applies")

        def expect_invalid(config):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump(config, handle)
                path = Path(handle.name)
            with self.assertRaises(ValueError):
                load_config(path)

        bad_multi = copy.deepcopy(self.config)
        bad_multi["events"]["instrument_fault"]["count"] = 2
        expect_invalid(bad_multi)  # two persistent faults can never satisfy the no-overlap rule

        bad_missing = copy.deepcopy(self.config)
        bad_missing["events"]["instrument_fault"] = {"count": 1, "scope_weights": {"REGION_SET": 1.0}}
        expect_invalid(bad_missing)  # neither persists flag nor duration_slots

    def test_efficiency_multiplier_range_is_respected(self) -> None:
        import copy

        tuned = copy.deepcopy(self.config)
        tuned["events"]["instrument_fault"]["instrument_efficiency_multiplier_range"] = [0.2, 0.2]
        slots = load_slots(OUTPUT / "slots.csv")
        events = generate_events(tuned, slots, [])
        faults = [event for event in events if event.condition == "instrument_fault"]
        self.assertTrue(faults)
        for event in faults:
            self.assertAlmostEqual(event.instrument_efficiency_multiplier, 0.2, places=9)
            # severity is still drawn and recorded, but no longer mediates the multiplier
            self.assertTrue(0.55 <= event.severity <= 1.0)

    def test_severity_scaling_still_default_for_other_events(self) -> None:
        import copy

        tuned = copy.deepcopy(self.config)
        tuned["events"]["cold_wave"]["severity_range"] = [0.9, 0.9]
        slots = load_slots(OUTPUT / "slots.csv")
        events = generate_events(tuned, slots, [])
        for event in events:
            if event.condition == "cold_wave":
                self.assertAlmostEqual(event.severity, 0.9, places=9)
                self.assertAlmostEqual(event.instrument_efficiency_multiplier, 1.0 + 0.9 * (0.68 - 1.0), places=9)

    def test_invalid_multiplier_range_rejected(self) -> None:
        import copy
        import json
        import tempfile

        for key in ("severity_range", "instrument_efficiency_multiplier_range"):
            for bad_range in ([0.0, 1.0], [0.9, 0.5], [1.0], "wide"):
                bad = copy.deepcopy(self.config)
                bad["events"]["cold_wave"][key] = bad_range
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                    json.dump(bad, handle)
                    path = Path(handle.name)
                with self.assertRaises(ValueError, msg=f"{key}={bad_range}"):
                    load_config(path)

    def test_forecasts_never_mention_instrument_faults(self) -> None:
        self.assertTrue(self.forecasts)
        self.assertFalse(any(item.condition == "instrument_fault" for item in self.forecasts))

    def test_instrument_fault_applies_only_inside_scope(self) -> None:
        base = next(item for item in self.weather if item.is_observable)
        target = next(iter(self.geometry.tiles.values()))
        control = max(
            self.geometry.tiles.values(),
            key=lambda item: abs(item.ra_deg - target.ra_deg),
        )
        event = WeatherEvent(
            "TEST", "instrument_fault", base.timestamp_utc,
            base.end_utc, "SKY_CAP_ICRS",
            {"ra_deg": target.ra_deg, "dec_deg": target.dec_deg, "radius_deg": 1.0},
            1.0, False, 1.0, 1.0, 1.0, 0.10,
        )
        runtime = WeatherSimulator(self.weather, [], [event], self.config, self.geometry)
        floor = float(self.config["quality"]["instrument_efficiency"]["minimum"])
        affected = runtime.get_effective_conditions(base.slot_id, target.tile_id)
        unaffected = runtime.get_effective_conditions(base.slot_id, control.tile_id)
        self.assertAlmostEqual(affected["instrument_efficiency"], max(floor, base.instrument_efficiency * 0.10), places=6)
        self.assertAlmostEqual(unaffected["instrument_efficiency"], base.instrument_efficiency, places=6)
        self.assertEqual(affected["active_event_ids"], ["TEST"])
        self.assertEqual(unaffected["active_event_ids"], [])

    def test_snapshot_view_hides_instrument_faults(self) -> None:
        base = next(item for item in self.weather if item.is_observable)
        target = next(iter(self.geometry.tiles.values()))
        event = WeatherEvent(
            "TEST", "instrument_fault", base.timestamp_utc,
            base.end_utc, "SKY_CAP_ICRS",
            {"ra_deg": target.ra_deg, "dec_deg": target.dec_deg, "radius_deg": 1.0},
            1.0, False, 1.0, 1.0, 1.0, 0.10,
        )
        runtime = WeatherSimulator(self.weather, [], [event], self.config, self.geometry)
        hidden = runtime.get_effective_conditions(base.slot_id, target.tile_id, include_instrument_faults=False)
        self.assertAlmostEqual(hidden["instrument_efficiency"], base.instrument_efficiency, places=6)
        self.assertEqual(hidden["active_event_ids"], [])


if __name__ == "__main__":
    unittest.main()
