from __future__ import annotations

import csv
import copy
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reference" / "config"
REFERENCE = ROOT / "reference" / "outputs" / "reference"

from challenge.contracts import TILE_WINDOW_COLUMNS  # noqa: E402
from challenge.observing_calendar import load_nights  # noqa: E402
from challenge.tile_geometry_simulator import (  # noqa: E402
    TileGeometrySimulator,
    _moon_equatorial_deg,
    _sun_equatorial_deg,
    _angular_separation_deg,
    generate_catalog,
    geometry_sample,
    geometry_sample_without_lunar,
    load_tiles,
)


class TileGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = TileGeometrySimulator.from_files(
            REFERENCE / "tiles.csv",
            CONFIG / "tile_config.json",
            CONFIG / "calendar_config.json",
            REFERENCE / "night_calendar.csv",
            REFERENCE / "slots.csv",
        )

    def test_catalog_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for output in (Path(first), Path(second)):
                generate_catalog(
                    CONFIG / "tile_config.json",
                    CONFIG / "calendar_config.json",
                    REFERENCE / "night_calendar.csv",
                    output,
                )
            for filename in ("tiles.csv", "targets.csv", "catalog_metadata.json"):
                expected = (Path(first) / filename).read_bytes()
                self.assertEqual(expected, (Path(second) / filename).read_bytes())
                self.assertEqual(expected, (REFERENCE / filename).read_bytes())

    def test_catalog_region_and_target_counts(self):
        tiles = load_tiles(REFERENCE / "tiles.csv")
        self.assertEqual(len(tiles), 64)
        by_region_class = Counter((tile.region_id, tile.scheduling_class) for tile in tiles)
        for index in range(8):
            self.assertEqual(by_region_class[(f"R{index:02d}", "REQUIRED")], 2)
            self.assertEqual(by_region_class[(f"R{index:02d}", "FLEXIBLE")], 6)
        targets = Counter()
        with (REFERENCE / "targets.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                targets[(row["tile_id"], row["target_class"])] += 1
        for tile in tiles:
            self.assertEqual(tile.n_lrg, targets[(tile.tile_id, "LRG")])
            self.assertEqual(tile.n_elg, targets[(tile.tile_id, "ELG")])
            self.assertEqual(tile.n_qso, targets[(tile.tile_id, "QSO")])
            self.assertEqual(tile.n_bgs, targets[(tile.tile_id, "BGS")])

    def test_airmass_tracks_altitude_and_is_zenith_normalized(self):
        tile = next(iter(self.simulator.tiles.values()))
        night = next(iter(self.simulator.nights.values()))
        samples = [
            self.simulator.get_tile_geometry(tile.tile_id, night.observing_start_utc + timedelta(minutes=15 * index))
            for index in range(night.slot_count)
        ]
        finite = [row for row in samples if row["airmass"] != float("inf")]
        best = max(finite, key=lambda row: row["altitude_deg"])
        worst = min(finite, key=lambda row: row["altitude_deg"])
        self.assertLess(best["airmass"], worst["airmass"])
        all_samples = []
        for candidate in self.simulator.tiles.values():
            all_samples.extend(
                self.simulator.get_tile_geometry(candidate.tile_id, night.observing_start_utc + timedelta(minutes=15 * index))
                for index in range(night.slot_count)
            )
        closest = min(all_samples, key=lambda row: abs(row["altitude_deg"] - 90.0))
        self.assertAlmostEqual(closest["airmass"], 1.0, delta=0.02)

    def test_windows_use_shared_calendar_contract(self):
        first = load_nights(REFERENCE / "night_calendar.csv")[0]
        windows = self.simulator.get_tile_windows(first.night_date, 3)
        self.assertTrue(windows)
        self.assertEqual(list(windows[0]), TILE_WINDOW_COLUMNS)
        for row in windows:
            self.assertGreaterEqual(row["window_seconds"], row["nominal_exptime_seconds"])
            self.assertIn(row["night_id"], self.simulator.nights)
            self.assertGreater(row["minimum_lunar_quality_factor"], 0.0)
            self.assertLessEqual(
                row["minimum_lunar_quality_factor"],
                row["mean_lunar_quality_factor"],
            )

    def test_lunar_quality_is_continuous_and_altitude_aware(self):
        source = next(iter(self.simulator.tiles.values()))
        chosen = None
        moon_below_moment = None
        for slots in self.simulator.slots_by_night.values():
            for slot in slots:
                moment = slot.timestamp_utc + timedelta(seconds=slot.duration_seconds / 2)
                moon_ra, moon_dec = _moon_equatorial_deg(moment)
                moon_tile = replace(source, ra_deg=moon_ra, dec_deg=moon_dec)
                moon_altitude = geometry_sample_without_lunar(
                    moon_tile, moment, self.simulator.calendar_config
                )["altitude_deg"]
                if moon_altitude <= 0.0 and moon_below_moment is None:
                    moon_below_moment = moment
                sun_ra, sun_dec = _sun_equatorial_deg(moment)
                illumination = (
                    1.0
                    - math.cos(
                        math.radians(
                            _angular_separation_deg(sun_ra, sun_dec, moon_ra, moon_dec)
                        )
                    )
                ) / 2.0
                if moon_altitude > 20.0 and illumination > 0.25:
                    chosen = moment, moon_ra, moon_dec
                if chosen and moon_below_moment:
                    break
            if chosen and moon_below_moment:
                break
        self.assertIsNotNone(chosen)
        self.assertIsNotNone(moon_below_moment)
        moment, moon_ra, moon_dec = chosen
        near = replace(source, ra_deg=moon_ra, dec_deg=moon_dec)
        far = replace(source, ra_deg=(moon_ra + 180.0) % 360.0, dec_deg=-moon_dec)
        near_factor = geometry_sample(
            near, moment, self.simulator.tile_config, self.simulator.calendar_config
        )["lunar_quality_factor"]
        far_factor = geometry_sample(
            far, moment, self.simulator.tile_config, self.simulator.calendar_config
        )["lunar_quality_factor"]
        self.assertGreater(near_factor, 0.0)
        self.assertLess(near_factor, far_factor)
        self.assertLessEqual(far_factor, 1.0)
        self.assertEqual(
            geometry_sample(
                source,
                moon_below_moment,
                self.simulator.tile_config,
                self.simulator.calendar_config,
            )["lunar_quality_factor"],
            1.0,
        )

    def test_lunar_quality_does_not_cut_observable_windows(self):
        first = load_nights(REFERENCE / "night_calendar.csv")[0]
        baseline = self.simulator.get_tile_windows(first.night_date, 1)
        stressed_config = copy.deepcopy(self.simulator.tile_config)
        stressed_config["lunar_model"]["maximum_penalty"] = 0.99
        stressed_config["lunar_model"]["angular_decay_scale_deg"] = 10000.0
        stressed = TileGeometrySimulator(
            list(self.simulator.tiles.values()),
            stressed_config,
            self.simulator.calendar_config,
            list(self.simulator.nights.values()),
            [slot for slots in self.simulator.slots_by_night.values() for slot in slots],
        )
        changed = stressed.get_tile_windows(first.night_date, 1)
        def window_identity(row):
            return row["window_id"], row["window_start_utc"], row["window_end_utc"]
        self.assertEqual(
            [window_identity(row) for row in baseline],
            [window_identity(row) for row in changed],
        )

    def test_time_limited_required_tiles_are_visible(self):
        nights = load_nights(REFERENCE / "night_calendar.csv")
        required = {
            tile.tile_id
            for tile in self.simulator.tiles.values()
            if tile.scheduling_class == "REQUIRED"
        }
        visible = {
            row["tile_id"]
            for row in self.simulator.get_tile_windows(nights[0].night_date, len(nights))
            if row["tile_id"] in required
        }
        self.assertEqual(required, visible)


if __name__ == "__main__":
    unittest.main()
