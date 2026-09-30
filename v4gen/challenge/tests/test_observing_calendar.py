from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "reference" / "config" / "calendar_config.json"
REFERENCE = ROOT / "reference" / "outputs" / "reference"

from challenge.observing_calendar import (  # noqa: E402
    build_calendar,
    generate,
    load_config,
    load_nights,
    load_slots,
    sun_altitude_deg,
    validate_calendar,
)


class ObservingCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)

    def test_calendar_uses_seasonally_variable_solar_nights(self):
        nights, slots = build_calendar(self.config)
        report = validate_calendar(nights, slots)
        self.assertEqual(report["night_count"], 180)
        self.assertGreater(report["maximum_slots_per_night"], report["minimum_slots_per_night"])
        self.assertNotEqual({night.slot_count for night in nights}, {48})
        threshold = float(self.config["site"]["sun_altitude_limit_deg"])
        for night in nights[::29]:
            self.assertLessEqual(
                sun_altitude_deg(night.observing_start_utc, self.config),
                threshold + 0.2,
            )
            self.assertLessEqual(
                sun_altitude_deg(night.observing_end_utc, self.config),
                threshold + 0.2,
            )

    def test_reference_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(CONFIG, Path(first))
            generate(CONFIG, Path(second))
            for filename in ("night_calendar.csv", "slots.csv", "calendar_metadata.json"):
                expected = (Path(first) / filename).read_bytes()
                self.assertEqual(expected, (Path(second) / filename).read_bytes())
                self.assertEqual(expected, (REFERENCE / filename).read_bytes())

    def test_reference_files_are_internally_consistent(self):
        nights = load_nights(REFERENCE / "night_calendar.csv")
        slots = load_slots(REFERENCE / "slots.csv")
        report = validate_calendar(nights, slots)
        metadata = json.loads((REFERENCE / "calendar_metadata.json").read_text())
        self.assertEqual(report["slot_count"], metadata["row_counts"]["slots"])
        self.assertEqual(sum(night.slot_count for night in nights), len(slots))


if __name__ == "__main__":
    unittest.main()
