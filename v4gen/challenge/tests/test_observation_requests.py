from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.observation_request_simulator import (
    ObservationRequestSimulator,
    generate,
    load_request_tiles,
    load_requests,
)


CONFIG = ROOT / "reference" / "config"
OUTPUT = ROOT / "reference" / "outputs" / "reference"


class ObservationRequestTests(unittest.TestCase):
    def test_schedule_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            generated = Path(raw)
            generate(CONFIG / "request_config.json", OUTPUT / "night_calendar.csv", OUTPUT / "tiles.csv", generated)
            for name in ("observation_requests.csv", "observation_request_tiles.csv"):
                self.assertEqual(
                    hashlib.sha256((generated / name).read_bytes()).digest(),
                    hashlib.sha256((OUTPUT / name).read_bytes()).digest(),
                )

    def test_request_relations_and_deadlines_are_valid(self) -> None:
        requests = load_requests(OUTPUT / "observation_requests.csv")
        links = load_request_tiles(OUTPUT / "observation_request_tiles.csv")
        self.assertEqual({item.request_id for item in requests}, set(links))
        for item in requests:
            self.assertIn(item.deadline_class, {"ONE_WEEK", "TWO_WEEKS", "ONE_MONTH"})
            self.assertIn(item.completion_mode, {"ALL", "AT_LEAST_N"})
            self.assertLessEqual(item.required_tile_count, len(links[item.request_id]))
            self.assertGreater(item.deadline_utc, item.available_from_utc)

    def test_as_of_publication_has_no_future_leakage(self) -> None:
        runtime = ObservationRequestSimulator.from_files(
            OUTPUT / "observation_requests.csv", OUTPUT / "observation_request_tiles.csv"
        )
        first = runtime.requests[0]
        self.assertEqual(runtime.get_observation_requests(first.issued_at_utc - timedelta(seconds=1)), [])
        rows = runtime.get_observation_requests(first.issued_at_utc)
        self.assertTrue(rows)
        self.assertTrue(all(item["issued_at_utc"] <= first.public_dict()["issued_at_utc"] for item in rows))


if __name__ == "__main__":
    unittest.main()
