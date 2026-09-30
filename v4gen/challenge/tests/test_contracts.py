from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from challenge.contracts import DECISION_COLUMNS, read_exact_csv, write_exact_csv  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_exact_csv_round_trip_and_header_order(self):
        row = {
            "decision_id": 1,
            "slot_id": "N20260907-S001",
            "action": "wait",
            "tile_id": "",
            "program": "",
            "request_id": "",
            "reason": "test",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decisions.csv"
            write_exact_csv(path, DECISION_COLUMNS, [row])
            self.assertEqual(read_exact_csv(path, DECISION_COLUMNS)[0], {
                key: str(value) for key, value in row.items()
            })

    def test_extra_csv_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("decision_id,extra\n1,x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                read_exact_csv(path, DECISION_COLUMNS)


if __name__ == "__main__":
    unittest.main()
