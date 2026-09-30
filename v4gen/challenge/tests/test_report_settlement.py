"""Hidden tile anomaly tags (MP-049): silent score multipliers and report settlement."""

from __future__ import annotations

import json
import unittest
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.challenge_workflow import ChallengeWorkflow
from challenge.contracts import TILE_ANOMALY_COLUMNS, TILE_COLUMNS, read_exact_csv, write_exact_csv
from challenge.scenario_builder import generate_tile_anomalies
from challenge.scoring_core import (
    ChallengeScorer,
    Decision,
    Report,
    score_files,
)


CONFIG = ROOT / "reference" / "config"
OUTPUT = ROOT / "reference" / "outputs" / "reference"


def complete_start(scorer: ChallengeScorer, tile_id: str):
    tile = scorer.tiles[tile_id]
    for index, slot in enumerate(scorer.slots[:-1]):
        remaining = tile.nominal_exptime_seconds
        cursor = index
        okay = True
        while remaining > 0:
            current = scorer.slots[cursor]
            if current.night_id != slot.night_id or not scorer._tile_legal(tile, current.timestamp_utc):
                okay = False
                break
            weather = scorer.weather.get_effective_conditions(current.slot_id, tile_id)
            if not weather["is_observable"]:
                okay = False
                break
            remaining -= min(remaining, current.duration_seconds)
            cursor += 1
        if okay:
            return index
    raise AssertionError(f"no complete opportunity found for {tile_id}")


def observe(scorer: ChallengeScorer, tile_id: str, index: int, decision_id: str = "D1") -> dict:
    scorer.slot_index = index
    scorer.offset_seconds = 0
    return scorer.apply_decision(Decision(decision_id, scorer.slots[index].slot_id, "observe", tile_id, "DARK", "", "test"))


class TileAnomalyGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_matches_shipped_file(self) -> None:
        config = json.loads((CONFIG / "tile_config.json").read_text(encoding="utf-8"))
        tile_ids = sorted(row["tile_id"] for row in read_exact_csv(OUTPUT / "tiles.csv", TILE_COLUMNS))
        first = generate_tile_anomalies(config, tile_ids)
        second = generate_tile_anomalies(config, tile_ids)
        self.assertEqual(first, second)
        shipped = read_exact_csv(OUTPUT / "tile_anomalies.csv", TILE_ANOMALY_COLUMNS)
        self.assertEqual(first, shipped)
        self.assertTrue(any(row["anomaly_tag"] == "nova" for row in first))
        self.assertTrue(any(row["anomaly_tag"] == "reddening" for row in first))

    def test_absent_config_section_means_no_tags(self) -> None:
        self.assertEqual(generate_tile_anomalies({"seed": 7}, ["T1", "T2"]), [])
        self.assertEqual(generate_tile_anomalies({"seed": 7, "anomaly_tags": {}}, ["T1", "T2"]), [])


class AnomalyScoringTests(unittest.TestCase):
    def test_nova_factor_applies_silently(self) -> None:
        tagged = ChallengeScorer.from_files(ROOT / "reference")
        tile_id = next(t for t, tags in tagged.tile_anomalies.items() if "nova" in tags)
        index = complete_start(tagged, tile_id)
        plain = ChallengeScorer.from_files(ROOT / "reference")
        plain.tile_anomalies = {}
        scored = observe(tagged, tile_id, index)
        baseline = observe(plain, tile_id, index)
        self.assertEqual(scored["outcome"], "completed")
        self.assertEqual(baseline["outcome"], "completed")
        ratio = (scored["base_science_score"] + scored["program_bonus_score"]) / (baseline["base_science_score"] + baseline["program_bonus_score"])
        self.assertAlmostEqual(ratio, 1.5, places=4)

    def test_both_tags_multiply(self) -> None:
        tagged = ChallengeScorer.from_files(ROOT / "reference")
        tile_id = next(iter(tagged.tile_anomalies))
        tagged.tile_anomalies = {tile_id: frozenset({"nova", "reddening"})}
        index = complete_start(tagged, tile_id)
        plain = ChallengeScorer.from_files(ROOT / "reference")
        plain.tile_anomalies = {}
        scored = observe(tagged, tile_id, index)
        baseline = observe(plain, tile_id, index)
        ratio = (scored["base_science_score"] + scored["program_bonus_score"]) / (baseline["base_science_score"] + baseline["program_bonus_score"])
        self.assertAlmostEqual(ratio, 1.5 * 0.8, places=4)

    def test_published_tile_value_stays_untagged(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        snapshot = workflow.decision_snapshot(1)
        tagged_tiles = set(workflow.scorer.tile_anomalies)
        shown = {row["tile_id"]: row["tile_science_value"] for row in snapshot["candidate_tiles"]}
        for tile_id in tagged_tiles & set(shown):
            self.assertEqual(shown[tile_id], round(workflow.scorer.tile_values[tile_id], 6))


class ReportSettlementTests(unittest.TestCase):
    def _fault(self, scorer: ChallengeScorer):
        return next(event for event in scorer.weather.events if event.condition == "instrument_fault")

    def test_fault_report_ledger_all_branches(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        fault = self._fault(scorer)
        quiet = min(event.actual_start_utc for event in scorer.weather.events if event.condition == "instrument_fault") - timedelta(hours=1)

        # First misreport is free; the second since the last correct report costs the penalty.
        first = scorer.apply_report(Report("R1", "Instrument_Failure", ""), quiet)
        self.assertEqual(first["result"], "misreport")
        self.assertEqual(scorer.penalties["fault_misreport"], 0.0)
        second = scorer.apply_report(Report("R2", "Instrument_Failure", ""), quiet)
        self.assertEqual(second["result"], "misreport")
        self.assertEqual(scorer.penalties["fault_misreport"], 100.0)

        # Correct report: acknowledges the fault and resets the counter.
        inside = fault.actual_start_utc + timedelta(hours=1)
        correct = scorer.apply_report(Report("R3", "Instrument_Failure", ""), inside)
        self.assertEqual(correct["result"], "correct")
        self.assertEqual(scorer.misreport_count, 0)
        self.assertEqual(scorer.fault_correct_reports, 1)
        self.assertIn(fault.event_id, scorer.fault_acknowledgements)

        # Re-reporting the acknowledged, still-unrepaired fault is neutral.
        again = scorer.apply_report(Report("R4", "Instrument_Failure", ""), inside + timedelta(minutes=30))
        self.assertEqual(again["result"], "neutral")
        self.assertEqual(scorer.misreport_count, 0)
        self.assertEqual(scorer.penalties["fault_misreport"], 100.0)

        # The counter was reset: one new free misreport once the repair has ended the fault.
        repair_at = scorer.fault_acknowledgements[fault.event_id]
        after = repair_at + timedelta(hours=1)
        free = scorer.apply_report(Report("R5", "Instrument_Failure", ""), after)
        self.assertEqual(free["result"], "misreport")
        self.assertEqual(scorer.penalties["fault_misreport"], 100.0)

    def test_repair_truncates_fault_multiplier(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        scorer._repair_duration = timedelta(hours=1)  # shorter than any shipped fault duration
        fault = self._fault(scorer)
        slot = next(
            item for item in scorer.weather.weather
            if item.is_observable and fault.actual_start_utc + timedelta(hours=2) < item.timestamp_utc and item.end_utc < fault.actual_end_utc
        )
        tile_id = next(t for t in scorer.geometry.tiles if scorer.weather._applies(fault, t, slot))
        with_fault = scorer.weather.get_effective_conditions(slot.slot_id, tile_id)
        self.assertIn(fault.event_id, with_fault["active_event_ids"])
        as_of = fault.actual_start_utc + timedelta(minutes=30)
        scorer.apply_report(Report("R1", "Instrument_Failure", ""), as_of)
        self.assertEqual(scorer.weather.end_overrides[fault.event_id], as_of + timedelta(hours=1))
        repaired = scorer.weather.get_effective_conditions(slot.slot_id, tile_id)
        hidden = scorer.weather.get_effective_conditions(slot.slot_id, tile_id, include_instrument_faults=False)
        self.assertNotIn(fault.event_id, repaired["active_event_ids"])
        self.assertEqual(repaired["instrument_efficiency"], hidden["instrument_efficiency"])

    def test_unrepaired_fault_applies_until_survey_end(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        fault = self._fault(scorer)
        self.assertEqual(fault.actual_end_utc, scorer.slots[-1].end_utc)
        slot = next(item for item in reversed(scorer.weather.weather) if item.is_observable)
        tile_id = next(t for t in scorer.geometry.tiles if scorer.weather._applies(fault, t, slot))
        final = scorer.weather.get_effective_conditions(slot.slot_id, tile_id)
        self.assertIn(fault.event_id, final["active_event_ids"])
        baseline = scorer.weather.get_effective_conditions(slot.slot_id, tile_id, include_instrument_faults=False)
        self.assertLess(final["instrument_efficiency"], baseline["instrument_efficiency"])
        # Repair is the only early exit: an acknowledged report truncates it at as_of + x2.
        as_of = fault.actual_start_utc + timedelta(hours=1)
        scorer.apply_report(Report("R1", "Instrument_Failure", ""), as_of)
        repaired = scorer.weather.get_effective_conditions(slot.slot_id, tile_id)
        self.assertNotIn(fault.event_id, repaired["active_event_ids"])

    def test_tag_reports_settle_at_finalize_with_dedup(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        nova_tile = next(t for t, tags in scorer.tile_anomalies.items() if "nova" in tags)
        plain_tile = next(t for t in scorer.tiles if t not in scorer.tile_anomalies)
        when = scorer.slots[0].timestamp_utc
        self.assertEqual(scorer.apply_report(Report("R1", "NOVA", nova_tile), when)["result"], "recorded")
        self.assertEqual(scorer.apply_report(Report("R2", "NOVA", nova_tile), when)["result"], "duplicate_ignored")
        scorer.apply_report(Report("R3", "Reddening", nova_tile), when)  # wrong tag on the same tile settles independently
        scorer.apply_report(Report("R4", "NOVA", plain_tile), when)
        report = scorer.finalize()
        settlements = {(row["tile_id"], row["tag"]): row for row in report["reports"]["tag_settlements"]}
        self.assertEqual(len(settlements), 3)
        self.assertEqual(settlements[(nova_tile, "nova")]["delta"], 100.0)
        self.assertEqual(settlements[(nova_tile, "reddening")]["delta"], -150.0)
        self.assertEqual(settlements[(plain_tile, "nova")]["delta"], -150.0)
        self.assertEqual(report["score"]["report_reward"], 100.0)
        self.assertEqual(report["score"]["penalties"]["wrong_tag_report"], 300.0)
        expected_total = (
            report["score"]["base_science"] + report["score"]["program_bonus"] + report["score"]["request_reward"]
            + report["score"]["coverage_bonus"] + report["score"]["report_reward"] - sum(report["score"]["penalties"].values())
        )
        self.assertAlmostEqual(report["score"]["total"], expected_total, places=4)

    def test_score_files_settles_report_rows_in_decisions_csv(self) -> None:
        import tempfile

        scorer = ChallengeScorer.from_files(ROOT / "reference")
        nova_tile = next(t for t, tags in scorer.tile_anomalies.items() if "nova" in tags)
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            decisions = tmp / "decisions.csv"
            decisions.write_text(
                "decision_id,slot_id,action,tile_id,program,request_id,reason\n"
                f"D000001,{scorer.slots[0].slot_id},wait,,,,test\n"
                f"D000002,{scorer.slots[0].slot_id},report_nova,{nova_tile},,,tag report rides the trace\n",
                encoding="utf-8",
            )
            out = tmp / "report.json"
            result = score_files(ROOT / "reference", decisions, out, "trace_complete")
            self.assertEqual(result["score"]["report_reward"], 100.0)
            self.assertNotIn("reports", result["input_sha256"])
            self.assertEqual(result["input_sha256"]["decisions"], __import__("hashlib").sha256(decisions.read_bytes()).hexdigest())
            row = next(a for a in result["actions"] if a["action"] == "report_nova")
            self.assertEqual(row["outcome"], "report_recorded")
            # Without the report row the same trace scores identically minus the settlement.
            plain = tmp / "plain.csv"
            plain.write_text(decisions.read_text(encoding="utf-8").splitlines()[0] + "\n" + f"D000001,{scorer.slots[0].slot_id},wait,,,,test\n", encoding="utf-8")
            plain_result = score_files(ROOT / "reference", plain, tmp / "plain.json", "trace_complete")
            self.assertEqual(plain_result["score"]["report_reward"], 0.0)
            self.assertAlmostEqual(plain_result["score"]["total"], result["score"]["total"] - 100.0, places=4)


    def test_load_decisions_validates_report_rows(self) -> None:
        import tempfile

        from challenge.scoring_core import load_decisions

        header = "decision_id,slot_id,action,tile_id,program,request_id,reason\n"
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)

            def write(row: str) -> Path:
                path = tmp / "d.csv"
                path.write_text(header + row + "\n", encoding="utf-8")
                return path

            slot = "N20260907-S001"
            self.assertEqual(len(load_decisions(write(f"D1,{slot},report_instrument_failure,,,,x"))), 1)
            self.assertEqual(len(load_decisions(write(f"D1,{slot},report_nova,T00001,,,x"))), 1)
            for bad in (
                f"D1,{slot},report_nova,,,,x",                    # tag report needs a tile
                f"D1,{slot},report_instrument_failure,T00001,,,x",  # fault report names no tile
                f"D1,{slot},report_nova,T00001,DARK,,x",          # no program on report rows
                f"D1,{slot},report_nova,T00001,,RQ0001,x",        # no request on report rows
                f"D1,{slot},report_something,T00001,,,x",         # unknown action
            ):
                with self.assertRaises(ValueError, msg=bad):
                    load_decisions(write(bad))


if __name__ == "__main__":
    unittest.main()
