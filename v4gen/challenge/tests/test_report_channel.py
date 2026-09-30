"""Report channel and realized-score feedback in the participant protocol (MP-050)."""

from __future__ import annotations

import unittest
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.challenge_workflow import ChallengeWorkflow
from challenge.contracts import DECISION_COLUMNS, format_utc, read_exact_csv
from challenge.participant_agent.protocol import decision_response
from challenge.scoring_core import score_files


def complete_start(scorer, tile_id: str):
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


def drive(workflow: ChallengeWorkflow, sequence: int, response: dict):
    """One decision step through the public response path."""
    slot = workflow.scorer.current_slot()
    decision, reports, dropped = workflow._decision_from_response(sequence, slot.slot_id, response)
    result, outcomes = workflow._commit(sequence, decision, reports)
    return result, outcomes, dropped


class ScoreFeedbackTests(unittest.TestCase):
    def test_tile_last_finished_null_before_first_completion(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        snapshot = workflow.decision_snapshot(1)
        self.assertIsNone(snapshot["tile_last_finished"])
        self.assertNotIn("fault_status", snapshot)

    def test_tile_last_finished_carries_realized_score(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        tile_id = next(iter(workflow.scorer.tiles))
        workflow.scorer.slot_index = complete_start(workflow.scorer, tile_id)
        result, _, _ = drive(workflow, 1, {"action": "observe", "tile_id": tile_id, "program": "DARK"})
        self.assertEqual(result["outcome"], "completed")
        snapshot = workflow.decision_snapshot(2)
        feedback = snapshot["tile_last_finished"]
        self.assertEqual(feedback["tile_id"], tile_id)
        self.assertAlmostEqual(
            feedback["score"], result["base_science_score"] + result["program_bonus_score"], places=6
        )
        # Waits do not replace the feedback entry.
        drive(workflow, 2, {"action": "wait"})
        self.assertEqual(workflow.decision_snapshot(3)["tile_last_finished"], feedback)

    def test_interrupted_observation_reports_zero(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        scorer = workflow.scorer
        # A multi-slot exposure started on the last slot of a night breaks at dawn and scores zero.
        selected = None
        for night_id, night_slots in scorer.geometry.slots_by_night.items():
            last = night_slots[-1]
            index = scorer.slot_indices[last.slot_id]
            for tile in scorer.tiles.values():
                if tile.nominal_exptime_seconds > last.duration_seconds and scorer._tile_legal(tile, last.timestamp_utc):
                    if scorer.weather.get_effective_conditions(last.slot_id, tile.tile_id)["is_observable"]:
                        selected = tile.tile_id, index
                        break
            if selected:
                break
        self.assertIsNotNone(selected)
        tile_id, index = selected
        scorer.slot_index = index
        result, _, _ = drive(workflow, 1, {"action": "observe", "tile_id": tile_id, "program": "DARK"})
        self.assertEqual(result["outcome"], "geometry_or_night_interrupted")
        self.assertEqual(workflow.decision_snapshot(2)["tile_last_finished"], {"tile_id": tile_id, "score": 0.0})


class FaultStatusPublicationTests(unittest.TestCase):
    def _report_fault(self):
        workflow = ChallengeWorkflow(ROOT / "reference")
        fault = next(event for event in workflow.scorer.weather.events if event.condition == "instrument_fault")
        slot = next(
            item for item in workflow.scorer.weather.weather
            if fault.actual_start_utc < item.end_utc and fault.actual_end_utc > item.timestamp_utc
        )
        workflow.scorer.slot_index = workflow.scorer.slot_indices[slot.slot_id]
        _, outcomes, _ = drive(workflow, 1, {"action": "wait", "reports": [{"kind": "Instrument_Failure"}]})
        self.assertEqual(outcomes[0]["result"], "correct")
        return workflow, fault, outcomes[0]

    def _night_start_index(self, workflow: ChallengeWorkflow, moment_ge, moment_lt=None) -> int:
        for night_id in sorted(workflow.scorer.geometry.nights):
            first = workflow.scorer.geometry.slots_by_night[night_id][0]
            if first.timestamp_utc >= moment_ge and (moment_lt is None or first.timestamp_utc < moment_lt):
                return workflow.scorer.slot_indices[first.slot_id]
        raise AssertionError("no night start in the requested window")

    def test_correct_report_flows_into_time_gated_fault_status(self) -> None:
        workflow, fault, outcome = self._report_fault()
        feed = workflow._fault_feed[0]
        self.assertEqual(feed["event_id"], fault.event_id)
        repair_at = feed["repair_complete_utc"]
        self.assertEqual(format_utc(repair_at), outcome["repair_complete_utc"])
        # Latency gate: nothing to publish before x1 days have passed.
        self.assertIsNone(workflow._fault_status(feed["reported_at"]))
        # Night-start publication during the repair window.
        index = self._night_start_index(workflow, feed["reported_at"] + timedelta(days=workflow._fault_latency_days), repair_at)
        workflow.scorer.slot_index = index
        workflow.scorer.offset_seconds = 0
        snapshot = workflow.decision_snapshot(2)
        status = snapshot["fault_status"]
        self.assertEqual(status["status"], "fault")
        self.assertEqual(status["event_id"], fault.event_id)
        self.assertEqual(status["spatial_scope_type"], fault.spatial_scope_type)
        self.assertEqual(status["spatial_scope_payload"], fault.spatial_scope_payload)
        self.assertAlmostEqual(status["instrument_efficiency_multiplier"], fault.instrument_efficiency_multiplier, places=6)
        self.assertEqual(status["repair_complete_utc"], format_utc(repair_at))
        # Nightly cadence: mid-night snapshots do not carry the field.
        workflow.scorer.slot_index = index + 1
        self.assertNotIn("fault_status", workflow.decision_snapshot(3))
        # Re-published at each later night start while unrepaired; gone once repair completes.
        # (Widen the repair window so it spans several night starts — placeholders x1=1d, x2=2d hold only one.)
        workflow._fault_feed[0]["repair_complete_utc"] = feed["reported_at"] + timedelta(days=4)
        next_night = self._night_start_index(workflow, workflow.scorer.slots[index].timestamp_utc + timedelta(hours=20))
        workflow.scorer.slot_index = next_night
        self.assertIn("fault_status", workflow.decision_snapshot(4))
        after_repair = self._night_start_index(workflow, workflow._fault_feed[0]["repair_complete_utc"])
        workflow.scorer.slot_index = after_repair
        self.assertNotIn("fault_status", workflow.decision_snapshot(5))

    def test_misreport_gets_one_shot_instrument_normal_notice(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        _, outcomes, _ = drive(workflow, 1, {"action": "wait", "reports": [{"kind": "Instrument_Failure"}]})
        self.assertEqual(outcomes[0]["result"], "misreport")
        self.assertEqual(workflow.scorer.misreport_total, 1)
        self.assertEqual(workflow.scorer.penalties["fault_misreport"], 0.0)  # first misreport is free
        notice = workflow._normal_notices[0]
        self.assertIsNone(workflow._fault_status(notice["respond_at"] - timedelta(hours=1)))
        index = self._night_start_index(workflow, notice["respond_at"])
        workflow.scorer.slot_index = index
        workflow.scorer.offset_seconds = 0
        snapshot = workflow.decision_snapshot(2)
        self.assertEqual(snapshot["fault_status"]["status"], "normal")
        self.assertEqual(snapshot["fault_status"]["reference_report_id"], "D000002")
        # One-shot: the next night start no longer carries it.
        later = self._night_start_index(workflow, workflow.scorer.slots[index].timestamp_utc + timedelta(hours=20))
        workflow.scorer.slot_index = later
        self.assertNotIn("fault_status", workflow.decision_snapshot(3))


class ReportValidationTests(unittest.TestCase):
    def test_malformed_entries_dropped_action_kept(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        tile_id = next(iter(workflow.scorer.tiles))
        response = {
            "action": "wait",
            "reports": [
                {"kind": "BOGUS"},                              # unknown kind
                {"kind": "NOVA"},                               # missing tile
                {"kind": "NOVA", "tile_id": "NOPE"},            # unknown tile
                {"kind": "Instrument_Failure", "tile_id": tile_id},  # fault report must not name a tile
                "not-a-dict",
                {"kind": "NOVA", "tile_id": tile_id},           # accepted
                {"kind": "NOVA", "tile_id": tile_id},           # duplicate tolerated, settlement dedupes
            ],
        }
        decision, reports, dropped = workflow._decision_from_response(1, workflow.scorer.current_slot().slot_id, response)
        self.assertEqual(dropped, 5)
        self.assertEqual(reports, [{"kind": "NOVA", "tile_id": tile_id}] * 2)
        self.assertEqual(decision.action, "wait")
        _, outcomes = workflow._commit(1, decision, reports)
        self.assertEqual([item["result"] for item in outcomes], ["recorded", "duplicate_ignored"])

    def test_reports_never_move_the_cursor(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        tile_id = next(iter(workflow.scorer.tiles))
        before = (workflow.scorer.slot_index, workflow.scorer.offset_seconds)
        response = {"action": "wait", "reports": [
            {"kind": "NOVA", "tile_id": tile_id},
            {"kind": "Reddening", "tile_id": tile_id},
            {"kind": "Instrument_Failure"},
        ]}
        result, outcomes, _ = drive(workflow, 1, response)
        self.assertEqual(len(outcomes), 3)
        # The only cursor movement came from the wait itself (one slot), not the reports.
        self.assertEqual((workflow.scorer.slot_index, workflow.scorer.offset_seconds), (before[0] + 1, 0))
        report_rows = [row for row in workflow.committed if row.action.startswith("report_")]
        self.assertEqual(len(report_rows), 3)
        self.assertTrue(all(row.slot_id == result["slot_id"] for row in report_rows))
        # Report rows share the decisions.csv row-id sequence with their carrier.
        self.assertEqual([row.decision_id for row in workflow.committed], ["D000001", "D000002", "D000003", "D000004"])
        self.assertEqual(report_rows[0].action, "report_nova")
        self.assertEqual(report_rows[2].action, "report_instrument_failure")

    def test_decision_response_envelope_carries_reports(self) -> None:
        without = decision_response(3, {"action": "wait"})
        self.assertNotIn("reports", without)
        with_reports = decision_response(3, {"action": "wait"}, reports=[{"kind": "NOVA", "tile_id": "T00001"}])
        self.assertEqual(with_reports["reports"], [{"kind": "NOVA", "tile_id": "T00001"}])


class ReportArtifactTests(unittest.TestCase):
    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 0.001
            return self.t

    def test_decisions_csv_report_rows_replay_to_the_live_total(self) -> None:
        import tempfile

        from challenge.challenge_workflow import GlobalDeadlineExpired

        workflow = ChallengeWorkflow(ROOT / "reference", clock=self._FakeClock())
        nova_tile = next(t for t, tags in workflow.scorer.tile_anomalies.items() if "nova" in tags)
        state = {"n": 0}

        def provider(snapshot, deadline):
            state["n"] += 1
            if state["n"] > 5:
                raise GlobalDeadlineExpired  # deterministic cutoff, no wall-clock dependence
            if state["n"] == 1:
                return {"action": "wait", "reports": [
                    {"kind": "NOVA", "tile_id": nova_tile},
                    {"kind": "Instrument_Failure"},  # no active fault at survey start: free misreport
                ]}
            return {"action": "wait"}

        result = workflow.run(provider, wallclock_seconds=1e12)
        self.assertEqual(result["termination_reason"], "global_wallclock_expired")
        self.assertEqual(result["committed_action_count"], 7)  # 5 carrier decisions + 2 report rows
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            workflow.write_outputs(out, result)
            rows = read_exact_csv(out / "decisions.csv", DECISION_COLUMNS)
            report_rows = [row for row in rows if row["action"].startswith("report_")]
            self.assertEqual(len(report_rows), 2)
            self.assertEqual(report_rows[0]["action"], "report_nova")
            self.assertEqual(report_rows[1]["action"], "report_instrument_failure")
            replay = score_files(ROOT / "reference", out / "decisions.csv", out / "score_report.json",
                                 "global_wallclock_expired")
            live = result["score_report"]
            self.assertEqual(replay["score"], live["score"])
            self.assertEqual(replay["reports"], live["reports"])
            self.assertEqual(replay["score"]["report_reward"], 100.0)
            self.assertEqual(replay["reports"]["fault_misreports"], 1)
            self.assertNotIn("fault_misreport", replay["score"]["penalties"])

    def test_run_without_reports_writes_a_plain_decisions_csv(self) -> None:
        import tempfile

        from challenge.challenge_workflow import GlobalDeadlineExpired

        workflow = ChallengeWorkflow(ROOT / "reference", clock=self._FakeClock())
        state = {"n": 0}

        def provider(snapshot, deadline):
            state["n"] += 1
            if state["n"] > 3:
                raise GlobalDeadlineExpired
            return {"action": "wait"}

        result = workflow.run(provider, wallclock_seconds=1e12)
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            workflow.write_outputs(out, result)
            self.assertFalse((out / "report.csv").exists())
            rows = read_exact_csv(out / "decisions.csv", DECISION_COLUMNS)
            self.assertTrue(all(not row["action"].startswith("report_") for row in rows))
            replay = score_files(ROOT / "reference", out / "decisions.csv", out / "score_report.json",
                                 result["termination_reason"])
            self.assertEqual(replay["score"], result["score_report"]["score"])


if __name__ == "__main__":
    unittest.main()
