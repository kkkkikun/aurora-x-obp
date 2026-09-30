"""Anomaly detection and calibrated reporting in the minimal agent (MP-051)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from anomaly_detection import AnomalyDetector
from challenge.challenge_workflow import ChallengeWorkflow
from challenge.contracts import DECISION_COLUMNS, read_exact_csv
from challenge.scenario_builder import generate_scenario
from challenge.scoring_core import score_files
from decision_graph import MinimalDecisionAgent


INITIAL = {
    "scoring_contract": {"score_config": {"program_bonus": {"DARK": 0.25, "BRIGHT": 0.15, "BACKUP": 0.08}}},
    "tile_catalog": {"tiles": [
        {"tile_id": "T1", "ra_deg": 10.0, "dec_deg": 20.0},
        {"tile_id": "T2", "ra_deg": 40.0, "dec_deg": -10.0},
        {"tile_id": "T3", "ra_deg": 200.0, "dec_deg": 40.0},
    ]},
}


def snapshot(feedback=None, fault_status=None, now="2026-10-01T03:00:00Z", candidates=None):
    snap = {
        "cursor": {"timestamp_utc": now, "night_id": "N" + now[:10].replace("-", "")},
        "tile_last_finished": feedback,
        "candidate_tiles": candidates or [],
    }
    if fault_status is not None:
        snap["fault_status"] = fault_status
    return snap


def observe_then_feedback(detector: AnomalyDetector, tile_id: str, expected: float, realized: float, now="2026-10-01T03:00:00Z"):
    detector.note_observation(tile_id, expected)
    return detector.process_snapshot(snapshot(feedback={"tile_id": tile_id, "score": realized}, now=now))


class DetectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = AnomalyDetector(INITIAL)

    def test_normal_deviation_produces_no_report(self) -> None:
        for realized in (100.0, 95.0, 105.0, 91.0, 110.0):
            self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, realized), [])

    def test_nova_signature_reported_when_reads_dominate(self) -> None:
        for day, hour, realized in ((1, 3, 149.0), (1, 5, 149.1), (1, 7, 149.2), (2, 3, 148.9)):
            self.assertEqual(
                observe_then_feedback(self.detector, "T1", 100.0, realized, now=f"2026-10-{day:02d}T{hour:02d}:00:00Z"), []
            )
        reports = observe_then_feedback(self.detector, "T1", 100.0, 147.0, now="2026-10-03T03:00:00Z")
        self.assertEqual(reports, [{"kind": "NOVA", "tile_id": "T1"}])  # 5/5 in band, 3 nights
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 150.0, now="2026-10-04T03:00:00Z"), [])  # already reported
        self.assertEqual(self.detector.bests["T1"], 150.0)

    def test_weather_edge_never_dominates(self) -> None:
        # Isolated in-band dips among normal reads must not reach 80% dominance.
        for day, realized in enumerate((80.0, 101.0, 99.0, 81.0, 100.0, 102.0), start=1):
            self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, realized, now=f"2026-10-{day:02d}T03:00:00Z"), [])
        self.assertNotIn(("T1", "Reddening"), self.detector._reported_tags)

    def test_tag_survives_an_occasional_out_of_band_read(self) -> None:
        # 0.92 = reddening 0.8 with a weather edge on top: out of band, but the
        # permanent tag still dominates the history.
        for day, realized in ((1, 80.0), (2, 92.0), (2, 81.0), (3, 79.0)):
            self.assertEqual(observe_then_feedback(self.detector, "T2", 100.0, realized, now=f"2026-10-{day:02d}T03:00:00Z"), [])
        reports = observe_then_feedback(self.detector, "T2", 100.0, 82.0, now="2026-10-03T07:00:00Z")
        self.assertEqual(reports, [{"kind": "Reddening", "tile_id": "T2"}])  # 4/5 in band, 3 nights

    def test_interrupted_zero_carries_no_signal(self) -> None:
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 0.0), [])
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 0.0), [])
        self.assertEqual(self.detector._recent_ratios, [])  # interruptions are not collapse evidence
        self.assertEqual(self.detector._tag_reads, {})

    def test_fault_needs_two_collapses_inside_the_window(self) -> None:
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 30.0), [])   # one collapse is not enough
        self.assertEqual(observe_then_feedback(self.detector, "T3", 100.0, 100.0), [])  # interleaved normal read
        reports = observe_then_feedback(self.detector, "T2", 100.0, 20.0)
        self.assertEqual(reports, [{"kind": "Instrument_Failure"}])
        # Pending: no re-report until the platform answers.
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 30.0), [])
        self.assertEqual(observe_then_feedback(self.detector, "T2", 100.0, 20.0), [])

    def test_fault_status_fault_silences_until_repair(self) -> None:
        observe_then_feedback(self.detector, "T1", 100.0, 30.0)
        observe_then_feedback(self.detector, "T2", 100.0, 20.0)
        self.assertEqual(self.detector._fault_pending, True)
        self.detector.process_snapshot(snapshot(fault_status={
            "status": "fault", "spatial_scope_type": "REGION_SET", "spatial_scope_payload": {"region_ids": ["R01"]},
            "repair_complete_utc": "2026-10-03T03:00:00Z",
        }, now="2026-10-02T03:00:00Z"))
        self.assertFalse(self.detector._fault_pending)
        # Collapses while the acknowledged fault is under repair are not re-reported.
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 30.0, now="2026-10-02T04:00:00Z"), [])
        self.assertEqual(observe_then_feedback(self.detector, "T2", 100.0, 20.0, now="2026-10-02T05:00:00Z"), [])
        # After the repair time, fresh collapse evidence reports again (a new fault).
        reports = observe_then_feedback(self.detector, "T1", 100.0, 30.0, now="2026-10-04T03:00:00Z")
        if not reports:  # the window may need a second post-repair collapse
            reports = observe_then_feedback(self.detector, "T2", 100.0, 20.0, now="2026-10-04T04:00:00Z")
        self.assertEqual(reports, [{"kind": "Instrument_Failure"}])

    def test_instrument_normal_notice_rearms_evidence(self) -> None:
        observe_then_feedback(self.detector, "T1", 100.0, 30.0)
        observe_then_feedback(self.detector, "T2", 100.0, 20.0)
        self.assertTrue(self.detector._fault_pending)
        self.detector.process_snapshot(snapshot(fault_status={"status": "normal", "reference_report_id": "R000001"}))
        self.assertFalse(self.detector._fault_pending)
        self.assertEqual(observe_then_feedback(self.detector, "T1", 100.0, 30.0), [])

    def test_fault_scope_filtering(self) -> None:
        self.detector.process_snapshot(snapshot(fault_status={
            "status": "fault", "spatial_scope_type": "REGION_SET", "spatial_scope_payload": {"region_ids": ["R01"]},
            "repair_complete_utc": "2026-10-03T03:00:00Z",
        }, now="2026-10-02T03:00:00Z"))
        candidates = [
            {"tile_id": "T1", "region_id": "R01"},
            {"tile_id": "T2", "region_id": "R02"},
        ]
        filtered = self.detector.filter_fault_scope(snapshot(now="2026-10-02T04:00:00Z", candidates=candidates))
        self.assertEqual([c["tile_id"] for c in filtered["candidate_tiles"]], ["T2"])
        # The filter lifts once repair completes.
        lifted = self.detector.filter_fault_scope(snapshot(now="2026-10-04T04:00:00Z", candidates=candidates))
        self.assertEqual(len(lifted["candidate_tiles"]), 2)

    def test_sky_cap_scope_uses_tile_coordinates(self) -> None:
        self.detector.process_snapshot(snapshot(fault_status={
            "status": "fault", "spatial_scope_type": "SKY_CAP_ICRS",
            "spatial_scope_payload": {"ra_deg": 10.5, "dec_deg": 20.5, "radius_deg": 5.0},
            "repair_complete_utc": "2026-10-03T03:00:00Z",
        }, now="2026-10-02T03:00:00Z"))
        candidates = [{"tile_id": "T1", "region_id": "R01"}, {"tile_id": "T3", "region_id": "R03"}]
        filtered = self.detector.filter_fault_scope(snapshot(now="2026-10-02T04:00:00Z", candidates=candidates))
        self.assertEqual([c["tile_id"] for c in filtered["candidate_tiles"]], ["T3"])


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


def run_agent(scenario: Path):
    workflow = ChallengeWorkflow(root=scenario, clock=FakeClock())
    agent = MinimalDecisionAgent(workflow.initial_publication(), model=None, top_k=12)
    result = workflow.run(lambda snap, deadline: agent.decide(snap), wallclock_seconds=1e12)
    return workflow, result


class ReportingEndToEndTests(unittest.TestCase):
    """End-to-end detection: shipped reference scenario from just before the fault, and a synthetic clean scenario."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        clean_base = base / "clean-base"
        shutil.copytree(ROOT / "reference" / "config", clean_base / "config")
        weather_config = json.loads((clean_base / "config" / "weather_config.json").read_text(encoding="utf-8"))
        weather_config["events"]["instrument_fault"]["count"] = 0
        (clean_base / "config" / "weather_config.json").write_text(json.dumps(weather_config, indent=2) + "\n", encoding="utf-8")
        tile_config = json.loads((clean_base / "config" / "tile_config.json").read_text(encoding="utf-8"))
        tile_config.pop("anomaly_tags", None)
        (clean_base / "config" / "tile_config.json").write_text(json.dumps(tile_config, indent=2) + "\n", encoding="utf-8")
        cls.clean = base / "clean"
        generate_scenario(cls.clean, scenario_id="clean", seed=77, days=14, start_date="2026-10-05", global_wallclock_seconds=7200, base=clean_base)
        assert not (cls.clean / "outputs" / "reference" / "tile_anomalies.csv").exists()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_clean_scenario_produces_zero_reports(self) -> None:
        workflow, result = run_agent(self.clean)
        self.assertEqual(result["termination_reason"], "survey_complete")
        self.assertFalse(any(row.action.startswith("report_") for row in workflow.committed))
        report = result["score_report"]
        self.assertEqual(report["score"]["report_reward"], 0.0)
        self.assertEqual(report["reports"]["tag_settlements"], [])

    def test_anomalous_scenario_detects_and_reports(self) -> None:
        # Start the run just before the fault onset: every tile is uncompleted, so
        # the completion phase observes the fault region during the fault.
        workflow = ChallengeWorkflow(root=ROOT / "reference", clock=FakeClock())
        fault = next(event for event in workflow.scorer.weather.events if event.condition == "instrument_fault")
        before = max(
            index for index, slot in enumerate(workflow.scorer.slots)
            if slot.timestamp_utc < fault.actual_start_utc
        )
        workflow.scorer.slot_index = before
        workflow.scorer.offset_seconds = 0
        agent = MinimalDecisionAgent(workflow.initial_publication(), model=None, top_k=12)
        result = workflow.run(lambda snap, deadline: agent.decide(snap), wallclock_seconds=1e12)
        self.assertEqual(result["termination_reason"], "survey_complete")
        report = result["score_report"]
        self.assertGreaterEqual(report["reports"]["fault_correct_reports"], 1)
        settlements = report["reports"]["tag_settlements"]
        self.assertTrue(settlements, "tagged tiles should be detected")
        self.assertTrue(all(row["delta"] > 0 for row in settlements), f"conservative bands must not misfire: {settlements}")
        self.assertNotIn("fault_misreport", report["score"]["penalties"])
        # The committed decisions.csv carries every report as a report_* row (full
        # replay equivalence on a from-start run is covered in test_report_channel.py;
        # a teleported start cannot replay, since decisions.csv has no record of the
        # skipped prefix).
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            workflow.write_outputs(out, result)
            rows = read_exact_csv(out / "decisions.csv", DECISION_COLUMNS)
            report_rows = [row for row in rows if row["action"].startswith("report_")]
            committed_report_rows = [row for row in workflow.committed if row.action.startswith("report_")]
            self.assertEqual(len(report_rows), len(committed_report_rows))
            self.assertGreater(len(report_rows), 0)


if __name__ == "__main__":
    unittest.main()
