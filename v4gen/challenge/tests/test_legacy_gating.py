"""Pre-anomaly scenarios must keep their exact contract under the gated mechanics.

The gate is the score config: without the anomaly sections a scenario scores,
validates and publishes exactly as it did before the anomaly release.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from challenge.challenge_workflow import ChallengeWorkflow
from challenge.contracts import anomaly_mechanics_enabled
from challenge.scoring_core import ChallengeScorer, Decision, load_decisions

KIT_DEV_REFERENCE = Path(__file__).resolve().parents[2] / "starter_kit" / "scenarios" / "dev-reference"
V2_REFERENCE = Path(__file__).resolve().parents[1] / "reference"


def test_the_gate_is_the_score_config():
    legacy = json.loads((KIT_DEV_REFERENCE / "config" / "score_config.json").read_text(encoding="utf-8"))
    mechanics = json.loads((V2_REFERENCE / "config" / "score_config.json").read_text(encoding="utf-8"))
    assert not anomaly_mechanics_enabled(legacy)
    assert anomaly_mechanics_enabled(mechanics)


def test_legacy_scorer_keeps_the_pre_anomaly_rules():
    scorer = ChallengeScorer.from_files(KIT_DEV_REFERENCE)
    assert not scorer.mechanics
    slot = scorer.slots[0]
    # A report action is an unknown action on a legacy scenario, never a settlement.
    outcome = scorer.apply_decision(Decision("D000001", slot.slot_id, "report_nova", "T00001", "", "", ""))
    assert outcome["outcome"] == "unknown_action"
    assert scorer.penalties["invalid_action"] > 0


def test_legacy_results_files_reject_report_rows(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text(
        "decision_id,slot_id,action,tile_id,program,request_id,reason\n"
        "D000001,S000001,report_nova,T00001,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observe or wait"):
        load_decisions(path, allow_reports=False)
    assert len(load_decisions(path, allow_reports=True)) == 1


def test_legacy_workflow_speaks_the_v1_contract():
    workflow = ChallengeWorkflow(root=KIT_DEV_REFERENCE)
    assert not workflow.mechanics
    snapshot = workflow.decision_snapshot(1)
    assert snapshot["schema_version"] == "decision-snapshot-v2"
    assert "tile_last_finished" not in snapshot
    assert "fault_status" not in snapshot
    assert "instrument_efficiency" in snapshot["current_site_weather"]
    # Both protocol generations are accepted; reports are dropped with a count, not booked.
    decision, reports, dropped = workflow._decision_from_response(
        1, snapshot["cursor"]["slot_id"],
        {"protocol_version": "participant-agent-protocol-v1", "action": "wait", "reason": ""})
    assert decision.action == "wait" and reports == [] and dropped == 0
    decision, reports, dropped = workflow._decision_from_response(
        2, snapshot["cursor"]["slot_id"],
        {"protocol_version": "participant-agent-protocol-v2", "action": "wait", "reason": "",
         "reports": [{"kind": "NOVA", "tile_id": "T00001"}]})
    assert decision.action == "wait" and reports == [] and dropped == 1


def test_mechanics_workflow_speaks_the_v2_contract():
    workflow = ChallengeWorkflow(root=V2_REFERENCE)
    assert workflow.mechanics
    snapshot = workflow.decision_snapshot(1)
    assert snapshot["schema_version"] == "decision-snapshot-v3"
    assert "tile_last_finished" in snapshot
    assert "instrument_efficiency" not in snapshot["current_site_weather"]
