from __future__ import annotations

import json
import os
import sys
from unittest import mock
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.challenge_workflow import ChallengeWorkflow
from protocol import ProtocolError, decision_response, parse_platform_message
from challenge.run_challenge import JsonLineAgentProcess


class ParticipantProtocolTests(unittest.TestCase):
    def test_publication_exposes_exact_score_contract_and_tile_values(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        publication = workflow.initial_publication()
        self.assertEqual(publication["schema_version"], "initial-publication-v2")
        self.assertEqual(
            publication["scoring_contract"]["score_config"], workflow.scorer.config
        )
        first = publication["tile_catalog"]["tiles"][0]
        self.assertEqual(
            first["tile_science_value"],
            round(workflow.scorer.tile_values[first["tile_id"]], 6),
        )

    def test_snapshot_weather_never_carries_instrument_efficiency(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        snapshot = workflow.decision_snapshot(1)
        self.assertNotIn("instrument_efficiency", snapshot["current_site_weather"])
        self.assertTrue(snapshot["candidate_tiles"])
        for candidate in snapshot["candidate_tiles"]:
            self.assertNotIn("instrument_efficiency", candidate["effective_weather"])

    def test_snapshot_has_public_current_progress_without_future_truth(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        snapshot = workflow.decision_snapshot(1)
        self.assertEqual(snapshot["schema_version"], "decision-snapshot-v3")
        self.assertNotIn("weather_events", snapshot)
        self.assertIn("flexible_completed_by_region", snapshot["progress"])
        self.assertTrue(snapshot["candidate_tiles"])
        minimum_altitude = float(
            workflow.scorer.geometry.tile_config["geometry"]["minimum_altitude_deg"]
        )
        candidate = snapshot["candidate_tiles"][0]
        for key in (
            "tile_science_value",
            "window_start_utc",
            "window_end_utc",
            "geometry",
            "effective_weather",
        ):
            self.assertIn(key, candidate)
        self.assertTrue(
            all(
                float(item["geometry"]["altitude_deg"]) >= minimum_altitude
                for item in snapshot["candidate_tiles"]
            )
        )
        for request in snapshot["active_requests"]:
            self.assertIn("satisfied_tile_count", request)
            self.assertIn("is_complete", request)
            for requirement in request["tile_requirements"]:
                self.assertIn("completed_visits", requirement)
                self.assertIn("remaining_visits", requirement)

    def test_protocol_rejects_wrong_sequence_and_wraps_response(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        snapshot = workflow.decision_snapshot(1)
        message = {
            "protocol_version": "participant-agent-protocol-v2",
            "message_type": "decision_request",
            "decision_sequence": 2,
            "payload": snapshot,
        }
        with self.assertRaises(ProtocolError):
            parse_platform_message(message)
        response = decision_response(
            1,
            {
                "action": "wait",
                "reason": "test",
                "decision_source": "deterministic",
            },
        )
        self.assertEqual(response["message_type"], "decision_response")
        self.assertEqual(response["decision_sequence"], 1)
        self.assertEqual(response["tile_id"], "")

    def test_minimal_process_receives_initialization_then_one_request(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        # platform note: `env KEY=VALUE cmd` does not exist on Windows; the variable is set in this process instead
        env_patch = mock.patch.dict(os.environ, {"MODEL_PROVIDER": "deterministic"})
        env_patch.start()
        provider = JsonLineAgentProcess(
            [
                sys.executable,
                "-B",
                str(ROOT / "participant_agent" / "minimal_agent.py"),
            ]
        )
        try:
            provider.publish_initial(workflow.initial_publication())
            response = provider(workflow.decision_snapshot(1), time.monotonic() + 10)
        finally:
            env_patch.stop()
            provider.close(force=True)
        self.assertEqual(response["protocol_version"], "participant-agent-protocol-v2")
        self.assertEqual(response["message_type"], "decision_response")
        self.assertEqual(response["decision_sequence"], 1)
        self.assertIn(response["action"], {"observe", "wait"})


if __name__ == "__main__":
    unittest.main()
