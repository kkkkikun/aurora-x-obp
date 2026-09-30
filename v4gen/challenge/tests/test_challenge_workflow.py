from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.challenge_workflow import ChallengeWorkflow, load_workflow_config
from challenge.run_challenge import JsonLineAgentProcess


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class ChallengeWorkflowTests(unittest.TestCase):
    def test_config_has_one_global_clock_and_no_per_decision_limit(self) -> None:
        config = load_workflow_config(ROOT / "reference" / "config" / "workflow_config.json")
        self.assertGreater(config["global_wallclock_seconds"], 0)
        self.assertIsNone(config["per_decision_timeout_seconds"])
        self.assertIsNone(config["synthetic_timeout_action"])

    def test_response_finishing_at_cutoff_is_ignored_atomically(self) -> None:
        clock = FakeClock()
        workflow = ChallengeWorkflow(ROOT / "reference", clock)

        def provider(snapshot, deadline):
            clock.value += 0.5
            return {"action": "wait", "reason": "fake-clock boundary test"}

        result = workflow.run(provider, wallclock_seconds=2.0)
        self.assertEqual(result["termination_reason"], "global_wallclock_expired")
        self.assertEqual(result["committed_action_count"], 3)
        self.assertTrue(result["ignored_in_flight_response"])
        self.assertEqual(len(workflow.committed), 3)
        self.assertNotIn("runtime", result["score_report"]["score"]["penalties"])

    def test_initial_publication_reports_the_clock_that_applies(self) -> None:
        clock = FakeClock()
        workflow = ChallengeWorkflow(ROOT / "reference", clock)
        published = []

        def provider(snapshot, deadline):
            clock.value += 10.0
            return {"action": "wait", "reason": "clock test"}

        provider.publish_initial = published.append
        result = workflow.run(provider, wallclock_seconds=18000.0)
        self.assertEqual(published[0]["global_wallclock_seconds"], 18000.0)
        self.assertEqual(result["initial_publication"]["global_wallclock_seconds"], 18000.0)

    def test_malformed_in_flight_response_is_not_committed_or_replaced(self) -> None:
        clock = FakeClock()
        workflow = ChallengeWorkflow(ROOT / "reference", clock)

        def provider(snapshot, deadline):
            return {"action": "teleport"}

        result = workflow.run(provider, wallclock_seconds=2.0)
        self.assertEqual(result["termination_reason"], "agent_error")
        self.assertEqual(result["committed_action_count"], 0)
        self.assertEqual(workflow.committed, [])

    def test_process_transport_terminates_in_flight_agent_at_global_cutoff(self) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        provider = JsonLineAgentProcess(
            [sys.executable, "-B", str(ROOT / "tests" / "fixtures" / "slow_jsonl_agent.py")]
        )
        started = time.monotonic()
        try:
            result = workflow.run(provider, wallclock_seconds=0.25)
        finally:
            provider.close(force=True)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(result["termination_reason"], "global_wallclock_expired")
        self.assertTrue(result["ignored_in_flight_response"])
        self.assertEqual(result["committed_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
