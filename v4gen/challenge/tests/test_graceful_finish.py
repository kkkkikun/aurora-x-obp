"""The starter-kit transport's graceful finish: one last message, EOF, bounded grace."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from challenge.run_challenge import FINISH_GRACE_SECONDS, JsonLineAgentProcess


def agent_code(finish_branch: str) -> str:
    return """import json, sys
for line in sys.stdin:
    m = json.loads(line)
    t = m["message_type"]
    if t == "initialize":
        continue
    if t == "finish":
%s
    else:
        print(json.dumps({"protocol_version": m["protocol_version"],
                          "message_type": "decision_response",
                          "decision_sequence": m["decision_sequence"],
                          "action": "wait"}), flush=True)
print("agent saw stdin EOF", file=sys.stderr, flush=True)
""" % finish_branch


class GracefulFinishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def start_agent(self, finish_branch: str) -> JsonLineAgentProcess:
        script = self.dir / "agent.py"
        script.write_text(agent_code(finish_branch))
        provider = JsonLineAgentProcess([sys.executable, "-B", str(script)])
        provider.publish_initial({"schema_version": "initial-publication-v2"})
        response = provider({"decision_sequence": 1}, time.monotonic() + 10)
        self.assertEqual(response["action"], "wait")
        return provider

    def test_finish_message_then_eof_then_self_exit(self) -> None:
        record = self.dir / "record.json"
        provider = self.start_agent(
            "        import pathlib\n"
            f"        pathlib.Path({str(record)!r}).write_text(json.dumps(m, sort_keys=True))\n"
            "        print('agent wrote summary', file=sys.stderr, flush=True)\n"
            "        sys.exit(0)\n"
        )
        process = provider.process
        started = time.monotonic()
        provider.finish("survey_complete", 1, grace_seconds=10)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(process.returncode, 0)
        self.assertIsNone(provider.process)
        message = json.loads(record.read_text())
        self.assertEqual(message["message_type"], "finish")
        self.assertEqual(message["protocol_version"], provider.protocol_version)
        self.assertEqual(
            message["payload"],
            {"termination_reason": "survey_complete", "last_decision_sequence": 1, "grace_seconds": 10},
        )

    def test_ignoring_agent_is_stopped_after_the_grace_period(self) -> None:
        provider = self.start_agent(
            "        import time as t\n"
            "        print('agent ignores finish', file=sys.stderr, flush=True)\n"
            "        t.sleep(120)\n"
        )
        process = provider.process
        started = time.monotonic()
        provider.finish("global_wallclock_expired", 1, grace_seconds=1)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 1)
        self.assertLess(elapsed, 15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIsNone(provider.process)

    def test_crashing_agent_does_not_fail_the_run(self) -> None:
        provider = self.start_agent("        raise SystemExit(3)\n")
        process = provider.process
        provider.finish("survey_complete", 1, grace_seconds=5)  # must not raise
        self.assertEqual(process.returncode, 3)
        self.assertIsNone(provider.process)

    def test_agent_that_exits_on_stdin_eof_skips_the_wait(self) -> None:
        provider = self.start_agent("        continue  # no reply, no exit: the loop ends on EOF\n")
        process = provider.process
        started = time.monotonic()
        provider.finish("survey_complete", 1, grace_seconds=10)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(process.returncode, 0)

    def test_finish_is_a_noop_after_close(self) -> None:
        provider = self.start_agent("        sys.exit(0)\n")
        provider.close(force=True)
        provider.finish("survey_complete", 1, grace_seconds=1)  # must not raise
        self.assertIsNone(provider.process)

    def test_default_grace_matches_the_platform(self) -> None:
        self.assertEqual(FINISH_GRACE_SECONDS, 30)


if __name__ == "__main__":
    unittest.main()
