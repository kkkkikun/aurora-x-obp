#!/usr/bin/env python3
"""Local transport for the starter kit: runs a participant agent process against a v4 card through the
platform's own engine adapter, challenge/v4_workflow.py (vendored byte for byte).

  engine -> agent   initialize        (once, no reply; v4-initialize-v1)
  engine -> agent   decision_request  (v4-decision-snapshot-v1), agent replies with one decision_response
  engine -> agent   finish            (once at the end, no reply; v4-finish-v1), then stdin is closed

What this file adds is only what the platform's container transport does: a JSON-Lines subprocess with
its own folder and environment, the global deadline on every read and write (an agent still computing
at the deadline is stopped at once), and the finish message with the 30 s grace. Scoring, validation,
the wall-clock rule and all payloads come from v4_workflow / v4_runner.
Pure standard library.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Mapping

from .v4_workflow import PROTOCOL_VERSION, V4_SCENARIO_PATH, V4Workflow

MAX_RESPONSE_BYTES = 512 * 1024
FINISH_GRACE_SECONDS = 30.0


class GlobalDeadlineExpired(TimeoutError):
    """The global wall clock ran out while talking to the agent (v4_workflow expects a TimeoutError)."""


class AgentProtocolError(ValueError):
    """The agent's output is not a valid JSON-Lines response."""


def scenario_path(card_dir: Path) -> Path:
    return Path(card_dir) / V4_SCENARIO_PATH


def load_card(card_dir: Path) -> dict:
    """Public card metadata (task_card) from config/v4_scenario.json."""
    config = json.loads(scenario_path(card_dir).read_text(encoding="utf-8"))
    card = dict(config.get("task_card") or {"card_id": str(config["name"])})
    card.setdefault("scenario_slug", str(config["name"]))
    return card


class AgentProcess:
    """A persistent JSON-Lines subprocess. Reads and writes run on helper threads so the global
    deadline holds on every platform (including Windows) even for multi-megabyte messages."""

    def __init__(self, command: list[str], *, cwd: Path, env: Mapping[str, str], stderr=None,
                 initialization_seconds: float = 30.0):
        self.command = list(command)
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.stderr = stderr
        self.initialization_seconds = initialization_seconds
        self.process: subprocess.Popen | None = None
        self._lines: "queue.Queue[bytes | Exception | None]" = queue.Queue()

    def start(self) -> None:
        if self.process is not None:
            return
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
                 else {"start_new_session": True})
        self.process = subprocess.Popen(self.command, cwd=str(self.cwd), env=self.env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, **group)
        threading.Thread(target=self._pump, args=(self.process.stdout,), daemon=True).start()

    def _pump(self, stream) -> None:
        buffer = bytearray()
        while True:
            try:
                chunk = stream.read(65536) if hasattr(stream, "read") else b""
            except (OSError, ValueError):
                chunk = b""
            if not chunk:
                self._lines.put(None)
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, _, rest = bytes(buffer).partition(b"\n")
                buffer = bytearray(rest)
                self._lines.put(line)
            if len(buffer) > MAX_RESPONSE_BYTES:
                self._lines.put(AgentProtocolError(f"response line exceeds {MAX_RESPONSE_BYTES} bytes"))
                return

    def send(self, message: Mapping, deadline: float) -> None:
        self.start()
        data = (json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        failure: list[BaseException] = []

        def write() -> None:
            try:
                self.process.stdin.write(data)
                self.process.stdin.flush()
            except BaseException as exc:  # noqa: BLE001 - a closed pipe when the agent exits
                failure.append(exc)

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            self.close(force=True)
            raise GlobalDeadlineExpired()
        if failure:
            raise AgentProtocolError(f"agent exited before reading the next message (exit code {self.process.poll()})")

    def receive(self, deadline: float) -> dict:
        try:
            item = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            self.close(force=True)  # still computing at the deadline: stopped at once, no finish message
            raise GlobalDeadlineExpired() from None
        if item is None:
            raise AgentProtocolError(f"agent exited without a response (exit code {self.process.poll() if self.process else None})")
        if isinstance(item, Exception):
            raise item
        if len(item) > MAX_RESPONSE_BYTES:
            raise AgentProtocolError(f"response line exceeds {MAX_RESPONSE_BYTES} bytes")
        try:
            payload = json.loads(item.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError) as exc:
            raise AgentProtocolError("stdout must carry one JSON object per line; print logs to stderr") from exc
        if not isinstance(payload, dict):
            raise AgentProtocolError("response must be a JSON object")
        return payload

    def publish_initial(self, payload: Mapping) -> None:
        self.send({"protocol_version": PROTOCOL_VERSION, "message_type": "initialize", "payload": payload},
                  time.monotonic() + self.initialization_seconds)

    def request(self, sequence: int, payload: Mapping, deadline: float) -> dict:
        self.send({"protocol_version": PROTOCOL_VERSION, "message_type": "decision_request",
                   "decision_sequence": sequence, "payload": payload}, deadline)
        return self.receive(deadline)

    def finish(self, payload: Mapping, grace_seconds: float = FINISH_GRACE_SECONDS) -> None:
        """One final `finish` line, stdin EOF, then up to grace_seconds to exit. Never raises."""
        process = self.process
        if process is None:
            return
        grace_deadline = time.monotonic() + grace_seconds
        if process.poll() is None:
            try:
                self.send({"protocol_version": PROTOCOL_VERSION, "message_type": "finish", "payload": dict(payload)},
                          grace_deadline)
            except Exception:  # noqa: BLE001
                pass
        if self.process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=max(0.0, grace_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        self.close()

    def close(self, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=10)
                else:
                    import signal  # noqa: PLC0415
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.process = None


def run_card(card_dir: Path, agent: AgentProcess, output_dir: Path, *, wallclock_seconds: float | None = None,
             grace_seconds: float = FINISH_GRACE_SECONDS) -> dict:
    """Run one card like the platform's colocated v4 path (trusted_engine.run_v4_session)."""
    workflow = V4Workflow(Path(card_dir))

    def decide(message: dict, deadline: float) -> dict:
        agent.send(message, deadline)
        return agent.receive(deadline)

    def initialize(payload: dict) -> None:
        agent.send({"protocol_version": PROTOCOL_VERSION, "message_type": "initialize", "payload": payload},
                   time.monotonic() + agent.initialization_seconds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = workflow.run(decide, output_dir, wallclock_seconds=wallclock_seconds, initialize=initialize)
    init_error = result.pop("initialization_error", None)
    if init_error is None:
        agent.finish({**V4Workflow.finish_payload(result), "termination_reason": result["termination_reason"],
                      "last_decision_sequence": result["last_decision_sequence"], "grace_seconds": grace_seconds},
                     grace_seconds=grace_seconds)
    else:
        agent.close(force=True)
    error = None
    if init_error is not None:
        error = f"initialize: {init_error}"
    elif result["termination_reason"] == "agent_error":
        error = f"decision {result['decision_requests']}: {result['termination_detail']}"
    return {**result, "error": error, "card_id": load_card(card_dir)["card_id"]}
