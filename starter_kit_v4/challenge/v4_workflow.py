#!/usr/bin/env python3
"""v4 engine adapter: participant-agent-protocol-v4 on top of ``v4_runner.run_scenario``.

Transport-neutral and pure standard library. The caller supplies ``decide(message,
deadline) -> response`` (for example one JSON-Lines exchange with a colocated container)
and, optionally, ``initialize(payload)`` (sends the initialize message). This module
builds every participant-visible message from the public part of the bundle only, runs
the global wall clock, validates response envelopes, settles the score on the valid
history on any agent failure, and writes the result files.

Bundle contract (one task card, ``V4_SCENARIO_PATH`` marks a v4 bundle)::

    config/v4_scenario.json       v4-scenario-v1; product paths relative to config/
    config/v4_fiber_config.json   fibre grid + exposure bounds (public)
    config/v4_score_config.json   the public score contract
    public/...                    targets, footprint, night calendar, bulletins, forecasts
    truth/...                     slots, weather truth, events, quake effects, stress events

Optional scenario keys read here: ``task_card`` (object, copied to initialize),
``site.sun_altitude_limit_deg`` (default -18) and ``limits.global_wallclock_seconds``
(the per-card cap; the platform may only lower it).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Mapping

from .contracts import write_text_lf
from .v4_config_check import DEFAULT_MAX_CONSECUTIVE_REPORTS
from .v4_fiber_map import FiberGrid
from .v4_runner import (
    AgentTermination,
    TERMINATION_AGENT_ERROR,
    TERMINATION_WALLCLOCK,
    load_scenario,
    run_scenario,
)

PROTOCOL_VERSION = "participant-agent-protocol-v4"
INITIALIZE_SCHEMA = "v4-initialize-v1"
SNAPSHOT_SCHEMA = "v4-decision-snapshot-v1"
FINISH_SCHEMA = "v4-finish-v1"
RESULT_SCHEMA = "v4-workflow-result-v1"
V4_SCENARIO_PATH = Path("config") / "v4_scenario.json"
# Organizer decision 2026-09-28: every card run is capped at 900 s of wall clock.
MAX_WALLCLOCK_SECONDS = 900.0
DEFAULT_SUN_ALTITUDE_LIMIT_DEG = -18.0
RESPONSE_MAX_BYTES = 512 * 1024  # enforced by the transport (project_platform.transport)
ACTIONS_FILE = "actions.jsonl"
ENVELOPE_KEYS = ("protocol_version", "message_type", "decision_sequence", "reason", "decision_source")
TARGET_COLUMNS = ["target_id", "ra_deg", "dec_deg", "target_class", "feature_flux", "science_weight", "required"]
FIBER_LAYOUT = (
    "row-major, fiber 0 bottom-left; rows along +alt, columns along +az at exposure start; "
    "gnomonic plane centred on the actual pointing"
)


class ProtocolViolation(ValueError):
    """The response envelope does not answer the current request."""


def is_v4_bundle(root: Path) -> bool:
    return (Path(root) / V4_SCENARIO_PATH).is_file()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_bundle(root: Path):
    """Load and cross-check a v4 bundle; every product must live inside the bundle."""
    root = Path(root).resolve()
    scenario_path = root / V4_SCENARIO_PATH
    config = json.loads(scenario_path.read_text(encoding="utf-8"))
    base = scenario_path.parent
    paths = [config["fiber_config"], config["score_config"], *config["products"].values()]
    if config.get("stress", {}).get("enabled"):
        paths.append(config["stress"]["stress_events_csv"])
    for value in paths:
        resolved = (base / str(value)).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"bundle product {value!r} is missing or outside the bundle")
    return load_scenario(scenario_path)


class V4Workflow:
    """One v4 card run. ``root`` is an extracted bundle (see the module docstring)."""

    def __init__(self, root: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.root = Path(root)
        self.scenario_path = self.root / V4_SCENARIO_PATH
        self.scenario = validate_bundle(self.root)
        self.config = self.scenario.config
        self.clock = clock

    # --- public messages -------------------------------------------------------------

    def wallclock_budget(self, requested: float | None) -> float:
        cap = float(self.config.get("limits", {}).get("global_wallclock_seconds", MAX_WALLCLOCK_SECONDS))
        cap = min(cap, MAX_WALLCLOCK_SECONDS)
        budget = cap if requested is None else min(float(requested), cap)
        if not budget > 0:
            raise ValueError("wallclock budget must be positive")
        return budget

    def initialize_payload(self, wallclock_seconds: float) -> dict:
        scenario = self.scenario
        config = self.config
        base = self.scenario_path.parent
        products = config["products"]
        site = config["site"]
        fiber = scenario.fiber_config
        grid = FiberGrid.from_config(fiber)
        footprint: dict[str, list[list[float]]] = {}
        for row in sorted(_read_csv(base / products["footprint_csv"]),
                          key=lambda item: (item["component_id"], int(item["vertex_index"]))):
            footprint.setdefault(row["component_id"], []).append([float(row["ra_deg"]), float(row["dec_deg"])])
        nights = [
            {
                "night_id": row["night_id"],
                "night_date": row["night_date"],
                "observing_start_utc": row["observing_start_utc"],
                "observing_end_utc": row["observing_end_utc"],
                "slot_count": int(row["slot_count"]),
            }
            for row in _read_csv(base / products["night_calendar_csv"])
        ]
        slot_seconds = int(round((scenario.slots[0].end_utc - scenario.slots[0].start_utc).total_seconds()))
        task_card = dict(config.get("task_card") or {"card_id": str(config["name"])})
        return {
            "schema_version": INITIALIZE_SCHEMA,
            "task_card": task_card,
            "site": {
                "name": str(site.get("name", "")),
                "latitude_deg": float(site["latitude_deg"]),
                "longitude_deg": float(site["longitude_deg"]),
                "utc_offset_hours": float(site["utc_offset_hours"]),
                "sun_altitude_limit_deg": float(site.get("sun_altitude_limit_deg", DEFAULT_SUN_ALTITUDE_LIMIT_DEG)),
                "minimum_altitude_deg": float(config["minimum_altitude_deg"]),
            },
            "survey": {
                "start_utc": _utc(scenario.survey_start),
                "end_utc": _utc(scenario.survey_end),
                "slot_seconds": slot_seconds,
                "nights": nights,
            },
            "instrument": {
                "n_fibers": grid.n_fibers,
                "grid_side": grid.n_side,
                "fiber_area_deg2": grid.fiber_area_deg2,
                "gap_deg": grid.gap_deg,
                "glass_side_deg": round(grid.fiber_side_deg, 6),
                "pitch_deg": round(grid.pitch_deg, 6),
                "fov_side_deg": round(grid.fov_side_deg, 6),
                "layout": FIBER_LAYOUT,
                "exposure": {
                    "min_duration_seconds": int(fiber["exposure"]["min_duration_seconds"]),
                    "max_duration_seconds": int(fiber["exposure"]["max_duration_seconds"]),
                },
            },
            "scoring": dict(scenario.score_config),
            "footprint": [{"component_id": key, "vertices": value} for key, value in sorted(footprint.items())],
            "targets": {
                "columns": list(TARGET_COLUMNS),
                "rows": [[target[column] for column in TARGET_COLUMNS] for target in scenario.targets],
            },
            "limits": {
                "global_wallclock_seconds": wallclock_seconds,
                "max_consecutive_reports": scenario.score_config["reporting"].get(
                    "max_consecutive_reports", DEFAULT_MAX_CONSECUTIVE_REPORTS
                ),
                "response_max_bytes": RESPONSE_MAX_BYTES,
                "decision_timeout": "global only (no per-decision timeout)",
            },
        }

    @staticmethod
    def initialize_message(payload: Mapping) -> dict:
        return {"protocol_version": PROTOCOL_VERSION, "message_type": "initialize", "payload": payload}

    @staticmethod
    def request_message(sequence: int, snapshot: Mapping, elapsed: float, remaining: float) -> dict:
        payload = {"schema_version": SNAPSHOT_SCHEMA, **snapshot,
                   "wallclock": {"elapsed_seconds": round(elapsed, 3), "remaining_seconds": round(max(0.0, remaining), 3)}}
        return {"protocol_version": PROTOCOL_VERSION, "message_type": "decision_request",
                "decision_sequence": sequence, "payload": payload}

    @staticmethod
    def action_from_response(response, sequence: int) -> dict:
        if not isinstance(response, Mapping):
            raise ProtocolViolation("response must be a JSON object")
        if (response.get("protocol_version") != PROTOCOL_VERSION
                or response.get("message_type") != "decision_response"
                or type(response.get("decision_sequence")) is not int
                or response["decision_sequence"] != sequence):
            raise ProtocolViolation("response does not answer the current decision_request "
                                    f"({PROTOCOL_VERSION}, decision_response, decision_sequence {sequence})")
        for key in ("reason", "decision_source"):
            if key in response and not isinstance(response[key], str):
                raise ProtocolViolation(f"{key} must be a string")
        # The action fields sit next to the envelope fields, as in protocol v2.
        return {key: value for key, value in response.items() if key not in ENVELOPE_KEYS}

    @staticmethod
    def finish_payload(result: Mapping) -> dict:
        counts = result["score_report"]["counts"]
        return {"schema_version": FINISH_SCHEMA, "termination_reason": result["termination_reason"],
                "decisions": counts["decisions"], "observe_actions": counts["observe_actions"]}

    # --- run ---------------------------------------------------------------------------

    def run(
        self,
        decide: Callable[[dict, float], Mapping],
        output_dir: Path,
        *,
        wallclock_seconds: float | None = None,
        initialize: Callable[[dict], None] | None = None,
        deadline_cap: Callable[[], float | None] | None = None,
    ) -> dict:
        """Run the card; returns the workflow result (also written to workflow_result.json).

        ``decide`` raises TimeoutError at the deadline (the transport's GlobalDeadlineExpired)
        and any other exception for a broken agent. ``initialize`` failures propagate after
        the run is settled with no decisions, so the caller can fail the job like v3.
        ``deadline_cap`` optionally returns an external (session) deadline in the same clock.
        """
        budget = self.wallclock_budget(wallclock_seconds)
        state: dict = {"sequence": 0, "started": None, "deadline": None, "ignored_in_flight": False,
                       "initialization_error": None, "agent_seconds": 0.0}
        actions: list[str] = []  # every action the runner received, for organizer replay

        def deadline() -> float:
            cap = deadline_cap() if deadline_cap is not None else None
            return state["deadline"] if cap is None else min(state["deadline"], cap)

        def factory(_context):
            if initialize is not None:
                try:
                    initialize(self.initialize_payload(budget))
                except Exception as error:  # noqa: BLE001 - reported to the caller below
                    state["initialization_error"] = error
                    raise AgentTermination(TERMINATION_AGENT_ERROR, "agent initialization failed") from None
            state["started"] = self.clock()
            state["deadline"] = state["started"] + budget

            def agent(snapshot):
                limit = deadline()
                now = self.clock()
                if now >= limit:
                    raise AgentTermination(TERMINATION_WALLCLOCK)
                state["sequence"] += 1
                sequence = state["sequence"]
                message = self.request_message(sequence, snapshot, now - state["started"], limit - now)
                try:
                    response = decide(message, limit)
                except TimeoutError:
                    state["ignored_in_flight"] = True
                    raise AgentTermination(TERMINATION_WALLCLOCK) from None
                except Exception as error:  # noqa: BLE001 - a broken agent, never the engine
                    raise AgentTermination(TERMINATION_AGENT_ERROR, _safe_detail(error)) from None
                finally:
                    state["agent_seconds"] += self.clock() - now
                if self.clock() >= limit:
                    state["ignored_in_flight"] = True
                    raise AgentTermination(TERMINATION_WALLCLOCK)
                try:
                    action = self.action_from_response(response, sequence)
                except ProtocolViolation as error:
                    raise AgentTermination(TERMINATION_AGENT_ERROR, str(error)) from None
                actions.append(json.dumps(action, ensure_ascii=False, sort_keys=True, allow_nan=False))
                return action

            return agent

        report = run_scenario(self.scenario_path, factory, Path(output_dir))
        report.pop("organizer_only", None)
        started = state["started"]
        elapsed = 0.0 if started is None else max(0.0, min(self.clock(), state["deadline"]) - started)
        termination = report["termination"]
        result = {
            "schema_version": RESULT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "scenario": report["scenario"],
            "termination_reason": termination["reason"],
            "termination_detail": termination["detail"],
            "global_wallclock_seconds": budget,
            "accounted_wallclock_seconds": round(elapsed, 3),
            "agent_wallclock_seconds": round(state["agent_seconds"], 3),
            "ignored_in_flight_response": state["ignored_in_flight"],
            "decision_requests": state["sequence"],
            "last_decision_sequence": state["sequence"],
            "committed_action_count": report["counts"]["decisions"],
            "score_report": report,
        }
        write_text_lf(Path(output_dir) / ACTIONS_FILE, "".join(line + "\n" for line in actions))
        write_text_lf(Path(output_dir) / "workflow_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        if state["initialization_error"] is not None:
            result["initialization_error"] = state["initialization_error"]
        return result


def replay(root: Path, actions_path: Path, output_dir: Path) -> dict:
    """Organizer verification: re-run a recorded ``actions.jsonl`` against the bundle.

    The runner is deterministic, so the replay reproduces the score report and
    decisions.csv byte for byte. A run that ended on the wall clock replays as
    ``agent_finished``; an invalid final action replays as the same ``agent_error``.
    """
    validate_bundle(root)
    lines = [line for line in Path(actions_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def factory(_context):
        iterator = iter(json.loads(line) for line in lines)
        return lambda _snapshot: next(iterator, None)

    report = run_scenario(Path(root) / V4_SCENARIO_PATH, factory, Path(output_dir))
    report.pop("organizer_only", None)
    return report


def result_summary(result: Mapping) -> dict:
    """Small database summary for a v4 run (``observer_finish_run`` reads score.total)."""
    report = result["score_report"]
    components = report["components"]
    counts = report["counts"]
    return {
        "schema_version": "observer-run-summary-v1",
        "gameplay": "v4",
        "score": {"total": report["total"], **{key: components[key] for key in sorted(components)}},
        "required_missing": counts["required_missing"],
        "targets_observed": counts["targets_observed"],
        "observe_actions": counts["observe_actions"],
        "invalidated_observations": counts["invalidated_observations"],
        "termination_reason": result["termination_reason"],
        "committed_action_count": result["committed_action_count"],
        "accounted_wallclock_seconds": result["accounted_wallclock_seconds"],
    }


def _utc(moment) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_detail(error: Exception) -> str:
    # Transport errors carry fixed participant-facing wording; anything else is reduced
    # to its type so project output or credentials never reach the result files.
    text = str(error) if type(error).__name__ in ("ExecutionError", "ProtocolViolation") else ""
    return (type(error).__name__ + (": " + text if text else ""))[:300]
