#!/usr/bin/env python3
"""v4 runner: continuous-time, replayable run loop over the v4 scenario products.

Drives a Python-callable agent through the survey: validate and execute observe/wait/
report actions, evaluate fiber hits under the actual (offset-applied) pointing with the
MP-056 geometry, apply weather and events per target with the MP-054 truth, serve the
latest published bulletin/forecast at each decision moment, trigger the data-loss
stress event at runtime (observe-action window invalidation -> best-score recompute ->
state_resync message, time not refunded), and write decisions.csv / observations.csv /
messages.jsonl / score_report.json.

Agent contract (protocol wiring belongs to the platform layer): a callable
`agent(snapshot) -> action dict | None`. The snapshot carries now_utc, survey_end_utc,
observe_action_index, running_total, latest_bulletin, latest_forecast, new_messages
(bulletins/forecasts/state_resync delivered since the previous decision) and
last_result (hit target ids + scores of the previous observe, without fiber ids — per
the MP-056 ruling). Actions: {"action": "observe", ...} (MP-056 schema plus a "program"
field), {"action": "wait", "duration_seconds": n}, {"action": "report"}, or None to end
the run early.

Pure standard library; deterministic for a fixed scenario and agent.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .contracts import sha256_file, write_exact_csv, write_text_lf
from .v4_config_check import (
    DEFAULT_FALSE_REPORT_FREE_ALLOWANCE,
    DEFAULT_MAX_CONSECUTIVE_REPORTS,
    validate_scenario,
)
from .v4_fiber_map import (
    FiberGrid,
    altaz_to_radec,
    min_altitude_during,
    radec_to_altaz,
    validate_action,
)
from .v4_scorer import (
    FORCE_CLOSE_TYPE,
    FAULT_TYPE,
    ZERO_SCORE_TYPE,
    BestLedger,
    ScoreEvent,
    SlotTruth,
    TargetScore,
    WeatherTruth,
    lunar_quality_factor,
    program_band,
    program_multiplier,
    required_penalty,
    score_target_exposure,
    uniformity_penalty,
    validate_lunar_model,
)


PROGRAMS = ("DARK", "BRIGHT", "BACKUP")
DEFAULT_PROGRAM = "BACKUP"

# Termination reasons recorded in score_report.json["termination"]["reason"].
TERMINATION_SURVEY_COMPLETE = "survey_complete"
TERMINATION_AGENT_FINISHED = "agent_finished"
TERMINATION_AGENT_ERROR = "agent_error"
TERMINATION_WALLCLOCK = "global_wallclock_expired"
TERMINATION_REASONS = (
    TERMINATION_SURVEY_COMPLETE,
    TERMINATION_AGENT_FINISHED,
    TERMINATION_AGENT_ERROR,
    TERMINATION_WALLCLOCK,
)


class InvalidAgentAction(ValueError):
    """The agent's action violates the action contract. The run ends as agent_error and the
    score is settled on the valid history (platform hardening; the prototype aborted)."""


class AgentTermination(Exception):
    """Raised by an agent callable (e.g. the platform transport adapter) to stop the run.

    ``reason`` is one of TERMINATION_REASONS; the run is settled on the valid history."""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in TERMINATION_REASONS:
            raise ValueError(f"unknown termination reason {reason!r}")
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


DECISION_COLUMNS = [
    "decision_id",
    "observe_index",
    "action",
    "start_utc",
    "end_utc",
    "duration_seconds",
    "alt_deg",
    "az_deg",
    "program",
    "assigned_count",
    "hit_count",
    "valid",
    "invalidated_by",
]

OBSERVATION_COLUMNS = [
    "observation_id",
    "observe_index",
    "target_id",
    "fiber_id",
    "factor",
    "quality",
    "prog_mult",
    "score",
    "valid",
    "invalidated_by",
]


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(base: Path, value: str) -> Path:
    return (base / value).resolve()


@dataclass(frozen=True)
class Scenario:
    config: Mapping
    scenario_dir: Path
    site: Mapping
    fiber_config: Mapping
    score_config: Mapping
    targets: list[dict]
    slots: list[SlotTruth]
    events: list[ScoreEvent]
    bulletins: list[dict]
    forecasts: list[dict]
    stress_rows: list[dict[str, str]]
    survey_start: datetime
    survey_end: datetime

    @property
    def minimum_altitude_deg(self) -> float:
        return float(self.config["minimum_altitude_deg"])

    @property
    def stress_enabled(self) -> bool:
        return bool(self.config.get("stress", {}).get("enabled", False))


def load_scenario(path: Path) -> Scenario:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "v4-scenario-v1":
        raise ValueError("unsupported scenario schema_version")
    base = path.parent
    fiber_config = json.loads(_resolve(base, config["fiber_config"]).read_text(encoding="utf-8"))
    score_config = json.loads(_resolve(base, config["score_config"]).read_text(encoding="utf-8"))
    score_config["reporting"].setdefault(
        "false_report_free_allowance", DEFAULT_FALSE_REPORT_FREE_ALLOWANCE
    )
    score_config["reporting"].setdefault(
        "max_consecutive_reports", DEFAULT_MAX_CONSECUTIVE_REPORTS
    )
    validate_lunar_model(score_config)
    products = config["products"]

    targets = []
    for row in _read_csv(_resolve(base, products["targets_csv"])):
        targets.append(
            {
                "target_id": row["target_id"],
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "target_class": row["target_class"],
                "feature_flux": float(row["feature_flux"]),
                "science_weight": float(row["science_weight"]),
                "required": row["required"].strip().lower() == "true",
            }
        )

    truth_rows = {row["slot_id"]: row for row in _read_csv(_resolve(base, products["weather_truth_csv"]))}
    slots: list[SlotTruth] = []
    for row in _read_csv(_resolve(base, products["slots_csv"])):
        truth = truth_rows[row["slot_id"]]
        observable = truth["is_observable"] == "true"
        start = _parse_utc(row["timestamp_utc"])
        slots.append(
            SlotTruth(
                row["slot_id"],
                start,
                start + timedelta(seconds=int(row["duration_seconds"])),
                observable,
                float(truth["seeing_arcsec"]) if observable else 0.0,
                float(truth["transparency"]) if observable else 0.0,
                float(truth["sky_quality"]) if observable else 0.0,
                float(truth["instrument_efficiency"]) if observable else 0.0,
            )
        )

    def _opt_float(row: Mapping[str, str], key: str) -> float | None:
        return None if row[key].strip() == "" else float(row[key])

    events = []
    for row in _read_csv(_resolve(base, products["events_csv"])):
        events.append(
            ScoreEvent(
                row["event_id"],
                row["event_type"],
                _parse_utc(row["actual_start_utc"]),
                _parse_utc(row["actual_end_utc"]),
                row["scope_type"],
                _opt_float(row, "azimuth_start_deg"),
                _opt_float(row, "azimuth_end_deg"),
                _opt_float(row, "min_altitude_deg"),
                _opt_float(row, "max_altitude_deg"),
                float(row["seeing_multiplier"]),
                float(row["transparency_multiplier"]),
                float(row["sky_quality_multiplier"]),
                float(row["instrument_efficiency_multiplier"]),
                row["force_close"].strip().lower() == "true",
            )
        )

    stress = config.get("stress", {})
    if not isinstance(stress.get("enabled", False), bool):
        raise ValueError("stress.enabled must be a boolean")
    stress_rows = (
        _read_csv(_resolve(base, stress["stress_events_csv"]))
        if stress.get("enabled")
        else []
    )
    scenario = Scenario(
        config=config,
        scenario_dir=base,
        site=config["site"],
        fiber_config=fiber_config,
        score_config=score_config,
        targets=targets,
        slots=slots,
        events=events,
        bulletins=_read_jsonl(_resolve(base, products["bulletins_jsonl"])),
        forecasts=_read_jsonl(_resolve(base, products["forecasts_jsonl"])),
        stress_rows=stress_rows,
        survey_start=slots[0].start_utc if slots else None,
        survey_end=slots[-1].end_utc if slots else None,
    )
    validate_scenario(scenario)
    return scenario


# --- data-loss runtime trigger ---------------------------------------------------------


@dataclass(frozen=True)
class DataLossPlan:
    event_id: str
    trigger_mode: str
    trigger_ref: str
    trigger_utc: datetime
    window_start_fraction: float
    window_end_fraction: float
    window_max_fraction: float


def _data_loss_plan(scenario: Scenario) -> DataLossPlan | None:
    if not scenario.stress_enabled:
        return None
    rows = [row for row in scenario.stress_rows if row["event_type"] == "data_loss"]
    if not rows:
        return None
    row = rows[0]
    mode = row["trigger_mode"]
    ref = row["trigger_ref"]
    if mode == "after_earthquake":
        quake = next(
            (event for event in scenario.events if event.event_id == ref), None
        )
        if quake is None:
            raise ValueError(f"data_loss trigger_ref {ref} is not a scenario event")
        trigger_utc = quake.actual_end_utc
    elif mode == "fixed_slot":
        slot = next((slot for slot in scenario.slots if slot.slot_id == ref), None)
        if slot is None:
            raise ValueError(f"data_loss fixed_slot {ref} is not a scenario slot")
        trigger_utc = slot.start_utc
    elif mode == "fixed_date":
        trigger_utc = _parse_utc(ref).astimezone(timezone.utc)
    else:
        raise ValueError(f"unknown data_loss trigger_mode {mode}")
    return DataLossPlan(
        row["event_id"],
        mode,
        ref,
        trigger_utc,
        float(row["window_start_fraction"]),
        float(row["window_end_fraction"]),
        float(row["window_max_fraction"]),
    )


def _pointing_offset(scenario: Scenario) -> tuple[float, float]:
    if not scenario.stress_enabled:
        return 0.0, 0.0
    rows = [row for row in scenario.stress_rows if row["event_type"] == "pointing_offset"]
    if not rows:
        return 0.0, 0.0
    return math.degrees(float(rows[0]["alt_offset_rad"])), math.degrees(
        float(rows[0]["az_offset_rad"])
    )


# --- run loop ---------------------------------------------------------------------------


def run_scenario(
    scenario_path: Path,
    agent_factory: Callable[[Mapping], Callable[[Mapping], Mapping | None]],
    output_dir: Path,
) -> dict[str, object]:
    scenario = load_scenario(scenario_path)
    grid = FiberGrid.from_config(scenario.fiber_config)
    score_config = scenario.score_config
    site = scenario.site
    lat = float(site["latitude_deg"])
    lon = float(site["longitude_deg"])
    min_alt = scenario.minimum_altitude_deg
    offset_alt, offset_az = _pointing_offset(scenario)
    loss_plan = _data_loss_plan(scenario)

    faults = [event for event in scenario.events if event.event_type == FAULT_TYPE]
    weather = WeatherTruth(scenario.slots, faults)
    slot_starts = [slot.start_utc for slot in scenario.slots]
    night_ends = {
        slot.slot_id.split("-S", 1)[0]: slot.end_utc for slot in scenario.slots
    }
    directional_events = [
        event for event in scenario.events if event.scope_type == "HORIZON_SECTOR"
    ]
    ledger = BestLedger()

    # Precompute unit vectors for the field-of-view candidate filter.
    targets_by_id: dict[str, dict] = {}
    target_vectors: dict[str, tuple[float, float, float]] = {}
    for target in scenario.targets:
        targets_by_id[target["target_id"]] = target
        ra = math.radians(target["ra_deg"])
        dec = math.radians(target["dec_deg"])
        target_vectors[target["target_id"]] = (
            math.cos(dec) * math.cos(ra),
            math.cos(dec) * math.sin(ra),
            math.sin(dec),
        )
    fov_radius_deg = math.degrees(
        math.atan(math.sqrt(2.0) * math.radians(grid.fov_side_deg / 2.0))
    ) + 0.3
    cos_fov_radius = math.cos(math.radians(fov_radius_deg))

    bulletin_times = [_parse_utc(item["issued_at_utc"]) for item in scenario.bulletins]
    forecast_times = [_parse_utc(item["issued_at_utc"]) for item in scenario.forecasts]
    bulletin_cursor = 0
    forecast_cursor = 0

    context = {
        "scenario": {
            "name": scenario.config["name"],
            "minimum_altitude_deg": scenario.minimum_altitude_deg,
        },
        "site": dict(scenario.site),
        "fiber": dict(scenario.fiber_config["field"]),
        "score": dict(score_config),
        "targets": list(scenario.targets),
        "footprint": _read_csv(
            _resolve(scenario.scenario_dir, scenario.config["products"]["footprint_csv"])
        ),
    }
    context.update(scenario.config.get("agent_params", {}))
    decisions: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    messages: list[dict] = []
    report_settlement = 0.0
    invalidation_notes: list[dict[str, object]] = []

    now = scenario.survey_start
    observe_index = 0
    decision_id = 0
    last_result: dict | None = None
    pending_report_result: dict | None = None
    loss_triggered = False
    consecutive_reports = 0
    max_consecutive_reports = score_config["reporting"].get(
        "max_consecutive_reports", DEFAULT_MAX_CONSECUTIVE_REPORTS
    )
    termination_reason = TERMINATION_SURVEY_COMPLETE
    termination_detail = ""
    pending_until: datetime | None = None  # active wait-until expansion
    carried_messages: list[dict] = []  # published during a wait-until expansion
    max_wait_seconds = int(scenario.fiber_config["exposure"]["max_duration_seconds"])

    try:
        agent = agent_factory(context)
    except AgentTermination as stop:
        agent = None
        termination_reason, termination_detail = stop.reason, stop.detail

    while agent is not None and now < scenario.survey_end:
        # Deliver newly published bulletins/forecasts, report results, and state resyncs.
        new_messages: list[dict] = []
        while bulletin_cursor < len(scenario.bulletins) and bulletin_times[bulletin_cursor] <= now:
            new_messages.append(scenario.bulletins[bulletin_cursor])
            messages.append(scenario.bulletins[bulletin_cursor])
            bulletin_cursor += 1
        while forecast_cursor < len(scenario.forecasts) and forecast_times[forecast_cursor] <= now:
            new_messages.append(scenario.forecasts[forecast_cursor])
            messages.append(scenario.forecasts[forecast_cursor])
            forecast_cursor += 1
        if pending_report_result is not None:
            new_messages.append(pending_report_result)
            messages.append(pending_report_result)
            pending_report_result = None
        # Contract: process a data-loss trigger at the first decision after its
        # event time. An exposure already in progress completes before N is frozen.
        if loss_plan is not None and not loss_triggered and now >= loss_plan.trigger_utc:
            loss_triggered = True
            total_observe = observe_index
            first = math.floor(loss_plan.window_start_fraction * total_observe)
            last = math.floor(loss_plan.window_end_fraction * total_observe)
            invalidated = ledger.invalidate_window(first, last, loss_plan.event_id)
            for row in decisions:
                if (
                    row["action"] == "observe"
                    and row["valid"] == "true"
                    and first <= int(row["observe_index"]) < last
                ):
                    row["valid"] = "false"
                    row["invalidated_by"] = loss_plan.event_id
            for row in observations:
                if row["valid"] == "true" and first <= int(row["observe_index"]) < last:
                    row["valid"] = "false"
                    row["invalidated_by"] = loss_plan.event_id
            best = ledger.best()
            resync = {
                "record_type": "state_resync",
                "issued_at_utc": _format_utc(now),
                "trigger_event_id": loss_plan.event_id,
                "invalidated_window": {
                    "action_count_at_trigger": total_observe,
                    "action_index_start": first,
                    "action_index_end_exclusive": last,
                    "window_start_fraction": loss_plan.window_start_fraction,
                    "window_end_fraction": loss_plan.window_end_fraction,
                    "window_max_fraction": loss_plan.window_max_fraction,
                },
                "observed_target_ids": sorted(best),
                "best_scores": [
                    {"target_id": target_id, "best_score": round(best[target_id][1], 6)}
                    for target_id in sorted(best)
                ],
            }
            invalidation_notes.append(
                {
                    "event_id": loss_plan.event_id,
                    "invalidated_observations": invalidated,
                    "window": [first, last],
                }
            )
            new_messages.append(resync)
            messages.append(resync)

        if pending_until is not None:
            if now < pending_until:
                # wait-until expansion: successive <= max-wait runner waits with no agent
                # round trip; everything published meanwhile is delivered afterwards.
                carried_messages.extend(new_messages)
                end = min(now + timedelta(seconds=max_wait_seconds), pending_until, scenario.survey_end)
                decision_id += 1
                decisions.append(_wait_row(decision_id, now, end))
                last_result = {"action": "wait"}
                now = end
                continue
            pending_until = None
        if carried_messages:
            new_messages = carried_messages + new_messages
            carried_messages = []

        latest_bulletin = scenario.bulletins[bulletin_cursor - 1] if bulletin_cursor else None
        latest_forecast = scenario.forecasts[forecast_cursor - 1] if forecast_cursor else None
        snapshot = {
            "now_utc": _format_utc(now),
            "survey_end_utc": _format_utc(scenario.survey_end),
            "observe_action_index": observe_index,
            "running_total": round(sum(item[1] for item in ledger.best().values()), 6),
            "latest_bulletin": latest_bulletin,
            "latest_forecast": latest_forecast,
            "new_messages": new_messages,
            "last_result": last_result,
        }
        try:
            raw_action = agent(snapshot)
        except AgentTermination as stop:
            termination_reason, termination_detail = stop.reason, stop.detail
            break
        if raw_action is None:
            termination_reason = TERMINATION_AGENT_FINISHED
            break
        try:
            action = normalize_action(raw_action, scenario, targets_by_id, now)
            if action["action"] == "report" and consecutive_reports >= max_consecutive_reports:
                raise InvalidAgentAction("too many consecutive report actions")
        except InvalidAgentAction as exc:
            termination_reason = TERMINATION_AGENT_ERROR
            termination_detail = str(exc)[:500]
            break
        kind = action["action"]

        if kind == "finish":
            termination_reason = TERMINATION_AGENT_FINISHED
            break

        if kind == "wait":
            consecutive_reports = 0
            if "until_utc" in action:
                pending_until = action["until_utc"]
                continue
            decision_id += 1
            end = min(now + timedelta(seconds=action["duration_seconds"]), scenario.survey_end)
            decisions.append(_wait_row(decision_id, now, end))
            last_result = {"action": "wait"}
            now = end
            continue

        decision_id += 1
        if kind == "report":
            consecutive_reports += 1
            active_fault = weather.active_fault(now)
            repaired = active_fault is not None and active_fault.event_id not in weather.fault_repairs
            score_delta = weather.report_fault(now, score_config)
            report_settlement += score_delta
            pending_report_result = {
                "record_type": "report_result",
                "issued_at_utc": _format_utc(now),
                "correct": repaired,
                "repaired": repaired,
                "score_delta": score_delta,
            }
            decisions.append(
                {
                    "decision_id": decision_id,
                    "observe_index": "",
                    "action": "report",
                    "start_utc": _format_utc(now),
                    "end_utc": _format_utc(now),
                    "duration_seconds": 0,
                    "alt_deg": "",
                    "az_deg": "",
                    "program": "",
                    "assigned_count": 0,
                    "hit_count": 0,
                    "valid": "true",
                    "invalidated_by": "",
                }
            )
            last_result = {
                "action": "report",
                "correct": repaired,
                "repaired": repaired,
                "score_delta": score_delta,
            }
            continue

        consecutive_reports = 0
        program = action["program"]
        cmd_alt = float(action["pointing"]["alt_deg"])
        cmd_az = float(action["pointing"]["az_deg"])
        duration = int(action["duration_seconds"])
        start = now
        end = min(start + timedelta(seconds=duration), scenario.survey_end)
        slot_index = bisect_right(slot_starts, start) - 1
        if slot_index >= 0 and start < scenario.slots[slot_index].end_utc:
            # An exposure may cross weather slots, but stops at the end of its night.
            night_id = scenario.slots[slot_index].slot_id.split("-S", 1)[0]
            end = min(end, night_ends[night_id])
        duration = int((end - start).total_seconds())
        # Hit classification happens ONCE at exposure start (2026-09-28 ruling): the
        # actual center is converted to RA/Dec at start and tracked sidereally, so the
        # target stays fixed relative to its fiber for the whole exposure. Airmass and
        # directional events follow the target's position through the exposure.
        actual_alt, actual_az = grid.actual_center(cmd_alt, cmd_az, offset_alt, offset_az)
        center_ra, center_dec = altaz_to_radec(actual_alt, actual_az, start, lat, lon)
        center_in_range = 0.0 <= actual_alt <= 90.0
        ra_r = math.radians(center_ra)
        dec_r = math.radians(center_dec)
        center_vec = (math.cos(dec_r) * math.cos(ra_r), math.cos(dec_r) * math.sin(ra_r), math.sin(dec_r))

        # Candidate targets inside the field, via the unit-vector dot filter.
        on_glass: dict[str, int] = {}
        candidate_targets = scenario.targets if center_in_range else ()
        for target in candidate_targets:
            vx, vy, vz = target_vectors[target["target_id"]]
            if vx * center_vec[0] + vy * center_vec[1] + vz * center_vec[2] < cos_fov_radius:
                continue
            if min_altitude_during(
                target["ra_deg"], target["dec_deg"], start, end, scenario.config
            ) < min_alt:
                continue
            result = grid.classify_target(
                target["ra_deg"], target["dec_deg"], start, cmd_alt, cmd_az,
                lat, lon, offset_alt, offset_az,
            )
            if result.region == "glass":
                on_glass[target["target_id"]] = int(result.fiber_id)

        assignments = action["assignments"]
        segments = list(weather.segments(start, end))
        active_directional = [
            event for event in directional_events if event.overlaps(start, end)
        ]
        hit_rows: list[tuple[dict, int, TargetScore, float]] = []
        for fiber_id, target_id in assignments.items():
            if on_glass.get(target_id) != fiber_id:
                continue  # assigned but not on this fiber's glass: no score, no penalty
            target = targets_by_id[target_id]

            def target_altaz_at(moment: datetime, target=target) -> tuple[float, float]:
                return radec_to_altaz(
                    target["ra_deg"], target["dec_deg"], moment, lat, lon
                )

            q_band = weather.band_quality(
                start,
                end,
                score_config,
                lambda moment: target_altaz_at(moment)[0],
                lambda moment, target=target: lunar_quality_factor(
                    target["ra_deg"], target["dec_deg"], moment, site, score_config
                ),
            )
            mult = program_multiplier(program, program_band(q_band, score_config), score_config)
            scored = score_target_exposure(
                target,
                segments,
                active_directional,
                weather,
                duration,
                mult,
                score_config,
                start,
                target_altaz_at,
                site,
            )
            hit_rows.append((target, fiber_id, scored, mult))
            ledger.record(observe_index, target_id, scored.factor, scored.score)

        for target, fiber_id, scored, mult in hit_rows:
            observations.append(
                {
                    "observation_id": len(observations) + 1,
                    "observe_index": observe_index,
                    "target_id": target["target_id"],
                    "fiber_id": fiber_id,
                    "factor": f"{scored.factor:.6f}",
                    "quality": f"{scored.quality:.6f}",
                    "prog_mult": f"{mult:.6f}",
                    "score": f"{scored.score:.6f}",
                    "valid": "true",
                    "invalidated_by": "",
                }
            )
        decisions.append(
            {
                "decision_id": decision_id,
                "observe_index": observe_index,
                "action": "observe",
                "start_utc": _format_utc(start),
                "end_utc": _format_utc(end),
                "duration_seconds": duration,
                "alt_deg": f"{cmd_alt:.4f}",
                "az_deg": f"{cmd_az:.4f}",
                "program": program,
                "assigned_count": len(assignments),
                "hit_count": len(hit_rows),
                "valid": "true",
                "invalidated_by": "",
            }
        )
        last_result = {
            "action": "observe",
            "observe_index": observe_index,
            "assigned_count": len(assignments),
            "hit_count": len(hit_rows),
            "hits": [
                {"target_id": target["target_id"], "score": round(scored.score, 6)}
                for target, _, scored, _ in hit_rows
            ],
        }
        observe_index += 1
        now = end

    # --- settlement -------------------------------------------------------------------
    best = ledger.best()
    max_factors = ledger.max_factors()
    sum_best = sum(item[1] for item in best.values())
    required_missing, required_cost = required_penalty(scenario.targets, max_factors, score_config)
    band_ratios, uniformity_cost = uniformity_penalty(scenario.targets, max_factors, score_config)
    total = sum_best - required_cost - uniformity_cost + report_settlement

    by_class: dict[str, float] = {}
    class_by_id = {target["target_id"]: target["target_class"] for target in scenario.targets}
    for target_id, (_, score) in best.items():
        by_class[class_by_id[target_id]] = by_class.get(class_by_id[target_id], 0.0) + score

    output_dir.mkdir(parents=True, exist_ok=True)
    write_exact_csv(output_dir / "decisions.csv", DECISION_COLUMNS, decisions)
    write_exact_csv(output_dir / "observations.csv", OBSERVATION_COLUMNS, observations)
    write_text_lf(
        output_dir / "messages.jsonl",
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in messages) + "\n",
    )
    report = {
        "schema_version": "v4-score-report-v1",
        "scenario": str(scenario.config["name"]),
        "total": round(total, 6),
        "components": {
            "sum_best_scores": round(sum_best, 6),
            "required_penalty": round(-required_cost, 6),
            "uniformity_penalty": round(-uniformity_cost, 6),
            "report_settlement": round(report_settlement, 6),
        },
        "counts": {
            "decisions": len(decisions),
            "observe_actions": observe_index,
            "observations": len(observations),
            "targets_observed": len(best),
            "required_missing": required_missing,
            "invalidated_observations": sum(
                note["invalidated_observations"] for note in invalidation_notes
            ),
        },
        "by_class": {key: round(value, 6) for key, value in sorted(by_class.items())},
        "uniformity_bands": {key: round(value, 6) for key, value in band_ratios.items()},
        "invalidations": invalidation_notes,
        "termination": {"reason": termination_reason, "detail": termination_detail},
        "sha256": {
            "scenario": sha256_file(scenario_path),
            "decisions": sha256_file(output_dir / "decisions.csv"),
            "observations": sha256_file(output_dir / "observations.csv"),
            "messages": sha256_file(output_dir / "messages.jsonl"),
        },
    }
    write_text_lf(
        output_dir / "score_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    # Hidden stress truth stays out of every written (participant-visible) artifact; the
    # caller gets it in memory for organizer diagnostics only.
    return dict(report, organizer_only={"pointing_offset_deg": {"alt": offset_alt, "az": offset_az}})


def _wait_row(decision_id: int, start: datetime, end: datetime) -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "observe_index": "",
        "action": "wait",
        "start_utc": _format_utc(start),
        "end_utc": _format_utc(end),
        "duration_seconds": int((end - start).total_seconds()),
        "alt_deg": "",
        "az_deg": "",
        "program": "",
        "assigned_count": 0,
        "hit_count": 0,
        "valid": "true",
        "invalidated_by": "",
    }


# --- action contract ------------------------------------------------------------------

OBSERVE_KEYS = {"action", "pointing", "assignments", "duration_seconds", "program"}
OBSERVE_REQUIRED = {"action", "pointing", "assignments", "duration_seconds"}


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidAgentAction(f"{label} must be a finite number")
    return float(value)


def _integer(value, label: str) -> int:
    number = _number(value, label)
    if number != int(number):
        raise InvalidAgentAction(f"{label} must be a whole number of seconds")
    return int(number)


def normalize_action(action, scenario: Scenario, targets_by_id: Mapping[str, Mapping], now: datetime) -> dict:
    """Validate one agent action and return its normalized form; raises InvalidAgentAction.

    Accepted forms (participant-agent-protocol-v4, section 2.3):
      {"action": "observe", "pointing": {"alt_deg", "az_deg"}, "assignments": {fiber: target_id},
       "duration_seconds": int, "program": "DARK"|"BRIGHT"|"BACKUP" (optional, default BACKUP)}
      {"action": "wait", "duration_seconds": int}  or  {"action": "wait", "until_utc": "...Z"}
      {"action": "report"}
      {"action": "finish"}
    """
    if not isinstance(action, Mapping):
        raise InvalidAgentAction("action must be a JSON object")
    kind = action.get("action")
    exposure = scenario.fiber_config["exposure"]
    low = int(exposure["min_duration_seconds"])
    high = int(exposure["max_duration_seconds"])
    if kind in ("report", "finish"):
        if set(action) != {"action"}:
            raise InvalidAgentAction(f"{kind} takes no fields")
        return {"action": kind}
    if kind == "wait":
        keys = set(action) - {"action"}
        if keys == {"duration_seconds"}:
            duration = _integer(action["duration_seconds"], "wait duration_seconds")
            if not low <= duration <= high:
                raise InvalidAgentAction(f"wait duration_seconds must lie in [{low}, {high}]")
            return {"action": "wait", "duration_seconds": duration}
        if keys == {"until_utc"}:
            value = action["until_utc"]
            if not isinstance(value, str) or not (value.endswith("Z") or value.endswith("+00:00")):
                raise InvalidAgentAction("wait until_utc must be an ISO-8601 UTC timestamp ending in Z")
            try:
                until = _parse_utc(value)
            except ValueError as exc:
                raise InvalidAgentAction("wait until_utc is not a valid timestamp") from exc
            if until <= now:
                raise InvalidAgentAction("wait until_utc must be later than now_utc")
            return {"action": "wait", "until_utc": until}
        raise InvalidAgentAction("wait needs exactly one of duration_seconds or until_utc")
    if kind != "observe":
        raise InvalidAgentAction(f"unknown action {kind!r}")
    if not OBSERVE_REQUIRED <= set(action) or not set(action) <= OBSERVE_KEYS:
        raise InvalidAgentAction(
            "observe needs pointing, assignments, duration_seconds and optional program, nothing else"
        )
    program = action.get("program", DEFAULT_PROGRAM)
    if program not in PROGRAMS:
        raise InvalidAgentAction("program must be DARK, BRIGHT or BACKUP")
    pointing = action["pointing"]
    if not isinstance(pointing, Mapping) or set(pointing) != {"alt_deg", "az_deg"}:
        raise InvalidAgentAction("pointing must contain exactly alt_deg and az_deg")
    alt = _number(pointing["alt_deg"], "pointing alt_deg")
    az = _number(pointing["az_deg"], "pointing az_deg")
    duration = _integer(action["duration_seconds"], "observe duration_seconds")
    assignments_in = action["assignments"]
    if not isinstance(assignments_in, Mapping):
        raise InvalidAgentAction("assignments must map fiber_id to target_id")
    n_fibers = int(scenario.fiber_config["field"]["n_fibers"])
    assignments: dict[int, str] = {}
    for key, target_id in assignments_in.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)) or not str(key).strip().isdigit():
            raise InvalidAgentAction(f"invalid fiber_id {key!r}")
        fiber_id = int(key)
        if not 0 <= fiber_id < n_fibers:
            raise InvalidAgentAction(f"fiber_id {fiber_id} outside 0..{n_fibers - 1}")
        if fiber_id in assignments:
            raise InvalidAgentAction(f"fiber_id {fiber_id} assigned twice")
        if not isinstance(target_id, str) or target_id not in targets_by_id:
            raise InvalidAgentAction(f"unknown target_id {str(target_id)[:64]!r}")
        assignments[fiber_id] = target_id
    try:
        validate_action(
            {"action": "observe", "pointing": {"alt_deg": alt, "az_deg": az},
             "assignments": assignments, "duration_seconds": duration},
            scenario.fiber_config,
        )
    except ValueError as exc:
        raise InvalidAgentAction(str(exc)) from exc
    return {
        "action": "observe",
        "pointing": {"alt_deg": alt, "az_deg": az},
        "assignments": assignments,
        "duration_seconds": duration,
        "program": program,
    }


def _load_agent(spec: str):
    module_name, _, factory_name = spec.partition(":")
    if not factory_name:
        raise ValueError("agent spec must be module:factory")
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--agent", required=True, help="module:factory, e.g. challenge.v4_probe_agent:make_agent")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_scenario(args.scenario, _load_agent(args.agent), args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
