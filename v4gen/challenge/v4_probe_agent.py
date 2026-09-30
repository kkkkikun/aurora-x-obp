#!/usr/bin/env python3
"""Scripted probe-and-greedy verification agent for the v4 runner (not a contract).

Strategy (deliberately simple, deterministic): two short BACKUP probes first; then, at
each decision, point at the best currently-up unobserved target (required first, then
science weight, then altitude) and greedily assign the nearest unobserved neighbors to
the fibers they fall in under the *commanded* pointing (the agent does not know the
pointing offset). The pointing is nudged by a small deterministic pattern so the anchor
target itself lands on glass rather than a frame corner. Program choice: DARK when the
latest bulletin carries no active weather notices, BACKUP otherwise; wait when nothing
worth observing is up. Pure standard library.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping

from .v4_fiber_map import FiberGrid, radec_to_altaz


WEATHER_KINDS = {"rain", "overcast", "haze", "cold_snap", "storm"}


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _unit(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))


def make_agent(context: Mapping):
    """Factory for the runner's `--agent challenge.v4_probe_agent:make_agent` form."""
    max_observes = int(context.get("max_observes", 240))
    site = context["site"]
    lat = float(site["latitude_deg"])
    lon = float(site["longitude_deg"])
    grid = FiberGrid.from_config({"field": context["fiber"]})
    targets = list(context["targets"])
    min_alt = float(context["scenario"]["minimum_altitude_deg"])
    half_fov = grid.fov_side_deg / 2.0
    observed: set[str] = set()
    state = {"observes": 0}
    # Deterministic nudge pattern to keep the anchor target off the frame.
    nudges = [
        (0.0, 0.0),
        (grid.pitch_deg / 2.0, grid.pitch_deg / 2.0),
        (-grid.pitch_deg / 2.0, grid.pitch_deg / 2.0),
        (grid.pitch_deg / 2.0, -grid.pitch_deg / 2.0),
        (-grid.pitch_deg / 2.0, -grid.pitch_deg / 2.0),
    ]

    def choose_program(snapshot) -> str:
        bulletin = snapshot.get("latest_bulletin") or {}
        if any(notice["event_kind"] in WEATHER_KINDS for notice in bulletin.get("notices", [])):
            return "BACKUP"
        return "DARK"

    def agent(snapshot: Mapping):
        now = _parse_utc(snapshot["now_utc"])
        if state["observes"] >= max_observes:
            return None
        result = snapshot.get("last_result") or {}
        if result.get("action") == "observe":
            for hit in result.get("hits", []):
                observed.add(hit["target_id"])
        remaining = [t for t in targets if t["target_id"] not in observed]
        if not remaining:
            return None

        # Anchor: best unobserved target currently above the altitude limit with margin.
        # Cheap hour-angle pre-filter before the exact alt/az computation.
        from .tile_geometry_simulator import _local_sidereal_deg

        lst = _local_sidereal_deg(now, lon)
        ra_span = 90.0 + (90.0 - (min_alt + 2.0))  # generous band around the meridian

        def altaz(target):
            return radec_to_altaz(target["ra_deg"], target["dec_deg"], now, lat, lon)

        up = []
        for target in remaining:
            if abs((target["ra_deg"] - lst + 180.0) % 360.0 - 180.0) > ra_span:
                continue
            t_alt, t_az = altaz(target)
            if t_alt >= min_alt + 2.0:
                up.append((target, t_alt, t_az))
        if not up:
            return {"action": "wait", "duration_seconds": 900}
        up.sort(
            key=lambda item: (
                not item[0]["required"],
                -float(item[0]["science_weight"]),
                -item[1],
                item[0]["target_id"],
            )
        )
        anchor, anchor_alt, anchor_az = up[0]

        # Nudge the pointing until the anchor lands on glass; remember its fiber.
        cmd_alt = cmd_az = None
        anchor_fiber = None
        for d_alt_nudge, d_az_nudge in nudges:
            candidate_alt = anchor_alt + d_alt_nudge
            candidate_az = anchor_az + d_az_nudge / max(1e-6, math.cos(math.radians(anchor_alt)))
            if not 0.0 <= candidate_alt <= 90.0:
                continue
            candidate_alt = round(candidate_alt, 4)
            candidate_az = round(candidate_az % 360.0, 4) % 360.0
            placement = grid.classify_target(
                anchor["ra_deg"], anchor["dec_deg"], now, candidate_alt, candidate_az, lat, lon
            )
            if placement.region == "glass":
                cmd_alt, cmd_az = candidate_alt, candidate_az
                anchor_fiber = placement.fiber_id
                break
        if cmd_alt is None:
            cmd_alt, cmd_az = anchor_alt, anchor_az
        cmd_alt, cmd_az = round(cmd_alt, 4), round(cmd_az, 4)

        assignments: dict[int, str] = {}
        if anchor_fiber is not None:
            assignments[anchor_fiber] = anchor["target_id"]
        ax, ay, az_ = _unit(anchor["ra_deg"], anchor["dec_deg"])

        def closeness(target):
            ux, uy, uz = _unit(target["ra_deg"], target["dec_deg"])
            return -(ux * ax + uy * ay + uz * az_)

        neighbors = sorted(remaining, key=lambda t: (closeness(t), t["target_id"]))
        for target in neighbors:
            if len(assignments) >= grid.n_fibers:
                break
            if target["target_id"] == anchor["target_id"]:
                continue
            t_alt, t_az = altaz(target)
            offsets = grid.tangent_offsets(t_alt, t_az, cmd_alt, cmd_az)
            if offsets is None:
                continue
            d_alt, d_az = offsets
            if abs(d_alt) > half_fov or abs(d_az) > half_fov:
                continue
            fiber_id, region = grid.classify_offset(d_alt, d_az)
            if region != "glass" or fiber_id in assignments:
                continue
            assignments[fiber_id] = target["target_id"]
        state["observes"] += 1
        probe = state["observes"] <= 2
        return {
            "action": "observe",
            "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
            "assignments": assignments,
            "duration_seconds": 60 if probe else 900,
            "program": "BACKUP" if probe else choose_program(snapshot),
        }

    return agent
