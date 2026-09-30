#!/usr/bin/env python3
"""v4 fiber map: square fiber-grid field geometry and the observe-action contract.

Pointing model (2026-09-28 ruling): the agent commands the field center in (alt, az);
the actual center is the command plus the pointing offset (delta_alt, delta_az) from the
stress truth (zero by default). At exposure start the actual center is converted to
RA/Dec and the telescope tracks sidereally, so a target stays fixed relative to its
fiber for the whole exposure and the hit test runs once, at exposure start.

Grid: `n_fibers` (a perfect square) square assignable regions of `fiber_area_deg2`
each. The official 16-fiber configuration sets `gap_deg=0`, so the regions tile a
6.4 deg2 field without blind strips. `gap_deg` remains in the public config for
protocol compatibility; it does not represent optical resolution or physical fiber
clearance. The grid is aligned with the alt/az axes at exposure start (rows along
alt, columns along az; it does not rotate with parallactic angle). Fiber ids are
row-major with fiber 0 at the lower-left corner (smaller local altitude and
azimuth offsets). Target positions use a gnomonic projection centered on the actual pointing;
the actual azimuth fixes the plane orientation when pointing at the zenith.

Hit semantics (2026-09-27 ruling): only a target that is BOTH assigned to a fiber AND
actually lands on that fiber's assignable region counts as a hit. Shared boundaries
belong to one cell deterministically. Targets on a different fiber or outside the
field are not hits — the geometry layer reports booleans only; scoring
and penalties belong to the scorer.

Pure standard library; the demo plot at the bottom uses matplotlib as a build-time-only
dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .tile_geometry_simulator import _local_sidereal_deg


SCHEMA_VERSION = "v4-fiber-map-v1"
SIDEREAL_DEGREES_PER_DAY = 360.98564736629  # Same rate as _local_sidereal_deg.

REGION_GLASS = "glass"
REGION_FRAME = "frame"
REGION_OUTSIDE = "outside"

# Observe-action contract for the v4 runner (MP-056). The geometry library validates
# and evaluates; the runner/scorer owns sequencing and scoring.
ACTION_SCHEMA = {
    "schema_version": "v4-action-v1",
    "action": "observe",
    "fields": {
        "action": "the string \"observe\"",
        "pointing": {
            "alt_deg": "commanded field-center altitude in [0, 90] deg",
            "az_deg": "commanded field-center azimuth in [0, 360) deg (0=north, 90=east)",
        },
        "assignments": (
            "object mapping fiber_id (string or int, 0..n_fibers-1) to target_id; "
            "each target may be assigned to at most one fiber"
        ),
        "duration_seconds": "exposure length within the configured [min, max] bounds",
    },
    "semantics": (
        "The actual field center is pointing + (delta_alt, delta_az) from the stress "
        "truth. A target scores only when it is assigned to a fiber and lands on that "
        "fiber's assignable region at exposure start. No fiber-id information "
        "is returned to the agent per ruling."
    ),
}


@dataclass(frozen=True)
class Classification:
    fiber_id: int | None
    region: str  # "glass" | "frame" | "outside"
    alt_deg: float
    az_deg: float


def _site(config: Mapping) -> tuple[float, float]:
    site = config["site"]
    return float(site["latitude_deg"]), float(site["longitude_deg"])


def radec_to_altaz(
    ra_deg: float, dec_deg: float, moment: datetime, latitude_deg: float, longitude_deg: float
) -> tuple[float, float]:
    """Equatorial -> horizontal (az 0=north, 90=east), same convention as the v3 geometry."""
    hour_angle = math.radians(
        (_local_sidereal_deg(moment, longitude_deg) - ra_deg + 180.0) % 360.0 - 180.0
    )
    latitude = math.radians(latitude_deg)
    declination = math.radians(dec_deg)
    sin_altitude = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    altitude = math.asin(max(-1.0, min(1.0, sin_altitude)))
    cos_altitude = max(1e-12, math.cos(altitude))
    sin_azimuth = -math.sin(hour_angle) * math.cos(declination) / cos_altitude
    cos_azimuth = (
        math.sin(declination) - math.sin(altitude) * math.sin(latitude)
    ) / (cos_altitude * max(1e-12, math.cos(latitude)))
    return math.degrees(altitude), math.degrees(math.atan2(sin_azimuth, cos_azimuth)) % 360.0


def altaz_to_radec(
    alt_deg: float, az_deg: float, moment: datetime, latitude_deg: float, longitude_deg: float
) -> tuple[float, float]:
    """Horizontal -> equatorial; exact inverse of radec_to_altaz."""
    altitude = math.radians(alt_deg)
    azimuth = math.radians(az_deg)
    latitude = math.radians(latitude_deg)
    sin_declination = (
        math.sin(altitude) * math.sin(latitude)
        + math.cos(altitude) * math.cos(latitude) * math.cos(azimuth)
    )
    declination = math.asin(max(-1.0, min(1.0, sin_declination)))
    cos_declination = max(1e-12, math.cos(declination))
    sin_hour = -math.sin(azimuth) * math.cos(altitude) / cos_declination
    cos_hour = (
        math.sin(altitude) - math.sin(declination) * math.sin(latitude)
    ) / (cos_declination * max(1e-12, math.cos(latitude)))
    hour_angle = math.degrees(math.atan2(sin_hour, cos_hour))
    ra = (_local_sidereal_deg(moment, longitude_deg) - hour_angle) % 360.0
    return ra, math.degrees(declination)


class FiberGrid:
    """Square fiber-grid field on a gnomonic plane centered at the actual pointing."""

    def __init__(self, fiber_area_deg2: float, gap_deg: float, n_fibers: int) -> None:
        n_side = math.isqrt(int(n_fibers))
        if int(n_fibers) < 1 or n_side * n_side != int(n_fibers):
            raise ValueError("n_fibers must be a positive perfect square")
        if fiber_area_deg2 <= 0.0 or gap_deg < 0.0:
            raise ValueError("fiber_area_deg2 must be positive and gap_deg non-negative")
        self.fiber_area_deg2 = float(fiber_area_deg2)
        self.gap_deg = float(gap_deg)
        self.n_fibers = int(n_fibers)
        self.n_side = n_side
        self.fiber_side_deg = math.sqrt(self.fiber_area_deg2)
        self.pitch_deg = self.fiber_side_deg + self.gap_deg
        self.fov_side_deg = self.n_side * self.pitch_deg
        self.glass_fill_fraction = (
            self.n_fibers * self.fiber_area_deg2 / self.fov_side_deg**2
        )

    @classmethod
    def from_config(cls, config: Mapping) -> "FiberGrid":
        field = config["field"]
        return cls(
            float(field["fiber_area_deg2"]),
            float(field["gap_deg"]),
            int(field["n_fibers"]),
        )

    def derived(self) -> dict[str, object]:
        return {
            "n_fibers": self.n_fibers,
            "n_side": self.n_side,
            "fiber_side_deg": round(self.fiber_side_deg, 6),
            "pitch_deg": round(self.pitch_deg, 6),
            "fov_side_deg": round(self.fov_side_deg, 6),
            "fov_area_deg2": round(self.fov_side_deg**2, 6),
            "glass_fill_fraction": round(self.glass_fill_fraction, 6),
        }

    def fiber_row_col(self, fiber_id: int) -> tuple[int, int]:
        if not 0 <= int(fiber_id) < self.n_fibers:
            raise ValueError(f"fiber_id {fiber_id} outside 0..{self.n_fibers - 1}")
        return int(fiber_id) // self.n_side, int(fiber_id) % self.n_side

    def fiber_center_offset(self, fiber_id: int) -> tuple[float, float]:
        """(increasing-alt, increasing-az) tangent-plane offsets, in degrees."""
        row, col = self.fiber_row_col(fiber_id)
        middle = (self.n_side - 1) / 2.0
        return (row - middle) * self.pitch_deg, (col - middle) * self.pitch_deg

    def classify_offset(self, d_alt_deg: float, d_az_scaled_deg: float) -> tuple[int | None, str]:
        """Grid-cell lookup in tangent-plane offsets; seams belong to one cell."""
        half = self.fov_side_deg / 2.0
        if abs(d_alt_deg) > half or abs(d_az_scaled_deg) > half:
            return None, REGION_OUTSIDE
        middle = self.n_side / 2.0
        row = min(max(int(math.floor(d_alt_deg / self.pitch_deg + middle)), 0), self.n_side - 1)
        col = min(max(int(math.floor(d_az_scaled_deg / self.pitch_deg + middle)), 0), self.n_side - 1)
        if self.gap_deg == 0.0:
            # Adjacent regions tile the whole field. Avoid a floating-point sliver
            # classified as frame at a shared boundary.
            return row * self.n_side + col, REGION_GLASS
        center_alt, center_az = self.fiber_center_offset(row * self.n_side + col)
        half_side = self.fiber_side_deg / 2.0
        if abs(d_alt_deg - center_alt) <= half_side and abs(d_az_scaled_deg - center_az) <= half_side:
            return row * self.n_side + col, REGION_GLASS
        return row * self.n_side + col, REGION_FRAME

    def actual_center(
        self, cmd_alt_deg: float, cmd_az_deg: float, offset_alt_deg: float = 0.0, offset_az_deg: float = 0.0
    ) -> tuple[float, float]:
        """Commanded pointing plus the stress-truth offset, applied in alt/az space."""
        return cmd_alt_deg + offset_alt_deg, (cmd_az_deg + offset_az_deg) % 360.0

    @staticmethod
    def tangent_offsets(
        target_alt_deg: float, target_az_deg: float,
        center_alt_deg: float, center_az_deg: float,
    ) -> tuple[float, float] | None:
        """(increasing-alt, increasing-az) gnomonic offsets; az orients zenith."""
        alt = math.radians(target_alt_deg)
        az = math.radians(target_az_deg)
        center_alt = math.radians(center_alt_deg)
        center_az = math.radians(center_az_deg)
        target = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
        center = (
            math.cos(center_alt) * math.cos(center_az),
            math.cos(center_alt) * math.sin(center_az),
            math.sin(center_alt),
        )
        up = (
            -math.sin(center_alt) * math.cos(center_az),
            -math.sin(center_alt) * math.sin(center_az),
            math.cos(center_alt),
        )
        right = (-math.sin(center_az), math.cos(center_az), 0.0)
        depth = sum(a * b for a, b in zip(target, center))
        if depth <= 0.0:
            return None
        return (
            math.degrees(sum(a * b for a, b in zip(target, up)) / depth),
            math.degrees(sum(a * b for a, b in zip(target, right)) / depth),
        )

    def classify_target(
        self,
        ra_deg: float,
        dec_deg: float,
        moment: datetime,
        cmd_alt_deg: float,
        cmd_az_deg: float,
        latitude_deg: float,
        longitude_deg: float,
        offset_alt_deg: float = 0.0,
        offset_az_deg: float = 0.0,
    ) -> Classification:
        """Where a sky target falls in the grid at exposure start (single instant)."""
        target_alt, target_az = radec_to_altaz(ra_deg, dec_deg, moment, latitude_deg, longitude_deg)
        actual_alt, actual_az = self.actual_center(cmd_alt_deg, cmd_az_deg, offset_alt_deg, offset_az_deg)
        if not 0.0 <= actual_alt <= 90.0:
            return Classification(None, REGION_OUTSIDE, target_alt, target_az)
        offsets = self.tangent_offsets(target_alt, target_az, actual_alt, actual_az)
        if offsets is None:
            return Classification(None, REGION_OUTSIDE, target_alt, target_az)
        d_alt, d_az_scaled = offsets
        fiber_id, region = self.classify_offset(d_alt, d_az_scaled)
        return Classification(fiber_id, region, target_alt, target_az)

    def assigned_hits(
        self,
        assignments: Mapping[int, str],
        targets: Mapping[str, tuple[float, float]],
        moment: datetime,
        cmd_alt_deg: float,
        cmd_az_deg: float,
        latitude_deg: float,
        longitude_deg: float,
        offset_alt_deg: float = 0.0,
        offset_az_deg: float = 0.0,
    ) -> dict[str, bool]:
        """Per assigned target: True iff it lands on its own fiber's glass at start."""
        hits: dict[str, bool] = {}
        for fiber_id, target_id in assignments.items():
            ra_deg, dec_deg = targets[target_id]
            result = self.classify_target(
                ra_deg, dec_deg, moment, cmd_alt_deg, cmd_az_deg,
                latitude_deg, longitude_deg, offset_alt_deg, offset_az_deg,
            )
            hits[target_id] = result.region == REGION_GLASS and result.fiber_id == int(fiber_id)
        return hits


# --- scorer query interface ----------------------------------------------------------


def target_altaz(
    ra_deg: float, dec_deg: float, moment: datetime, config: Mapping
) -> tuple[float, float]:
    latitude, longitude = _site(config)
    return radec_to_altaz(ra_deg, dec_deg, moment, latitude, longitude)


def min_altitude_during(
    ra_deg: float,
    dec_deg: float,
    start_utc: datetime,
    end_utc: datetime,
    config: Mapping,
    step_seconds: int = 60,
) -> float:
    """Exact minimum altitude over [start, end] for the fixed-RA/Dec tracking model.

    ``step_seconds`` is accepted for compatibility but does not affect the result.
    """
    if end_utc <= start_utc:
        raise ValueError("end must follow start")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    latitude, longitude = _site(config)
    start_altitude, _ = radec_to_altaz(ra_deg, dec_deg, start_utc, latitude, longitude)
    end_altitude, _ = radec_to_altaz(ra_deg, dec_deg, end_utc, latitude, longitude)
    minimum = min(start_altitude, end_altitude)

    # sin(alt) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(hour_angle).
    # Hour angle increases at a constant rate in our LST model. Its only
    # interior minimum occurs at lower culmination (hour angle = 180 degrees).
    start_hour_angle = (_local_sidereal_deg(start_utc, longitude) - ra_deg) % 360.0
    swept_degrees = SIDEREAL_DEGREES_PER_DAY * (end_utc - start_utc).total_seconds() / 86400.0
    if (180.0 - start_hour_angle) % 360.0 <= swept_degrees:
        latitude_rad = math.radians(latitude)
        declination_rad = math.radians(dec_deg)
        lower_sin_altitude = (
            math.sin(latitude_rad) * math.sin(declination_rad)
            - math.cos(latitude_rad) * math.cos(declination_rad)
        )
        lower_altitude = math.degrees(math.asin(max(-1.0, min(1.0, lower_sin_altitude))))
        minimum = min(minimum, lower_altitude)
    return minimum


def altitude_ok(
    ra_deg: float,
    dec_deg: float,
    start_utc: datetime,
    end_utc: datetime,
    minimum_altitude_deg: float,
    config: Mapping,
    step_seconds: int = 60,
) -> bool:
    """Whether a target stays at or above the altitude limit for the whole exposure."""
    return (
        min_altitude_during(ra_deg, dec_deg, start_utc, end_utc, config, step_seconds)
        >= minimum_altitude_deg
    )


# --- action validation -----------------------------------------------------------------


def validate_action(action: Mapping, config: Mapping, n_fibers: int | None = None) -> None:
    """Validate an observe action against ACTION_SCHEMA; raises ValueError on violation."""
    fibers = int(n_fibers) if n_fibers is not None else int(config["field"]["n_fibers"])
    exposure = config["exposure"]
    if not isinstance(action, Mapping):
        raise ValueError("action must be a mapping")
    if set(action) != {"action", "pointing", "assignments", "duration_seconds"}:
        raise ValueError("action must contain exactly action/pointing/assignments/duration_seconds")
    if action["action"] != "observe":
        raise ValueError("action must be \"observe\"")
    pointing = action["pointing"]
    if not isinstance(pointing, Mapping) or set(pointing) != {"alt_deg", "az_deg"}:
        raise ValueError("pointing must contain exactly alt_deg and az_deg")
    alt = float(pointing["alt_deg"])
    az = float(pointing["az_deg"])
    if not 0.0 <= alt <= 90.0:
        raise ValueError("pointing alt_deg must lie in [0, 90]")
    if not 0.0 <= az < 360.0:
        raise ValueError("pointing az_deg must lie in [0, 360)")
    duration = int(action["duration_seconds"])
    if not int(exposure["min_duration_seconds"]) <= duration <= int(exposure["max_duration_seconds"]):
        raise ValueError(
            f"duration_seconds must lie in [{exposure['min_duration_seconds']}, "
            f"{exposure['max_duration_seconds']}]"
        )
    assignments = action["assignments"]
    if not isinstance(assignments, Mapping):
        raise ValueError("assignments must map fiber_id to target_id")
    seen_targets: set[str] = set()
    for fiber_key, target_id in assignments.items():
        try:
            fiber_id = int(fiber_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid fiber_id {fiber_key!r}") from exc
        if not 0 <= fiber_id < fibers:
            raise ValueError(f"fiber_id {fiber_id} outside 0..{fibers - 1}")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"invalid target_id {target_id!r}")
        if target_id in seen_targets:
            raise ValueError(f"target {target_id} assigned to more than one fiber")
        seen_targets.add(target_id)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fiber-map schema_version")
    FiberGrid.from_config(config)  # validates field parameters
    return config


# --- demo CLI and plot -------------------------------------------------------------------


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def run_demo(config: Mapping, targets_path: Path, moment: datetime, png_path: Path | None) -> dict[str, object]:
    grid = FiberGrid.from_config(config)
    latitude, longitude = _site(config)
    with targets_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = {row["target_id"]: (float(row["ra_deg"]), float(row["dec_deg"])) for row in rows}

    # Point at the first catalogue target's position at the demo moment so the field is
    # guaranteed to sit on sky with targets in it.
    first = rows[0]
    cmd_alt, cmd_az = radec_to_altaz(
        float(first["ra_deg"]), float(first["dec_deg"]), moment, latitude, longitude
    )
    cmd_alt, cmd_az = round(cmd_alt, 1), round(cmd_az, 1)

    actual_alt, actual_az = grid.actual_center(cmd_alt, cmd_az)
    counts = {REGION_GLASS: 0, REGION_FRAME: 0, REGION_OUTSIDE: 0}
    per_cell: dict[int, dict[str, list[str]]] = {}
    plotted: list[tuple[float, float, str, str]] = []
    actual_ra, actual_dec = altaz_to_radec(actual_alt, actual_az, moment, latitude, longitude)
    for target_id, (ra_deg, dec_deg) in targets.items():
        result = grid.classify_target(
            ra_deg, dec_deg, moment, cmd_alt, cmd_az, latitude, longitude
        )
        counts[result.region] += 1
        if result.region != REGION_OUTSIDE:
            offsets = grid.tangent_offsets(result.alt_deg, result.az_deg, actual_alt, actual_az)
            assert offsets is not None
            plotted.append((offsets[0], offsets[1], result.region, target_id))
            cell = per_cell.setdefault(result.fiber_id, {REGION_GLASS: [], REGION_FRAME: []})
            cell[result.region].append(target_id)

    # Assign every glass target to its own fiber (one per fiber, first come).
    assignments: dict[int, str] = {}
    for fiber_id in sorted(per_cell):
        glass = per_cell[fiber_id][REGION_GLASS]
        if glass:
            assignments[fiber_id] = glass[0]
    hits = grid.assigned_hits(assignments, targets, moment, cmd_alt, cmd_az, latitude, longitude)

    if png_path is not None:
        _render_demo(grid, plotted, set(hits), png_path, cmd_alt, cmd_az, moment)

    return {
        "grid": grid.derived(),
        "demo": {
            "moment_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commanded_pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
            "actual_pointing": {"alt_deg": round(actual_alt, 4), "az_deg": round(actual_az, 4)},
            "actual_center_radec": {"ra_deg": round(actual_ra, 4), "dec_deg": round(actual_dec, 4)},
            "targets_total": len(targets),
            "on_glass": counts[REGION_GLASS],
            "on_frame": counts[REGION_FRAME],
            "outside_field": counts[REGION_OUTSIDE],
            "assigned": len(assignments),
            "assigned_hits": sum(hits.values()),
        },
    }


def _render_demo(grid, plotted, hit_ids, png_path, cmd_alt, cmd_az, moment) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    half = grid.fov_side_deg / 2.0
    fig, ax = plt.subplots(figsize=(9, 9))
    middle = (grid.n_side - 1) / 2.0
    for row in range(grid.n_side):
        for col in range(grid.n_side):
            center_alt = (row - middle) * grid.pitch_deg
            center_az = (col - middle) * grid.pitch_deg
            side = grid.fiber_side_deg
            ax.add_patch(
                Rectangle(
                    (center_az - side / 2, center_alt - side / 2), side, side,
                    facecolor="#2e5aac", edgecolor="#081f4e", linewidth=1.2, zorder=1,
                )
            )
    for d_alt, d_az, region, target_id in plotted:
        if region == REGION_GLASS:
            color, size, zorder = "white", 14, 2
        else:
            color, size, zorder = "#333333", 10, 2
        ax.scatter([d_az], [d_alt], s=size, c=color, edgecolors="none", zorder=zorder)
        if target_id in hit_ids:
            ax.scatter(
                [d_az], [d_alt], s=60, facecolors="none",
                edgecolors="#e01515", linewidths=1.0, zorder=3,
            )
    ax.set_xlim(-half * 1.05, half * 1.05)
    ax.set_ylim(-half * 1.05, half * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("gnomonic offset toward increasing azimuth [deg]")
    ax.set_ylabel("gnomonic offset toward increasing altitude [deg]")
    ax.set_title(
        f"v4 fiber grid demo \u2014 commanded ({cmd_alt}\u00b0, {cmd_az}\u00b0) at "
        f"{moment.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        "blue = assignable regions (grid aligned with alt/az axes at exposure start), "
        "white dots = targets in field, red rings = assigned hits"
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="s", color="none", markerfacecolor="#2e5aac",
                   markeredgecolor="#081f4e", markersize=10, label="assignable region"),
            Line2D([], [], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#888888", markersize=5, label="target in field"),
            Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#e01515", markersize=8, label="assigned hit"),
        ],
        loc="upper right",
        fontsize=8,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="fiber map config JSON")
    parser.add_argument("--targets", type=Path, default=None, help="targets.csv (default: config demo section)")
    parser.add_argument("--moment", type=str, default=None, help="UTC timestamp for the demo exposure start")
    parser.add_argument("--png", type=Path, default=None, help="demo plot output path")
    args = parser.parse_args()
    config = load_config(args.config)
    demo = config.get("demo", {})

    def resolve(value, default=None):
        if value is None:
            return default
        return (args.config.parent / str(value)).resolve()

    targets_path = args.targets or resolve(demo.get("targets_csv"))
    moment = _parse_utc(args.moment or str(demo.get("moment_utc")))
    png_path = args.png if args.png is not None else resolve(demo.get("png_output"))
    result = run_demo(config, targets_path, moment, png_path)
    result["demo_png"] = str(png_path) if png_path else None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
