#!/usr/bin/env python3
"""Organizer-side sky map renderer for the v4 catalogue.

Mollweide all-sky projection in equatorial coordinates (RA increasing to the left, the
astronomical convention): light-gray RA/Dec graticule with tick labels, dark-blue
footprint outlines drawn from densified great-circle edges over a deep-blue fill, small
white dots for ordinary targets, and small red ringed markers for required targets.
matplotlib/numpy are build-time-only dependencies; the runtime catalogue generator
stays pure standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FOOTPRINT_FILL = "#2e5aac"
FOOTPRINT_EDGE = "#081f4e"
REQUIRED_RED = "#e01515"


def _unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))


def _mollweide_project(ra_deg, dec_deg):
    """Mollweide projection centered on RA=0, x axis flipped so RA increases leftward."""
    import numpy as np

    lon = np.radians((np.asarray(ra_deg, dtype=float) + 180.0) % 360.0 - 180.0)
    lat = np.radians(np.asarray(dec_deg, dtype=float))
    # Solve 2θ + sin(2θ) = π·sin(φ) by Newton iteration. The target is clipped just
    # inside the poles so the derivative 2 + 2cos(2θ) stays positive.
    target = math.pi * np.clip(np.sin(lat), -1.0 + 1e-12, 1.0 - 1e-12)
    theta = np.array(lat, copy=True)
    for _ in range(100):
        residual = 2.0 * theta + np.sin(2.0 * theta) - target
        derivative = np.maximum(2.0 + 2.0 * np.cos(2.0 * theta), 1e-9)
        theta = theta - residual / derivative
    x = -(2.0 * math.sqrt(2.0) / math.pi) * lon * np.cos(theta)
    y = math.sqrt(2.0) * np.sin(theta)
    return x, y


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _densify_edges(vertices: list[tuple[float, float]], subdivisions: int = 16):
    """Sample great-circle arcs between consecutive vertices so the projected boundary
    follows the true geodesic instead of a straight chord."""
    dense: list[tuple[float, float]] = []
    vectors = [_unit_vector(*vertex) for vertex in vertices]
    for index, start in enumerate(vectors):
        end = vectors[(index + 1) % len(vectors)]
        cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(start, end))))
        omega = math.acos(cosine)
        for step in range(subdivisions):
            fraction = step / subdivisions
            if omega < 1e-9:
                point = start
            else:
                front = math.sin((1.0 - fraction) * omega) / math.sin(omega)
                back = math.sin(fraction * omega) / math.sin(omega)
                point = tuple(front * a + back * b for a, b in zip(start, end))
            point_dec = math.degrees(math.asin(max(-1.0, min(1.0, point[2]))))
            point_ra = math.degrees(math.atan2(point[1], point[0])) % 360.0
            dense.append((point_ra, point_dec))
    return dense


def render_sky_map(
    targets_path: Path,
    footprint_path: Path,
    pdf_path: Path,
    png_path: Path | None = None,
    png_dpi: int = 200,
    summary_path: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    # Labels sit on white sky and deep-blue footprint alike; a white halo keeps the
    # dark-gray text readable in both places.
    halo = [path_effects.withStroke(linewidth=2.2, foreground="white", alpha=0.85)]

    targets = _read_csv(targets_path)
    footprint_rows = _read_csv(footprint_path)

    ra = np.array([float(row["ra_deg"]) for row in targets])
    dec = np.array([float(row["dec_deg"]) for row in targets])
    required = np.array([row["required"].strip().lower() == "true" for row in targets])
    x_all, y_all = _mollweide_project(ra, dec)

    components: dict[str, list[tuple[float, float]]] = {}
    for row in footprint_rows:
        components.setdefault(row["component_id"], []).append(
            (float(row["ra_deg"]), float(row["dec_deg"]))
        )

    half_width = 2.0 * math.sqrt(2.0)
    half_height = math.sqrt(2.0)
    fig, ax = plt.subplots(figsize=(14, 7.6))

    # Graticule: meridians every 30 deg, parallels every 30 deg, clipped naturally by
    # the ellipse boundary.
    dense_dec = np.linspace(-90.0, 90.0, 361)
    for meridian in range(0, 360, 30):
        gx, gy = _mollweide_project(np.full_like(dense_dec, float(meridian)), dense_dec)
        ax.plot(gx, gy, color="#b9b9b9", linewidth=0.5, zorder=0)
    dense_ra = np.linspace(0.0, 360.0, 1441)
    for parallel in range(-60, 61, 30):
        gx, gy = _mollweide_project(dense_ra, np.full_like(dense_ra, float(parallel)))
        ax.plot(gx, gy, color="#b9b9b9", linewidth=0.5, zorder=0)

    # Footprint fill and boundary (geodesic edges).
    for component_id in sorted(components):
        dense = _densify_edges(components[component_id])
        vx, vy = _mollweide_project(
            np.array([point[0] for point in dense]),
            np.array([point[1] for point in dense]),
        )
        ax.fill(vx, vy, facecolor=FOOTPRINT_FILL, edgecolor="none", zorder=1)
        ax.plot(vx, vy, color=FOOTPRINT_EDGE, linewidth=1.6, zorder=3)

    # Targets: small translucent white dots keep 30k points readable as density.
    ordinary = ~required
    ax.scatter(
        x_all[ordinary],
        y_all[ordinary],
        s=1.6,
        c="white",
        alpha=0.55,
        edgecolors="none",
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        x_all[required],
        y_all[required],
        s=7,
        facecolors="none",
        edgecolors=REQUIRED_RED,
        linewidths=0.55,
        alpha=0.85,
        zorder=4,
    )
    ax.scatter(
        x_all[required],
        y_all[required],
        s=1.2,
        c=REQUIRED_RED,
        alpha=0.85,
        edgecolors="none",
        linewidths=0,
        zorder=4,
    )

    # Ellipse boundary and tick labels.
    ax.add_patch(
        Ellipse(
            (0.0, 0.0),
            width=2.0 * half_width,
            height=2.0 * half_height,
            facecolor="none",
            edgecolor="#444444",
            linewidth=1.0,
            zorder=5,
        )
    )
    # RA labels along the equator (every 60 deg; RA increases leftward).
    for meridian in range(0, 360, 60):
        mx, _ = _mollweide_project(np.array([float(meridian)]), np.array([0.0]))
        x = float(mx[0])
        if abs(x) > half_width - 0.02:
            x = math.copysign(half_width - 0.28, x)
        ax.text(
            x,
            -0.13,
            f"{meridian}\u00b0",
            fontsize=8,
            color="#555555",
            ha="center",
            va="top",
            zorder=5,
            path_effects=halo,
        )
    # Dec labels along the central meridian.
    for parallel in range(-60, 61, 30):
        _, py = _mollweide_project(np.array([0.0]), np.array([float(parallel)]))
        ax.text(
            0.13,
            float(py[0]),
            f"{parallel:+d}\u00b0",
            fontsize=8,
            color="#555555",
            ha="left",
            va="center",
            zorder=5,
            path_effects=halo,
        )

    ax.set_xlim(-half_width * 1.04, half_width * 1.04)
    ax.set_ylim(-half_height * 1.12, half_height * 1.14)
    ax.set_aspect("equal")
    ax.axis("off")

    title = "v4 survey footprint and target catalogue (Mollweide all-sky, equatorial)"
    if summary_path is not None and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        area = summary["footprint"]["total_area_deg2"]
        total = summary["targets"]["total_count"]
        n_required = summary["targets"]["required_count"]
        title += (
            f"\narea {area:.0f} deg$^2$ in {summary['footprint']['component_count']} components"
            f" \u2014 {total} targets, {n_required} required"
        )
    ax.set_title(title, fontsize=13)
    legend = ax.legend(
        handles=[
            Line2D([], [], color=FOOTPRINT_EDGE, linewidth=1.8, label="footprint boundary"),
            Line2D(
                [], [], marker="o", color="none", markerfacecolor="white",
                markeredgecolor=FOOTPRINT_FILL, markeredgewidth=2.2, markersize=5,
                label="target",
            ),
            Line2D(
                [], [], marker="o", color="none", markerfacecolor=REQUIRED_RED,
                markeredgecolor=REQUIRED_RED, markersize=5, label="required target",
            ),
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.92,
    )
    legend.set_zorder(6)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="sky map config JSON")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def resolve(key: str) -> Path | None:
        value = config.get(key)
        return (args.config.parent / value).resolve() if value else None

    render_sky_map(
        targets_path=resolve("targets_csv"),
        footprint_path=resolve("footprint_csv"),
        pdf_path=resolve("pdf_output"),
        png_path=resolve("png_output"),
        png_dpi=int(config.get("png_dpi", 200)),
        summary_path=resolve("summary_json"),
    )
    print(json.dumps({"pdf": str(resolve("pdf_output")), "png": str(resolve("png_output"))}))


if __name__ == "__main__":
    main()
