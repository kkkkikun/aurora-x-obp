#!/usr/bin/env python3
"""Generate the v4 participant-facing target catalogue on an irregular southern footprint.

v4 replaces pre-computed tiles with a published sky region (one or more disconnected
star-shaped spherical polygons) plus a raw target list; participants derive rise/set and
plan their own pointings. This module is footprint-first: it builds the region, samples
targets inside it (uniform plus clustered), and resamples any target that has no
qualifying observing window from the configured southern-hemisphere site during the
configured date range.

Runtime code uses only the Python standard library. The sky-map renderer lives in the
separate organizer-side tool ``v4_sky_map.py`` (build-time matplotlib/numpy).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import SEED_DERIVATION_KEY, SEED_DERIVATIONS, sha256_file, stream_seed, write_exact_csv, write_text_lf
from .tile_geometry_simulator import _local_sidereal_deg, _sun_equatorial_deg


SCHEMA_VERSION = "v4-catalog-v2"
TARGET_CLASSES = ("ELG", "BGS", "LRG", "QSO", "Star")

V4_TARGET_COLUMNS = [
    "target_id",
    "ra_deg",
    "dec_deg",
    "target_class",
    "feature_flux",
    "science_weight",
    "required",
]

FOOTPRINT_COLUMNS = ["component_id", "vertex_index", "ra_deg", "dec_deg"]

SQDEG_PER_SR = (180.0 / math.pi) ** 2
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


@dataclass(frozen=True)
class FootprintComponent:
    """Star-shaped spherical polygon: vertices at increasing azimuth around the center.

    The boundary is the great-circle arc chain between consecutive vertices, which makes
    both the spherical-excess area and the point-in-polygon winding test exact.
    """

    component_id: str
    center_ra_deg: float
    center_dec_deg: float
    vertices: tuple[tuple[float, float], ...]
    area_deg2: float
    max_radius_deg: float


@dataclass(frozen=True)
class Target:
    target_id: str
    ra_deg: float
    dec_deg: float
    target_class: str
    feature_flux: float
    science_weight: float
    required: bool

    def csv_row(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "ra_deg": f"{self.ra_deg:.6f}",
            "dec_deg": f"{self.dec_deg:.6f}",
            "target_class": self.target_class,
            "feature_flux": f"{self.feature_flux:.8f}",
            "science_weight": f"{self.science_weight:.4f}",
            "required": "true" if self.required else "false",
        }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping) -> None:
    required = {"schema_version", "seed", "site", "footprint", "targets", "observability", "output"}
    if not required <= set(config) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid v4 catalog config keys or schema_version")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    mode = config.get(SEED_DERIVATION_KEY)
    if mode is not None and mode not in SEED_DERIVATIONS:
        raise ValueError(f"unsupported {SEED_DERIVATION_KEY} {mode!r}")
    site = config["site"]
    if not -90.0 <= float(site["latitude_deg"]) <= 90.0:
        raise ValueError("site latitude is outside [-90, 90]")
    if not -180.0 < float(site["longitude_deg"]) <= 180.0:
        raise ValueError("site longitude is outside (-180, 180]")
    if not -14.0 <= float(site["utc_offset_hours"]) <= 14.0:
        raise ValueError("UTC offset is outside [-14, 14]")
    footprint = config["footprint"]
    if float(footprint["total_area_deg2"]) <= 0.0:
        raise ValueError("total footprint area must be positive")
    if not 0.0 < float(footprint["area_tolerance_fraction"]) < 1.0:
        raise ValueError("area tolerance fraction must lie in (0, 1)")
    centers = footprint["component_centers"]
    weights = footprint["component_area_weights"]
    if int(footprint["n_components"]) != len(weights) or not (1 <= len(weights) <= len(centers)):
        raise ValueError("n_components must equal len(component_area_weights) and not exceed candidate centers")
    if any(float(weight) <= 0.0 for weight in weights):
        raise ValueError("component area weights must be positive")
    if int(footprint["vertices_per_component"]) < 8:
        raise ValueError("vertices_per_component must be at least 8")
    if float(footprint["component_gap_deg"]) < 0.0:
        raise ValueError("component gap must be non-negative")
    if not 0.0 <= float(footprint["pole_margin_deg"]) < 30.0:
        raise ValueError("pole margin must lie in [0, 30)")
    if any(float(value) < 0.0 for value in footprint["harmonic_amplitude_ranges"]):
        raise ValueError("harmonic amplitude ranges must be non-negative")
    for center in centers:
        if not 0.0 <= float(center[0]) < 360.0 or not -90.0 <= float(center[1]) <= 90.0:
            raise ValueError("component center coordinates out of range")
    targets = config["targets"]
    if int(targets["total_count"]) < 1:
        raise ValueError("total target count must be positive")
    if not 0.0 <= float(targets["required_fraction"]) < 1.0:
        raise ValueError("required fraction must lie in [0, 1)")
    fractions = targets["class_fractions"]
    if set(fractions) != set(TARGET_CLASSES):
        raise ValueError("class_fractions must define exactly the five v4 target classes")
    if any(float(value) < 0.0 for value in fractions.values()):
        raise ValueError("class fractions must be non-negative")
    if not math.isclose(sum(float(v) for v in fractions.values()), 1.0, rel_tol=1e-9):
        raise ValueError("class fractions must sum to 1")
    if not 0.0 <= float(targets["clustered_fraction"]) <= 1.0:
        raise ValueError("clustered fraction must lie in [0, 1]")
    if int(targets["n_cluster_centers"]) < 1 or float(targets["cluster_sigma_deg"]) <= 0.0:
        raise ValueError("invalid cluster settings")
    models = targets["models"]
    if set(models) != set(TARGET_CLASSES):
        raise ValueError("target models must define exactly the five v4 target classes")
    for name, model in models.items():
        if float(model["flux_median"]) <= 0.0 or float(model["flux_log_sigma"]) < 0.0:
            raise ValueError(f"{name}: invalid flux model")
        if float(model["science_weight"]) <= 0.0:
            raise ValueError(f"{name}: invalid science weight")
    observability = config["observability"]
    start = date.fromisoformat(str(observability["start_date"]))
    end = date.fromisoformat(str(observability["end_date"]))
    if end <= start:
        raise ValueError("observability end_date must follow start_date")
    if not -90.0 < float(observability["sun_altitude_limit_deg"]) < 0.0:
        raise ValueError("sun altitude limit must lie in (-90, 0)")
    if not 0.0 < float(observability["minimum_altitude_deg"]) < 90.0:
        raise ValueError("minimum altitude must lie in (0, 90)")
    if int(observability["minimum_window_seconds"]) < 1:
        raise ValueError("minimum window must be positive")
    if int(observability["max_position_attempts"]) < 1:
        raise ValueError("max_position_attempts must be positive")


# --- spherical geometry helpers -------------------------------------------------


def _unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))


def _ra_degrees(vector: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector) -> tuple[float, float, float]:
    norm = math.sqrt(_dot(vector, vector))
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _local_frame(ra_deg: float, dec_deg: float):
    """East and north tangent-plane unit vectors at the given direction."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    east = (-math.sin(ra), math.cos(ra), 0.0)
    north = (-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec))
    return east, north


def _offset_radec(
    center_ra_deg: float, center_dec_deg: float, azimuth_deg: float, radius_deg: float
) -> tuple[float, float]:
    """Move from the center by `radius_deg` along local azimuth (0=north, 90=east)."""
    center = _unit_vector(center_ra_deg, center_dec_deg)
    east, north = _local_frame(center_ra_deg, center_dec_deg)
    azimuth = math.radians(azimuth_deg)
    direction = tuple(
        math.cos(azimuth) * north[i] + math.sin(azimuth) * east[i] for i in range(3)
    )
    radius = math.radians(radius_deg)
    vector = tuple(
        math.cos(radius) * center[i] + math.sin(radius) * direction[i] for i in range(3)
    )
    vector = _normalize(vector)
    dec = math.degrees(math.asin(max(-1.0, min(1.0, vector[2]))))
    return _ra_degrees(vector), dec


def _signed_fan_solid_angle(
    pivot: tuple[float, float, float], a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """Signed solid angle of the fan triangle pivot->a->b (Van Oosterom & Strackee)."""
    numerator = _dot(pivot, _cross(a, b))
    denominator = 1.0 + _dot(a, b) + _dot(pivot, a) + _dot(pivot, b)
    return 2.0 * math.atan2(numerator, denominator)


def _polygon_area_deg2(center, vertices: Sequence[tuple[float, float]]) -> float:
    """Exact spherical-excess area; fan-triangulate from the (interior) center."""
    pivot = _unit_vector(*center)
    vectors = [_unit_vector(*vertex) for vertex in vertices]
    total = 0.0
    for index, a in enumerate(vectors):
        total += _signed_fan_solid_angle(pivot, a, vectors[(index + 1) % len(vectors)])
    total = abs(total)
    if total > 2.0 * math.pi:
        total = 4.0 * math.pi - total
    return total * SQDEG_PER_SR


def footprint_contains(
    components: Sequence[FootprintComponent], ra_deg: float, dec_deg: float
) -> bool:
    """Point-in-polygon test: wind the tangent-plane bearings from the query to each vertex.

    The sum of signed bearing changes around a simple spherical polygon is ±2π inside and
    ~0 outside, provided the polygon lies inside the hemisphere centered on the query — so
    components beyond their maximum vertex radius (in particular anywhere near the query
    point's antipode, where the bearing frame degenerates) must be skipped first via the
    angular-distance pre-filter. (A signed fan solid angle is NOT a containment test: like
    the planar shoelace formula, its value is the polygon area regardless of the origin.)
    """
    pivot = _unit_vector(ra_deg, dec_deg)
    east, north = _local_frame(ra_deg, dec_deg)
    for component in components:
        center = _unit_vector(component.center_ra_deg, component.center_dec_deg)
        if math.degrees(math.acos(max(-1.0, min(1.0, _dot(pivot, center))))) > (
            component.max_radius_deg + 1e-3
        ):
            continue
        total = 0.0
        previous_bearing: float | None = None
        first_bearing = 0.0
        for vertex in component.vertices:
            vector = _unit_vector(*vertex)
            along = _dot(vector, pivot)
            relative = tuple(vector[i] - along * pivot[i] for i in range(3))
            norm = math.sqrt(_dot(relative, relative))
            if norm < 1e-12:
                return True  # query point coincides with a vertex
            bearing = math.atan2(
                _dot(relative, east) / norm, _dot(relative, north) / norm
            )
            if previous_bearing is None:
                first_bearing = bearing
            else:
                total += (bearing - previous_bearing + math.pi) % (2.0 * math.pi) - math.pi
            previous_bearing = bearing
        if previous_bearing is not None:
            total += (first_bearing - previous_bearing + math.pi) % (2.0 * math.pi) - math.pi
        if abs(total) > math.pi:
            return True
    return False


# --- footprint generation --------------------------------------------------------


def _component_radius_deg(area_deg2: float) -> float:
    """Radius of the spherical cap with the given area; the scale-up starting point."""
    area_sr = area_deg2 / SQDEG_PER_SR
    return math.degrees(math.acos(max(-1.0, 1.0 - area_sr / (2.0 * math.pi))))


def generate_footprint(config: Mapping, rng: random.Random) -> list[FootprintComponent]:
    """Build `n_components` disconnected star-shaped polygons matching the target area.

    Each component's vertex radii follow a smooth random harmonic profile around its
    center; a global per-component scale is iterated with sqrt(area ratio) updates until
    the exact spherical-excess area matches the component's share of the target area.
    """
    footprint = config["footprint"]
    n_components = int(footprint["n_components"])
    n_vertices = int(footprint["vertices_per_component"])
    total_area = float(footprint["total_area_deg2"])
    tolerance = float(footprint["area_tolerance_fraction"])
    weights = [float(value) for value in footprint["component_area_weights"]]
    amplitudes = [float(value) for value in footprint["harmonic_amplitude_ranges"]]
    if sum(amplitudes) >= 0.95:
        raise ValueError("harmonic amplitudes must sum below 0.95 to stay star-shaped")
    centers = rng.sample(
        [tuple(float(value) for value in center) for center in footprint["component_centers"]],
        n_components,
    )
    weight_total = sum(weights)
    components: list[FootprintComponent] = []
    for index in range(n_components):
        target_area = total_area * weights[index] / weight_total
        phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in amplitudes]
        coeffs = [rng.uniform(0.0, amplitude) for amplitude in amplitudes]
        cap_radius = _component_radius_deg(target_area)
        radii = []
        for vertex_index in range(n_vertices):
            azimuth = 2.0 * math.pi * vertex_index / n_vertices
            profile = 1.0 + sum(
                coeff * math.cos((harmonic + 1) * azimuth + phase)
                for harmonic, (coeff, phase) in enumerate(zip(coeffs, phases))
            )
            radii.append(cap_radius * profile)
        center = centers[index]
        scale = 1.0
        vertices: tuple[tuple[float, float], ...] = ()
        area = 0.0
        for _ in range(60):
            vertices = tuple(
                _offset_radec(
                    center[0],
                    center[1],
                    math.degrees(2.0 * math.pi * vertex_index / n_vertices),
                    scale * radii[vertex_index],
                )
                for vertex_index in range(n_vertices)
            )
            area = _polygon_area_deg2(center, vertices)
            if abs(area - target_area) / target_area <= tolerance / 20.0:
                break
            scale *= math.sqrt(target_area / area)
        else:
            raise RuntimeError(f"footprint component {index} area did not converge")
        components.append(
            FootprintComponent(
                component_id=f"C{index:02d}",
                center_ra_deg=center[0],
                center_dec_deg=center[1],
                vertices=vertices,
                area_deg2=area,
                max_radius_deg=max(
                    _angular_separation_deg(center, vertex) for vertex in vertices
                ),
            )
        )
    _check_disconnected(components, float(footprint["component_gap_deg"]))
    pole_margin = float(footprint["pole_margin_deg"])
    for component in components:
        worst = max(abs(vertex[1]) for vertex in component.vertices)
        if worst > 90.0 - pole_margin:
            raise RuntimeError(
                f"component {component.component_id} comes within "
                f"{90.0 - worst:.2f} deg of a pole (< configured margin {pole_margin}); "
                "pole-wrapping polygons are not supported"
            )
    combined = sum(component.area_deg2 for component in components)
    if abs(combined - total_area) / total_area > tolerance:
        raise RuntimeError(
            f"footprint area {combined:.3f} deg^2 outside tolerance of {total_area:.1f}"
        )
    return components


def _angular_separation_deg(a, b) -> float:
    cosine = max(-1.0, min(1.0, _dot(_unit_vector(*a), _unit_vector(*b))))
    return math.degrees(math.acos(cosine))


def _check_disconnected(components: Sequence[FootprintComponent], gap_deg: float) -> None:
    """Require pairwise vertex separation above the configured gap, so the published
    components read as genuinely disconnected pieces of sky."""
    for index, component in enumerate(components):
        for vertex in component.vertices:
            if _angular_separation_deg(
                (component.center_ra_deg, component.center_dec_deg), vertex
            ) >= 90.0:
                raise RuntimeError(
                    f"component {component.component_id} vertex beyond 90 deg from its center"
                )
        for other in components[index + 1 :]:
            nearest = min(
                _angular_separation_deg(a, b)
                for a in component.vertices
                for b in other.vertices
            )
            if nearest < gap_deg:
                raise RuntimeError(
                    f"components {component.component_id} and {other.component_id} "
                    f"come within {nearest:.2f} deg (< configured gap {gap_deg})"
                )


# --- observability ----------------------------------------------------------------


def _sun_altitude_deg(moment: datetime, latitude_deg: float, longitude_deg: float) -> float:
    sun_ra, sun_dec = _sun_equatorial_deg(moment)
    hour_angle = math.radians(
        (_local_sidereal_deg(moment, longitude_deg) - sun_ra + 180.0) % 360.0 - 180.0
    )
    latitude = math.radians(latitude_deg)
    declination = math.radians(sun_dec)
    sin_altitude = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_altitude))))


def _twilight_crossing(
    left: datetime,
    right: datetime,
    left_below: bool,
    threshold: float,
    latitude_deg: float,
    longitude_deg: float,
    tolerance_seconds: int,
) -> datetime:
    """Bisect a sun-altitude threshold crossing, mirroring observing_calendar._crossing."""
    while (right - left).total_seconds() > tolerance_seconds:
        middle = left + (right - left) / 2
        if (_sun_altitude_deg(middle, latitude_deg, longitude_deg) <= threshold) == left_below:
            left = middle
        else:
            right = middle
    return right.replace(microsecond=0)


class WindowChecker:
    """Observing-window oracle for the v4 catalogue.

    A target qualifies when, on at least one night in the date range, it stays at or
    above the minimum altitude for an unbroken stretch of at least `minimum_window_seconds`
    while the Sun is below the twilight limit. Per night the check is analytic: the
    night spans an LST interval (Sun below the limit), the target is high enough while
    its hour angle is within H0 of transit, and the window length is the LST overlap
    converted at the sidereal rate. Only the per-night dusk/dawn LST brackets need a
    crossing solve, so 30k targets over a full season cost seconds, not hours.
    """

    def __init__(self, config: Mapping) -> None:
        site = config["site"]
        observability = config["observability"]
        self.latitude_deg = float(site["latitude_deg"])
        self.longitude_deg = float(site["longitude_deg"])
        self.utc_offset = timedelta(hours=float(site["utc_offset_hours"]))
        self.sun_limit_deg = float(observability["sun_altitude_limit_deg"])
        self.minimum_altitude_deg = float(observability["minimum_altitude_deg"])
        self.minimum_window_seconds = int(observability["minimum_window_seconds"])
        start = date.fromisoformat(str(observability["start_date"]))
        end = date.fromisoformat(str(observability["end_date"]))
        self.night_brackets: list[tuple[float, float]] = []
        cursor = start
        while cursor < end:
            bracket = self._night_lst_bracket(cursor)
            if bracket is not None:
                self.night_brackets.append(bracket)
            cursor += timedelta(days=1)
        if not self.night_brackets:
            raise ValueError("no observing nights inside the configured date range")

    def _night_lst_bracket(self, local_date: date) -> tuple[float, float] | None:
        """(LST at dusk, LST at dawn) with dawn unwrapped past 360 degrees."""
        local_zone = timezone(self.utc_offset)
        local_noon = datetime.combine(local_date, time(hour=12), tzinfo=local_zone)
        start = local_noon.astimezone(timezone.utc)
        end = start + timedelta(days=1)
        step = timedelta(seconds=300)
        crossings: list[tuple[datetime, bool]] = []
        previous = start
        previous_below = (
            _sun_altitude_deg(previous, self.latitude_deg, self.longitude_deg) <= self.sun_limit_deg
        )
        cursor = previous + step
        while cursor <= end:
            below = (
                _sun_altitude_deg(cursor, self.latitude_deg, self.longitude_deg)
                <= self.sun_limit_deg
            )
            if below != previous_below:
                crossings.append(
                    (
                        _twilight_crossing(
                            previous,
                            cursor,
                            previous_below,
                            self.sun_limit_deg,
                            self.latitude_deg,
                            self.longitude_deg,
                            1,
                        ),
                        below,
                    )
                )
            previous = cursor
            previous_below = below
            cursor += step
        dusk = next((moment for moment, entered in crossings if entered), None)
        dawn = next(
            (moment for moment, entered in crossings if not entered and dusk and moment > dusk),
            None,
        )
        if dusk is None or dawn is None:
            return None
        lst_dusk = _local_sidereal_deg(dusk, self.longitude_deg)
        night_seconds = (dawn - dusk).total_seconds()
        return lst_dusk, lst_dusk + night_seconds * SIDEREAL_DEG_PER_SECOND

    def max_altitude_deg(self, dec_deg: float) -> float:
        return 90.0 - abs(dec_deg - self.latitude_deg)

    def _hour_angle_limit_deg(self, dec_deg: float) -> float | None:
        """Half-width of the hour-angle range above the altitude limit; None if never up."""
        latitude = math.radians(self.latitude_deg)
        declination = math.radians(dec_deg)
        cosine = (
            math.sin(math.radians(self.minimum_altitude_deg))
            - math.sin(latitude) * math.sin(declination)
        ) / (math.cos(latitude) * math.cos(declination))
        if cosine >= 1.0:
            return None
        if cosine <= -1.0:
            return 180.0
        return math.degrees(math.acos(cosine))

    def window_seconds(self, ra_deg: float, dec_deg: float) -> tuple[int, float]:
        """(nights with a qualifying window, longest qualifying-run seconds in the season)."""
        half_width = self._hour_angle_limit_deg(dec_deg)
        if half_width is None:
            return 0, 0.0
        minimum_lst = self.minimum_window_seconds * SIDEREAL_DEG_PER_SECOND
        nights = 0
        longest = 0.0
        for lst_dusk, lst_dawn in self.night_brackets:
            if half_width == 180.0:
                overlap = lst_dawn - lst_dusk
            else:
                ra_unwrapped = (ra_deg - lst_dusk) % 360.0 + lst_dusk
                overlap = max(
                    max(
                        0.0,
                        min(lst_dawn, transit + half_width)
                        - max(lst_dusk, transit - half_width),
                    )
                    for transit in (ra_unwrapped - 360.0, ra_unwrapped, ra_unwrapped + 360.0)
                )
            if overlap >= minimum_lst:
                nights += 1
                longest = max(longest, overlap / SIDEREAL_DEG_PER_SECOND)
        return nights, longest

    def has_observing_window(self, ra_deg: float, dec_deg: float) -> bool:
        return self.window_seconds(ra_deg, dec_deg)[0] >= 1


# --- target sampling ----------------------------------------------------------------


def _sample_in_disc(
    rng: random.Random, center_ra: float, center_dec: float, radius_deg: float
) -> tuple[float, float]:
    azimuth = rng.uniform(0.0, 360.0)
    radius = radius_deg * math.sqrt(rng.uniform(0.0, 1.0))
    return _offset_radec(center_ra, center_dec, azimuth, radius)


def _class_roster(config: Mapping, rng: random.Random) -> list[str]:
    """Deterministic per-target class assignment matching the configured fractions."""
    total = int(config["targets"]["total_count"])
    fractions = config["targets"]["class_fractions"]
    counts = {name: int(round(total * float(fractions[name]))) for name in TARGET_CLASSES}
    drift = total - sum(counts.values())
    if drift:
        # Largest-remainder style fix-up, in canonical class order for determinism.
        order = sorted(
            TARGET_CLASSES,
            key=lambda name: (abs(total * float(fractions[name]) - counts[name]), name),
            reverse=True,
        )
        for index in range(abs(drift)):
            counts[order[index % len(order)]] += 1 if drift > 0 else -1
    roster = [name for name in TARGET_CLASSES for _ in range(counts[name])]
    rng.shuffle(roster)
    return roster


def sample_targets(
    config: Mapping,
    components: Sequence[FootprintComponent],
    checker: WindowChecker,
    rng: random.Random,
) -> tuple[list[Target], dict[str, object]]:
    """Sample the catalogue: uniform-plus-clustered positions, v3-style flux models,
    and resampling of any position without a qualifying observing window."""
    targets_cfg = config["targets"]
    total = int(targets_cfg["total_count"])
    clustered_fraction = float(targets_cfg["clustered_fraction"])
    n_clusters = int(targets_cfg["n_cluster_centers"])
    cluster_sigma = float(targets_cfg["cluster_sigma_deg"])
    max_attempts = int(config["observability"]["max_position_attempts"])
    models = targets_cfg["models"]

    roster = _class_roster(config, rng)
    required_count = int(round(total * float(targets_cfg["required_fraction"])))
    required_flags = set(rng.sample(range(total), required_count)) if required_count else set()

    areas = [component.area_deg2 for component in components]
    max_radii = [component.max_radius_deg for component in components]

    cluster_centers: list[tuple[float, float]] = []
    while len(cluster_centers) < n_clusters:
        index = rng.choices(range(len(components)), weights=areas, k=1)[0]
        ra, dec = _sample_in_disc(
            rng,
            components[index].center_ra_deg,
            components[index].center_dec_deg,
            max_radii[index],
        )
        if footprint_contains(components, ra, dec):
            cluster_centers.append((ra, dec))

    targets: list[Target] = []
    position_attempts = 0
    resampled_positions = 0
    clustered_count = 0
    window_nights: list[int] = []
    window_seconds: list[float] = []
    for sequence in range(total):
        accepted = None
        attempts = 0
        for _ in range(max_attempts):
            attempts += 1
            if rng.random() < clustered_fraction:
                center_ra, center_dec = rng.choice(cluster_centers)
                east = rng.gauss(0.0, cluster_sigma)
                north = rng.gauss(0.0, cluster_sigma)
                radius = math.hypot(east, north)
                azimuth = math.degrees(math.atan2(east, north)) % 360.0
                ra, dec = _offset_radec(center_ra, center_dec, azimuth, radius)
                clustered = True
            else:
                index = rng.choices(range(len(components)), weights=areas, k=1)[0]
                ra, dec = _sample_in_disc(
                    rng,
                    components[index].center_ra_deg,
                    components[index].center_dec_deg,
                    max_radii[index],
                )
                clustered = False
            if not footprint_contains(components, ra, dec):
                continue
            nights, longest = checker.window_seconds(ra, dec)
            if nights >= 1:
                accepted = (ra, dec, nights, longest, clustered)
                break
            resampled_positions += 1
        if accepted is None:
            raise RuntimeError(
                f"no observable position found within {max_attempts} attempts "
                f"for target {sequence + 1}"
            )
        position_attempts += attempts
        ra, dec, nights, longest, clustered = accepted
        window_nights.append(nights)
        window_seconds.append(longest)
        clustered_count += 1 if clustered else 0
        target_class = roster[sequence]
        model = models[target_class]
        flux = rng.lognormvariate(
            math.log(float(model["flux_median"])), float(model["flux_log_sigma"])
        )
        targets.append(
            Target(
                target_id=f"V4T{sequence + 1:06d}",
                ra_deg=ra,
                dec_deg=dec,
                target_class=target_class,
                feature_flux=flux,
                science_weight=float(model["science_weight"]),
                required=sequence in required_flags,
            )
        )
    stats = {
        "position_attempts": position_attempts,
        "resampled_positions": resampled_positions,
        "clustered_count": clustered_count,
        "window_nights_min": min(window_nights),
        "window_nights_mean": sum(window_nights) / len(window_nights),
        "longest_window_seconds_min": min(window_seconds),
        "longest_window_seconds_mean": sum(window_seconds) / len(window_seconds),
    }
    return targets, stats


# --- catalogue assembly ------------------------------------------------------------


def generate_catalog(config_path: Path, output_dir: Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    if output_dir is None:
        output_dir = (config_path.parent / str(config["output"]["directory"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(stream_seed(config, "v4.catalog", 0))
    components = generate_footprint(config, rng)
    checker = WindowChecker(config)
    targets, stats = sample_targets(config, components, checker, rng)

    footprint_path = output_dir / "footprint.csv"
    footprint_rows = (
        {
            "component_id": component.component_id,
            "vertex_index": vertex_index,
            "ra_deg": f"{vertex[0]:.6f}",
            "dec_deg": f"{vertex[1]:.6f}",
        }
        for component in components
        for vertex_index, vertex in enumerate(component.vertices)
    )
    write_exact_csv(footprint_path, FOOTPRINT_COLUMNS, footprint_rows)
    targets_path = output_dir / "targets.csv"
    write_exact_csv(targets_path, V4_TARGET_COLUMNS, (target.csv_row() for target in targets))

    by_class = {name: 0 for name in TARGET_CLASSES}
    for target in targets:
        by_class[target.target_class] += 1
    required_count = sum(1 for target in targets if target.required)
    total_area = sum(component.area_deg2 for component in components)
    configured_area = float(config["footprint"]["total_area_deg2"])
    min_gap = min(
        (
            _angular_separation_deg(a, b)
            for index, component in enumerate(components)
            for other in components[index + 1 :]
            for a in component.vertices
            for b in other.vertices
        ),
        default=None,
    )
    summary = {
        "schema_version": "v4-catalog-summary-v1",
        "generator": "challenge/v4_catalog_generator.py",
        "site": config["site"],
        "footprint": {
            "boundary_semantics": "great-circle arcs between consecutive vertices",
            "component_count": len(components),
            "component_centers": [
                [component.center_ra_deg, component.center_dec_deg] for component in components
            ],
            "component_areas_deg2": [round(component.area_deg2, 3) for component in components],
            "total_area_deg2": round(total_area, 3),
            "configured_area_deg2": configured_area,
            "area_error_fraction": abs(total_area - configured_area) / configured_area,
            "min_component_gap_deg": round(min_gap, 3) if min_gap is not None else None,
        },
        "targets": {
            "total_count": len(targets),
            "by_class": by_class,
            "required_count": required_count,
            "required_fraction": required_count / len(targets),
            "clustered_count": stats["clustered_count"],
            "position_attempts": stats["position_attempts"],
            "resampled_positions": stats["resampled_positions"],
        },
        "observability": {
            "date_range": [
                str(config["observability"]["start_date"]),
                str(config["observability"]["end_date"]),
            ],
            "nights_scanned": len(checker.night_brackets),
            "sun_altitude_limit_deg": checker.sun_limit_deg,
            "minimum_altitude_deg": checker.minimum_altitude_deg,
            "minimum_window_seconds": checker.minimum_window_seconds,
            "targets_with_window": len(targets),
            "window_nights_min": stats["window_nights_min"],
            "window_nights_mean": round(stats["window_nights_mean"], 2),
            "longest_window_seconds_min": round(stats["longest_window_seconds_min"], 1),
            "longest_window_seconds_mean": round(stats["longest_window_seconds_mean"], 1),
        },
        # No seed and no config hash: this summary may be published with the public inputs,
        # and a config hash would let a small seed be brute-forced.
        "sha256": {
            "footprint": sha256_file(footprint_path),
            "targets": sha256_file(targets_path),
        },
    }
    write_text_lf(
        output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="v4 catalog config JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the output directory named in the config",
    )
    args = parser.parse_args()
    summary = generate_catalog(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
