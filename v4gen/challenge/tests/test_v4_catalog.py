"""Tests for the v4 footprint + target catalogue generator (challenge/v4_catalog_generator.py)."""
from __future__ import annotations

import copy
import json
import random
from datetime import date
from pathlib import Path

import pytest

from challenge import v4_catalog_generator as v4


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "v4"
DEFAULT_CONFIG_PATH = REFERENCE / "v4_catalog_config.json"

EXPECTED_TARGET_COLUMNS = [
    "target_id",
    "ra_deg",
    "dec_deg",
    "target_class",
    "feature_flux",
    "science_weight",
    "required",
]
EXPECTED_FOOTPRINT_COLUMNS = ["component_id", "vertex_index", "ra_deg", "dec_deg"]


def _small_config(tmp_path: Path, **overrides) -> dict:
    """Compact but structurally identical config: 2 components, 400 targets."""
    config = {
        "schema_version": "v4-catalog-v2",
        "seed": 424242,
        "site": {
            "name": "VISTA/Paranal (4MOST)",
            "latitude_deg": -24.6157,
            "longitude_deg": -70.3976,
            "utc_offset_hours": -4.0,
        },
        "footprint": {
            "total_area_deg2": 800.0,
            "area_tolerance_fraction": 0.02,
            "n_components": 2,
            "component_area_weights": [0.6, 0.4],
            "component_centers": [[335.0, -22.0], [65.0, -54.0], [145.0, -22.0]],
            "component_gap_deg": 2.0,
            "pole_margin_deg": 5.0,
            "vertices_per_component": 24,
            "harmonic_amplitude_ranges": [0.10, 0.06, 0.04],
        },
        "targets": {
            "total_count": 400,
            "required_fraction": 0.05,
            "class_fractions": {
                "ELG": 0.34,
                "BGS": 0.24,
                "LRG": 0.20,
                "QSO": 0.14,
                "Star": 0.08,
            },
            "clustered_fraction": 0.30,
            "n_cluster_centers": 12,
            "cluster_sigma_deg": 0.5,
            "models": {
                "ELG": {"flux_median": 0.55, "flux_log_sigma": 0.40, "science_weight": 1.0},
                "BGS": {"flux_median": 1.25, "flux_log_sigma": 0.30, "science_weight": 0.45},
                "LRG": {"flux_median": 0.35, "flux_log_sigma": 0.35, "science_weight": 1.0},
                "QSO": {"flux_median": 0.18, "flux_log_sigma": 0.50, "science_weight": 1.7},
                "Star": {"flux_median": 4.0, "flux_log_sigma": 0.25, "science_weight": 0.3},
            },
        },
        "observability": {
            "start_date": "2026-10-01",
            "end_date": "2027-02-01",
            "sun_altitude_limit_deg": -12.0,
            "minimum_altitude_deg": 30.0,
            "minimum_window_seconds": 900,
            "max_position_attempts": 64,
        },
        "output": {"directory": "unused-in-tests"},
    }
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split(".")
        config[section][key] = value
    return config


def _write_config(tmp_path: Path, config: dict, name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _run(tmp_path: Path, config: dict, name: str = "out") -> tuple[Path, dict]:
    config_path = _write_config(tmp_path, config, f"{name}.json")
    output_dir = tmp_path / name
    summary = v4.generate_catalog(config_path, output_dir)
    return output_dir, summary


def test_byte_determinism(tmp_path):
    config = _small_config(tmp_path)
    first_dir, _ = _run(tmp_path, config, "first")
    second_dir, _ = _run(tmp_path, copy.deepcopy(config), "second")
    for name in ("targets.csv", "footprint.csv", "summary.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes(), name


def test_area_within_tolerance(tmp_path):
    config = _small_config(tmp_path)
    _, summary = _run(tmp_path, config)
    footprint = summary["footprint"]
    configured = float(config["footprint"]["total_area_deg2"])
    tolerance = float(config["footprint"]["area_tolerance_fraction"])
    assert footprint["component_count"] == int(config["footprint"]["n_components"])
    assert abs(footprint["total_area_deg2"] - configured) / configured <= tolerance
    component_sum = sum(footprint["component_areas_deg2"])
    assert component_sum == pytest.approx(footprint["total_area_deg2"], rel=1e-3)


def test_window_guarantee(tmp_path):
    config = _small_config(tmp_path)
    output_dir, summary = _run(tmp_path, config)
    checker = v4.WindowChecker(config)
    assert summary["observability"]["nights_scanned"] > 100
    minimum = int(config["observability"]["minimum_window_seconds"])
    with (output_dir / "targets.csv").open("r", encoding="utf-8", newline="") as handle:
        import csv

        for row in csv.DictReader(handle):
            nights, longest = checker.window_seconds(float(row["ra_deg"]), float(row["dec_deg"]))
            assert nights >= 1, row["target_id"]
            assert longest >= minimum, row["target_id"]
    # Sanity: the checker must reject points the site can never raise to 30 deg altitude.
    assert checker.max_altitude_deg(60.0) < float(config["observability"]["minimum_altitude_deg"])
    assert not checker.has_observing_window(10.0, 60.0)
    # ... and must find windows for a favorable southern target.
    assert checker.has_observing_window(60.0, -40.0)


def test_window_checks_transit_before_dusk(tmp_path):
    checker = v4.WindowChecker(_small_config(tmp_path))
    dusk_lst, dawn_lst = checker.night_brackets[0]
    checker.night_brackets = [(dusk_lst, dawn_lst)]
    ra = (dusk_lst - 10.0) % 360.0
    dec = checker.latitude_deg
    half_width = checker._hour_angle_limit_deg(dec)
    expected = min(dawn_lst - dusk_lst, half_width - 10.0) / v4.SIDEREAL_DEG_PER_SECOND
    nights, longest = checker.window_seconds(ra, dec)
    assert nights == 1
    assert longest == pytest.approx(expected)


def test_csv_contract_and_required_fraction(tmp_path):
    config = _small_config(tmp_path)
    output_dir, summary = _run(tmp_path, config)
    targets_path = output_dir / "targets.csv"
    with targets_path.open("r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_TARGET_COLUMNS
        rows = list(reader)
    total = int(config["targets"]["total_count"])
    assert len(rows) == total
    required = [row for row in rows if row["required"] == "true"]
    assert len(required) == int(round(total * float(config["targets"]["required_fraction"])))
    assert all(row["required"] in ("true", "false") for row in rows)
    assert len({row["target_id"] for row in rows}) == total
    models = config["targets"]["models"]
    for row in rows:
        assert row["target_class"] in v4.TARGET_CLASSES
        assert 0.0 <= float(row["ra_deg"]) < 360.0
        assert -90.0 <= float(row["dec_deg"]) <= 90.0
        assert float(row["feature_flux"]) > 0.0
        weight = float(row["science_weight"])
        assert weight == pytest.approx(float(models[row["target_class"]]["science_weight"]))
    for name, count in summary["targets"]["by_class"].items():
        expected = round(total * float(config["targets"]["class_fractions"][name]))
        assert count == expected, name

    with (output_dir / "footprint.csv").open("r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_FOOTPRINT_COLUMNS
        footprint_rows = list(reader)
    n_components = int(config["footprint"]["n_components"])
    vertices = int(config["footprint"]["vertices_per_component"])
    assert len(footprint_rows) == n_components * vertices


def test_clustering_preserves_footprint_coverage(tmp_path):
    config = _small_config(tmp_path)
    output_dir, summary = _run(tmp_path, config)
    rng = random.Random(int(config["seed"]))
    components = v4.generate_footprint(config, rng)
    with (output_dir / "targets.csv").open("r", encoding="utf-8", newline="") as handle:
        import csv

        rows = list(csv.DictReader(handle))
    # Clustered sampling may thin out regions but must never leak outside the footprint.
    for row in rows:
        assert v4.footprint_contains(
            components, float(row["ra_deg"]), float(row["dec_deg"])
        ), row["target_id"]
    # Every component keeps a meaningful share of targets (coverage preserved).
    per_component = [0.0] * len(components)
    for row in rows:
        point = (float(row["ra_deg"]), float(row["dec_deg"]))
        for index, component in enumerate(components):
            if v4.footprint_contains([component], point[0], point[1]):
                per_component[index] += 1
                break
    area_shares = [c.area_deg2 / sum(c.area_deg2 for c in components) for c in components]
    for share, count in zip(area_shares, per_component):
        actual = count / len(rows)
        assert share / 3.0 <= actual <= min(1.0, share * 3.0), (share, actual)
    # Clustering actually happened, and some positions were resampled or rejected.
    assert summary["targets"]["clustered_count"] > 0
    assert summary["targets"]["position_attempts"] >= len(rows)


def _ray_cast_contains(vertices, ra_deg: float, dec_deg: float) -> bool:
    """Independent point-in-polygon oracle: count meridian-edge crossings north of the query."""
    import math

    ra = math.radians(ra_deg)
    normal = (-math.sin(ra), math.cos(ra), 0.0)
    along_meridian = (math.cos(ra), math.sin(ra), 0.0)
    vectors = [v4._unit_vector(*vertex) for vertex in vertices]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    crossings = 0
    for index, a in enumerate(vectors):
        b = vectors[(index + 1) % len(vectors)]
        da, db = dot(normal, a), dot(normal, b)
        if da * db >= 0.0:
            continue
        omega = math.acos(max(-1.0, min(1.0, dot(a, b))))
        lo, hi = 0.0, 1.0
        point = a
        for _ in range(60):
            mid = (lo + hi) / 2.0
            point = tuple(
                (math.sin((1 - mid) * omega) * a[j] + math.sin(mid * omega) * b[j])
                / math.sin(omega)
                for j in range(3)
            )
            if dot(normal, point) * da > 0.0:
                lo = mid
            else:
                hi = mid
        # The meridian plane holds both ra and ra+180; keep only the query's half-circle,
        # and only crossings north of the query point along it.
        if dot(point, along_meridian) <= 0.0:
            continue
        crossing_dec = math.degrees(math.asin(max(-1.0, min(1.0, point[2]))))
        if crossing_dec > dec_deg:
            crossings += 1
    return crossings % 2 == 1


def test_footprint_contains_matches_ray_casting():
    """Cross-check footprint_contains against an independent meridian ray-casting oracle
    on a grid spanning every component, including outside and near-antipodal points."""
    config = v4.load_config(DEFAULT_CONFIG_PATH)
    rng = random.Random(int(config["seed"]))
    components = v4.generate_footprint(config, rng)
    mismatches = []
    for component in components:
        ras = [v[0] for v in component.vertices]
        decs = [v[1] for v in component.vertices]
        ra_min, ra_max = min(ras) - 8.0, max(ras) + 8.0
        dec_min, dec_max = max(-89.9, min(decs) - 8.0), min(89.9, max(decs) + 8.0)
        for step_ra in range(25):
            for step_dec in range(25):
                ra = ra_min + (ra_max - ra_min) * step_ra / 24.0
                dec = dec_min + (dec_max - dec_min) * step_dec / 24.0
                expected = _ray_cast_contains(component.vertices, ra, dec)
                actual = v4.footprint_contains([component], ra, dec)
                if expected != actual:
                    mismatches.append((component.component_id, round(ra, 2), round(dec, 2), expected))
    assert not mismatches, mismatches[:10]


def test_default_config_validates_and_builds_footprint():
    config = v4.load_config(DEFAULT_CONFIG_PATH)
    rng = random.Random(int(config["seed"]))
    components = v4.generate_footprint(config, rng)
    assert len(components) == int(config["footprint"]["n_components"])
    total = sum(c.area_deg2 for c in components)
    configured = float(config["footprint"]["total_area_deg2"])
    assert abs(total - configured) / configured <= float(
        config["footprint"]["area_tolerance_fraction"]
    )
    checker = v4.WindowChecker(config)
    assert (date(2027, 2, 1) - date(2026, 10, 1)).days == len(checker.night_brackets)
