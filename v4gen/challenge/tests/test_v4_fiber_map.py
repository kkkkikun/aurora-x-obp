"""Tests for the v4 fiber map geometry library and action contract (MP-056)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from challenge import v4_fiber_map as fm


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "v4"
CONFIG_PATH = REFERENCE / "v4_fiber_config.json"
LAT, LON = -24.6157, -70.3976
MOMENT = datetime(2026, 10, 2, 4, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def config():
    return fm.load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def grid():
    return fm.FiberGrid(fiber_area_deg2=0.4, gap_deg=0.0, n_fibers=16)


def test_derived_quantities(grid, config):
    derived = grid.derived()
    assert config["field"] == {"fiber_area_deg2": 0.4, "gap_deg": 0.0, "n_fibers": 16}
    side = math.sqrt(0.4)
    assert grid.fiber_side_deg == pytest.approx(side)
    assert grid.pitch_deg == pytest.approx(side)
    assert grid.n_side == 4
    assert grid.fov_side_deg == pytest.approx(4 * side)
    assert derived["fov_area_deg2"] == pytest.approx(6.4)
    assert derived["glass_fill_fraction"] == pytest.approx(1.0)
    # fiber ids are row-major, fiber 0 at lower-left (lowest alt, smallest az).
    assert grid.fiber_row_col(0) == (0, 0)
    assert grid.fiber_row_col(15) == (3, 3)
    middle = (grid.n_side - 1) / 2.0
    assert grid.fiber_center_offset(0) == (-middle * grid.pitch_deg, -middle * grid.pitch_deg)


def test_n_fibers_must_be_perfect_square():
    with pytest.raises(ValueError):
        fm.FiberGrid(0.4, 0.05, 15)
    with pytest.raises(ValueError):
        fm.FiberGrid(0.4, 0.05, 0)


def _classify_radec(grid, ra, dec, cmd_alt, cmd_az, **offset):
    return grid.classify_target(ra, dec, MOMENT, cmd_alt, cmd_az, LAT, LON, **offset)


def _radec_for_altaz(alt, az):
    return fm.altaz_to_radec(alt, az, MOMENT, LAT, LON)


def _radec_for_plane_offset(center_alt, center_az, north_deg, east_deg):
    alt, az = math.radians(center_alt), math.radians(center_az)
    center = (math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt))
    north = (-math.sin(alt) * math.cos(az), -math.sin(alt) * math.sin(az), math.cos(alt))
    east = (-math.sin(az), math.cos(az), 0.0)
    vector = tuple(
        center[i] + math.radians(north_deg) * north[i] + math.radians(east_deg) * east[i]
        for i in range(3)
    )
    norm = math.sqrt(sum(value * value for value in vector))
    vector = tuple(value / norm for value in vector)
    target_alt = math.degrees(math.asin(vector[2]))
    target_az = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    return _radec_for_altaz(target_alt, target_az)


def test_hit_and_outside_field(grid):
    cmd_alt, cmd_az = 60.0, 180.0
    center_alt, center_az = grid.fiber_center_offset(0)  # lower-left fiber
    # Glass: target exactly at fiber 0's center.
    ra, dec = _radec_for_altaz(cmd_alt + center_alt, cmd_az + center_az / math.cos(math.radians(cmd_alt)))
    assert _classify_radec(grid, ra, dec, cmd_alt, cmd_az).region == fm.REGION_GLASS
    assert _classify_radec(grid, ra, dec, cmd_alt, cmd_az).fiber_id == 0
    # Outside the field entirely.
    ra, dec = _radec_for_altaz(cmd_alt + grid.fov_side_deg, cmd_az)
    assert _classify_radec(grid, ra, dec, cmd_alt, cmd_az).region == fm.REGION_OUTSIDE


def test_contiguous_regions_assign_shared_boundary_once(grid):
    cmd_alt, cmd_az = 60.0, 180.0
    _, center_az = grid.fiber_center_offset(5)
    # Exact tangent-plane seam uses the upper row; celestial coordinate
    # round-trips are checked a small distance to either side of it.
    assert grid.classify_offset(0.0, center_az) == (9, fm.REGION_GLASS)
    for north_deg, expected_fiber in ((-1e-6, 5), (1e-6, 9)):
        ra, dec = _radec_for_plane_offset(cmd_alt, cmd_az, north_deg, center_az)
        result = _classify_radec(grid, ra, dec, cmd_alt, cmd_az)
        assert (result.fiber_id, result.region) == (expected_fiber, fm.REGION_GLASS)
    assert grid.classify_offset(grid.fov_side_deg / 2, 0)[1] == fm.REGION_GLASS
    assert grid.classify_offset(grid.fov_side_deg / 2 + 1e-9, 0)[1] == fm.REGION_OUTSIDE


def test_tangent_projection_works_at_zenith_and_elsewhere(grid):
    north, east = grid.fiber_center_offset(10)
    for center_alt, center_az in ((60.0, 180.0), (90.0, 0.0), (90.0, 225.0)):
        ra, dec = _radec_for_plane_offset(center_alt, center_az, north, east)
        hit = _classify_radec(grid, ra, dec, center_alt, center_az)
        assert (hit.fiber_id, hit.region) == (10, fm.REGION_GLASS)
    offsets = grid.tangent_offsets(89.0, 90.0, 90.0, 0.0)
    assert offsets is not None
    assert offsets[1] == pytest.approx(math.degrees(math.tan(math.radians(1.0))), abs=1e-6)


def test_offset_past_zenith_cannot_hit(grid):
    ra, dec = _radec_for_altaz(89.9, 180.0)
    result = _classify_radec(grid, ra, dec, 90.0, 180.0, offset_alt_deg=0.1)
    assert result.region == fm.REGION_OUTSIDE


def test_neighbor_fiber_no_crosstalk(grid):
    cmd_alt, cmd_az = 60.0, 180.0
    center_alt, center_az = grid.fiber_center_offset(5)
    ra, dec = _radec_for_altaz(cmd_alt + center_alt, cmd_az + center_az / math.cos(math.radians(cmd_alt)))
    targets = {"T1": (ra, dec)}
    assert grid.assigned_hits({5: "T1"}, targets, MOMENT, cmd_alt, cmd_az, LAT, LON) == {"T1": True}
    # Same sky position assigned to a neighboring fiber is not a hit.
    assert grid.assigned_hits({6: "T1"}, targets, MOMENT, cmd_alt, cmd_az, LAT, LON) == {"T1": False}
    assert grid.assigned_hits({1: "T1"}, targets, MOMENT, cmd_alt, cmd_az, LAT, LON) == {"T1": False}


def test_azimuth_wrap(grid):
    cmd_alt, cmd_az = 60.0, 359.5
    # A target 0.4 deg east of the center crosses az=360; the wrap must not fling it outside.
    ra, dec = _radec_for_altaz(cmd_alt, 359.9)
    result = _classify_radec(grid, ra, dec, cmd_alt, cmd_az)
    assert result.region == fm.REGION_GLASS
    assert result.fiber_id is not None


def test_offset_changes_actual_center_and_hits(grid):
    cmd_alt, cmd_az = 60.0, 180.0
    center_alt, center_az = grid.fiber_center_offset(5)
    ra, dec = _radec_for_altaz(cmd_alt + center_alt, cmd_az + center_az / math.cos(math.radians(cmd_alt)))
    targets = {"T1": (ra, dec)}
    # Without offset the target lands on fiber 5.
    assert grid.assigned_hits({5: "T1"}, targets, MOMENT, cmd_alt, cmd_az, LAT, LON) == {"T1": True}
    # A +1-pitch altitude offset moves the true field up one row: the target now sits in
    # fiber 1 (same column, one row down) — the assignment to fiber 5 misses.
    offset = {"offset_alt_deg": grid.pitch_deg}
    result = _classify_radec(grid, ra, dec, cmd_alt, cmd_az, **offset)
    assert result.fiber_id == 1
    assert result.region == fm.REGION_GLASS
    assert grid.assigned_hits({5: "T1"}, targets, MOMENT, cmd_alt, cmd_az, LAT, LON, **offset) == {"T1": False}
    # Actual center bookkeeping: command + offset in alt/az space.
    actual_alt, actual_az = grid.actual_center(cmd_alt, 359.0, 0.25, 2.0)
    assert actual_alt == pytest.approx(60.25)
    assert actual_az == pytest.approx(1.0)


def test_altaz_radec_roundtrip():
    for alt, az in ((48.3, 117.2), (30.0, 1.0), (89.9, 270.5), (10.0, 359.9)):
        ra, dec = fm.altaz_to_radec(alt, az, MOMENT, LAT, LON)
        alt2, az2 = fm.radec_to_altaz(ra, dec, MOMENT, LAT, LON)
        assert alt2 == pytest.approx(alt, abs=1e-8)
        assert (az2 - az + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-8)


def test_altitude_queries(config):
    ra, dec = fm.altaz_to_radec(50.0, 100.0, MOMENT, LAT, LON)
    alt, az = fm.target_altaz(ra, dec, MOMENT, config)
    assert alt == pytest.approx(50.0, abs=1e-8)
    start = MOMENT
    end = MOMENT + timedelta(seconds=900)
    assert fm.altitude_ok(ra, dec, start, end, 30.0, config)
    assert not fm.altitude_ok(ra, dec, start, end, 60.0, config)
    assert fm.min_altitude_during(ra, dec, start, end, config) == pytest.approx(alt, abs=0.2)


def test_altitude_query_includes_short_exposure_endpoint(config):
    start = MOMENT
    end = start + timedelta(seconds=60)
    ra, dec = fm.altaz_to_radec(30.1, 270.0, start, LAT, LON)
    end_altitude, _ = fm.radec_to_altaz(ra, dec, end, LAT, LON)
    assert end_altitude < 30.0
    assert fm.min_altitude_during(ra, dec, start, end, config, step_seconds=120) == pytest.approx(end_altitude)
    assert not fm.altitude_ok(ra, dec, start, end, 30.0, config, step_seconds=120)


def test_analytic_altitude_minimum_catches_dip_between_samples():
    # Lower culmination occurs 60 s into this 120 s exposure. Checking only
    # the two endpoints would incorrectly accept the 30-degree limit.
    config = {"site": {"latitude_deg": 80.0, "longitude_deg": LON}}
    start = MOMENT
    middle = start + timedelta(seconds=60)
    end = start + timedelta(seconds=120)
    ra, dec = fm.altaz_to_radec(29.99998, 0.0, middle, 80.0, LON)
    for moment in (start, end):
        altitude, _ = fm.radec_to_altaz(ra, dec, moment, 80.0, LON)
        assert altitude > 30.0
    assert fm.min_altitude_during(ra, dec, start, end, config, step_seconds=120) == pytest.approx(29.99998)
    assert not fm.altitude_ok(ra, dec, start, end, 30.0, config, step_seconds=120)


def _valid_action():
    return {
        "action": "observe",
        "pointing": {"alt_deg": 55.0, "az_deg": 200.0},
        "assignments": {"3": "V4T000001", 7: "V4T000002"},
        "duration_seconds": 900,
    }


def test_validate_action_ok(config):
    fm.validate_action(_valid_action(), config)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda a: a.update({"action": "wait"}), "observe"),
        (lambda a: a["pointing"].update({"alt_deg": 91.0}), "alt_deg"),
        (lambda a: a["pointing"].update({"alt_deg": -0.5}), "alt_deg"),
        (lambda a: a["pointing"].update({"az_deg": 360.0}), "az_deg"),
        (lambda a: a["pointing"].update({"az_deg": -1.0}), "az_deg"),
        (lambda a: a.update({"duration_seconds": 59}), "duration"),
        (lambda a: a.update({"duration_seconds": 3601}), "duration"),
        (lambda a: a["assignments"].update({"16": "V4T000003"}), "fiber_id 16"),
        (lambda a: a["assignments"].update({"-1": "V4T000003"}), "outside"),
        (lambda a: a["assignments"].update({"x": "V4T000003"}), "invalid fiber_id"),
        (lambda a: a["assignments"].update({5: "V4T000001"}), "more than one fiber"),
        (lambda a: a.update({"extra": 1}), "exactly"),
        (lambda a: a.pop("assignments"), "exactly"),
    ],
)
def test_validate_action_branches(config, mutate, needle):
    action = _valid_action()
    mutate(action)
    with pytest.raises(ValueError, match=needle):
        fm.validate_action(action, config)


def test_demo_cli_smoke(tmp_path, v4_reference_dir):
    pytest.importorskip("matplotlib")  # build-time-only dependency of the demo plot
    png = tmp_path / "demo.png"
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "challenge.v4_fiber_map",
         "--config", str(v4_reference_dir / "v4_fiber_config.json"), "--png", str(png)],
        cwd=ROOT.parent, capture_output=True, text=True, timeout=240,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout)
    assert result["grid"]["n_side"] == 4
    demo = result["demo"]
    assert demo["on_glass"] > 0
    assert demo["on_frame"] == 0
    assert demo["assigned_hits"] == demo["assigned"] > 0
    assert demo["on_glass"] + demo["on_frame"] + demo["outside_field"] == demo["targets_total"]
    assert png.is_file() and png.stat().st_size > 0
