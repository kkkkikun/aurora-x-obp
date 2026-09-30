#!/usr/bin/env python3
"""Generate local v4 practice cards with arbitrary seeds/shapes (organizer generators, public configs).

    python3 make_cards.py s101            # default alpha-shape (35 nights, 9400 targets)
    python3 make_cards.py s202 --nights 14 --targets 3000
Writes starter_kit_v4/cards/<name>/{config,public,truth}. Layout identical to the official demo card.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
V4GEN = KIT.parent / "v4gen"
sys.path.insert(0, str(V4GEN))

from challenge.v4_catalog_generator import generate_catalog  # noqa: E402
from challenge.v4_config_check import cross_validate_generator_configs  # noqa: E402
from challenge.v4_weather_simulator import generate  # noqa: E402

SITE = {"name": "Paranal, Chile (virtual)", "latitude_deg": -24.6157, "longitude_deg": -70.3976, "utc_offset_hours": -4.0}


def catalog_config(seed: int, start: str, end: str, targets: int, area: float) -> dict:
    nights = (datetime_days(end) - datetime_days(start)).days
    return {
        "schema_version": "v4-catalog-v2", "seed": seed, "seed_derivation": "sha256-v1", "site": SITE,
        "footprint": {
            "total_area_deg2": area, "area_tolerance_fraction": 0.03, "n_components": 3,
            "component_area_weights": [0.4, 0.33, 0.27],
            "component_centers": [[350.0, -28.0], [35.0, -48.0], [220.0, -12.0]],
            "component_gap_deg": 2.0, "pole_margin_deg": 5.0, "vertices_per_component": 32,
            "harmonic_amplitude_ranges": [0.16, 0.10, 0.06],
        },
        "targets": {
            "total_count": targets, "required_fraction": 0.05,
            "class_fractions": {"ELG": 0.34, "BGS": 0.24, "LRG": 0.20, "QSO": 0.14, "Star": 0.08},
            "clustered_fraction": 0.25, "n_cluster_centers": max(20, targets // 400), "cluster_sigma_deg": 0.5,
            "models": {
                "ELG": {"flux_median": 0.55, "flux_log_sigma": 0.40, "science_weight": 1.0},
                "BGS": {"flux_median": 1.25, "flux_log_sigma": 0.30, "science_weight": 0.45},
                "LRG": {"flux_median": 0.35, "flux_log_sigma": 0.35, "science_weight": 1.0},
                "QSO": {"flux_median": 0.18, "flux_log_sigma": 0.50, "science_weight": 1.7},
                "Star": {"flux_median": 4.0, "flux_log_sigma": 0.25, "science_weight": 0.3},
            },
        },
        "observability": {"start_date": start, "end_date": end, "sun_altitude_limit_deg": -18.0,
                          "minimum_altitude_deg": 30.0, "minimum_window_seconds": 900, "max_position_attempts": 64},
        "output": {"directory": "."},
        "_nights": nights,
    }


def weather_config(seed: int, start: str, end: str, nights: int, args_stress=(False,)) -> dict:
    scale = max(1.0, nights / 7.0)
    return {
        "schema_version": "v4-weather-v2", "seed": seed, "seed_derivation": "sha256-v1", "site": SITE,
        "survey": {"start_date": start, "end_date": end, "slot_seconds": 900, "sun_altitude_limit_deg": -18.0},
        "quality": {
            "seeing_arcsec": {"nominal": 1.10, "seasonal_amplitude": 0.15, "night_sigma": 0.16, "slot_sigma": 0.07, "minimum": 0.50, "maximum": 4.00},
            "transparency": {"nominal": 0.88, "seasonal_amplitude": 0.07, "night_sigma": 0.06, "slot_sigma": 0.025, "minimum": 0.08, "maximum": 1.00},
            "sky_quality": {"nominal": 1.00, "seasonal_amplitude": 0.12, "night_sigma": 0.10, "slot_sigma": 0.035, "minimum": 0.05, "maximum": 1.50},
            "instrument_efficiency": {"jitter_minimum": 0.90, "jitter_maximum": 1.00, "minimum": 0.10, "maximum": 1.00},
            "night_correlation": 0.82, "slot_correlation": 0.88, "seasonal_phase_day": 210,
        },
        "background_closure": {"start_probability_per_open_slot": 0.008, "reopen_probability_per_closed_slot": 0.32,
                               "seasonal_probability_amplitude": 0.006},
        "weather_events": {
            "sector_width_deg_range": [40.0, 120.0],
            "sector_altitude_limit_deg_range": [40.0, 80.0],
            "conditions": {
                "rainy": {"count": round(1 * scale), "duration_slots": [6, 12], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 1.8, "transparency_multiplier": 0.25, "sky_quality_multiplier": 0.35},
                "cloudy": {"count": round(2 * scale), "duration_slots": [8, 20], "scope_weights": {"ALL": 0.35, "HORIZON_SECTOR": 0.65}, "force_close": False, "seeing_multiplier": 1.25, "transparency_multiplier": 0.55, "sky_quality_multiplier": 0.65},
                "smoggy": {"count": round(1 * scale), "duration_slots": [12, 30], "scope_weights": {"ALL": 0.30, "HORIZON_SECTOR": 0.70}, "force_close": False, "seeing_multiplier": 1.15, "transparency_multiplier": 0.68, "sky_quality_multiplier": 0.72},
                "cold_wave": {"count": round(1 * scale), "duration_slots": [24, 48], "scope_weights": {"ALL": 1.0}, "force_close": False, "seeing_multiplier": 1.05, "transparency_multiplier": 0.96, "sky_quality_multiplier": 0.98},
                "tornado": {"count": 0, "duration_slots": [4, 8], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 2.0, "transparency_multiplier": 0.15, "sky_quality_multiplier": 0.25},
            },
        },
        "rocket_launch": {"count": round(1 * scale), "duration_slots": [2, 4], "azimuth_sector_width_deg": [40.0, 80.0], "max_altitude_deg": [30.0, 60.0]},
        "earthquake": {"count": round(1 * scale), "magnitude_range": [4.5, 6.0], "impact_coefficient": 0.75, "reference_magnitude": 6.8,
                       "max_degradation": 0.95, "decay_nights": 2.0, "negligible_degradation": 0.01,
                       "seeing_impact_coefficient": 0.0, "transparency_impact_coefficient": 0.0, "sky_quality_impact_coefficient": 0.0},
        "terrain_obstruction": {"sector_count": [1, 2], "width_deg_range": [30.0, 60.0], "max_altitude_deg_range": [35.0, 45.0]},
        "instrument_fault": {"count": 1, "instrument_efficiency_multiplier_range": [0.5, 0.6]},
        "publication": {"forecast_interval_days": 7, "forecast_horizon_days": 7},
        "stress_tests": {
            "enabled": bool(args_stress[0]),
            "data_loss": {"trigger": "after_earthquake", "window_max_fraction": 0.05},
            "pointing_offset": {"max_abs_alt_rad": 0.002, "max_abs_az_rad": 0.002},
        } if args_stress[0] else None,
        "output": {"directory": "."},
    }


def datetime_days(date: str):
    from datetime import date as _date
    y, m, d = map(int, date.split("-"))
    return _date(y, m, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--nights", type=int, default=35)
    ap.add_argument("--targets", type=int, default=9400)
    ap.add_argument("--area", type=float, default=1880.0)
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--wallclock", type=int, default=900)
    ap.add_argument("--stress", action="store_true")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else abs(hash(args.name)) % 10_000_000
    from datetime import timedelta
    end = (datetime_days(args.start) + timedelta(days=args.nights + 1)).isoformat()

    catalog = catalog_config(seed, args.start, end, args.targets, args.area)
    nights = catalog.pop("_nights")
    weather = weather_config(seed, args.start, end, nights, args_stress=(args.stress,))
    fiber = json.loads((V4GEN / "reference" / "v4" / "v4_fiber_config.json").read_text())
    fiber["site"] = SITE
    score = json.loads((V4GEN / "reference" / "v4" / "v4_score_config.json").read_text())
    scenario = {
        "schema_version": "v4-scenario-v1", "name": args.name,
        "task_card": {"card_id": args.name, "scenario_slug": f"v4-{args.name}", "phase": "local"},
        "site": {**SITE, "sun_altitude_limit_deg": -18.0},
        "minimum_altitude_deg": 30.0,
        "limits": {"global_wallclock_seconds": args.wallclock},
        "fiber_config": "v4_fiber_config.json", "score_config": "v4_score_config.json",
        "products": {
            "targets_csv": "../public/targets.csv", "footprint_csv": "../public/footprint.csv",
            "night_calendar_csv": "../public/v4_night_calendar.csv",
            "bulletins_jsonl": "../public/v4_bulletins.jsonl", "forecasts_jsonl": "../public/v4_forecasts.jsonl",
            "slots_csv": "../truth/v4_slots.csv", "weather_truth_csv": "../truth/v4_weather_truth.csv",
            "events_csv": "../truth/v4_events.csv", "earthquake_effects_csv": "../truth/v4_earthquake_effects.csv",
        },
        "stress": {"enabled": args.stress,
                   **({"stress_events_csv": "../truth/v4_stress_events.csv"} if args.stress else {})},
    }
    cross_validate_generator_configs(catalog, weather, scenario, fiber, require_hashed_seeds=True)

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "catalog.json").write_text(json.dumps(catalog))
        (work / "weather.json").write_text(json.dumps(weather))
        generate_catalog(work / "catalog.json", work)
        generate(work / "weather.json", work)
        out = (KIT / "cards" / args.name).resolve()
        if out.exists():
            shutil.rmtree(out)
        for sub in ("config", "public", "truth"):
            (out / sub).mkdir(parents=True)
        for name in ("targets.csv", "footprint.csv", "v4_night_calendar.csv", "v4_bulletins.jsonl", "v4_forecasts.jsonl"):
            shutil.copyfile(work / name, out / "public" / name)
        for name in ("v4_slots.csv", "v4_weather_truth.csv", "v4_events.csv", "v4_earthquake_effects.csv"):
            shutil.copyfile(work / name, out / "truth" / name)
        if args.stress:
            shutil.copyfile(work / "v4_stress_events.csv", out / "truth" / "v4_stress_events.csv")
        (out / "config" / "v4_scenario.json").write_text(json.dumps(scenario, indent=2))
        (out / "config" / "v4_fiber_config.json").write_text(json.dumps(fiber, indent=2))
        (out / "config" / "v4_score_config.json").write_text(json.dumps(score, indent=2))
    print(f"card {args.name}: seed={seed} nights={args.nights} targets={args.targets} wallclock={args.wallclock} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
