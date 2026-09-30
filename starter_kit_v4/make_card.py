#!/usr/bin/env python3
"""Organizer tool: (re)generate the public v4 demo card shipped in starter_kit_v4/cards/demo.

    python3 scripts/make_v4_demo_card.py [--v4-source DIR] [--out starter_kit_v4/cards/demo]

The demo card is small (7 nights, 2,400 targets) and PUBLIC. It uses a fixed, non-secret seed
and is NOT one of the Playground / competition cards (alpha, beta, A-H). It exists so participants can
run the starter kit end to end in about a minute.

--v4-source is a checkout whose challenge/ holds the vendored v4 generators (this repository once the
v4 modules are merged, which is the default). The generators never ship in the kit.

Card layout (the v4 card bundle layout, integration plan "Protocol changes" item 11):
  config/v4_scenario.json        scenario config with task_card, site (incl. sun limit) and limits
  config/v4_fiber_config.json    instrument geometry
  config/v4_score_config.json    public score config
  public/targets.csv, footprint.csv, v4_night_calendar.csv, v4_bulletins.jsonl, v4_forecasts.jsonl
  truth/v4_slots.csv, v4_weather_truth.csv, v4_events.csv, v4_earthquake_effects.csv (local scoring only)
The generator configs use hashed per-stream seeds and pass v4_config_check. No seed or summary file is
copied into the card.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_SEED = 5021  # public demo seed; never reuse for a real card
SITE = {"name": "Paranal, Chile (virtual)", "latitude_deg": -24.6157, "longitude_deg": -70.3976, "utc_offset_hours": -4.0}
START, END = "2026-10-01", "2026-10-08"

CATALOG = {
    "schema_version": "v4-catalog-v2",
    "seed": DEMO_SEED,
    "seed_derivation": "sha256-v1",
    "site": SITE,
    "footprint": {
        "total_area_deg2": 800.0, "area_tolerance_fraction": 0.03, "n_components": 2,
        "component_area_weights": [0.55, 0.45],
        "component_centers": [[350.0, -28.0], [35.0, -48.0]],
        "component_gap_deg": 2.0, "pole_margin_deg": 5.0, "vertices_per_component": 32,
        "harmonic_amplitude_ranges": [0.16, 0.10, 0.06],
    },
    "targets": {
        "total_count": 2400, "required_fraction": 0.05,
        "class_fractions": {"ELG": 0.34, "BGS": 0.24, "LRG": 0.20, "QSO": 0.14, "Star": 0.08},
        "clustered_fraction": 0.25, "n_cluster_centers": 20, "cluster_sigma_deg": 0.5,
        "models": {
            "ELG": {"flux_median": 0.55, "flux_log_sigma": 0.40, "science_weight": 1.0},
            "BGS": {"flux_median": 1.25, "flux_log_sigma": 0.30, "science_weight": 0.45},
            "LRG": {"flux_median": 0.35, "flux_log_sigma": 0.35, "science_weight": 1.0},
            "QSO": {"flux_median": 0.18, "flux_log_sigma": 0.50, "science_weight": 1.7},
            "Star": {"flux_median": 4.0, "flux_log_sigma": 0.25, "science_weight": 0.3},
        },
    },
    "observability": {"start_date": START, "end_date": END, "sun_altitude_limit_deg": -18.0,
                      "minimum_altitude_deg": 30.0, "minimum_window_seconds": 900, "max_position_attempts": 64},
    "output": {"directory": "."},
}

WEATHER = {
    "schema_version": "v4-weather-v2",
    "seed": DEMO_SEED,
    "seed_derivation": "sha256-v1",
    "site": SITE,
    "survey": {"start_date": START, "end_date": END, "slot_seconds": 900, "sun_altitude_limit_deg": -18.0},
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
            "rainy": {"count": 1, "duration_slots": [6, 12], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 1.8, "transparency_multiplier": 0.25, "sky_quality_multiplier": 0.35},
            "cloudy": {"count": 2, "duration_slots": [8, 20], "scope_weights": {"ALL": 0.35, "HORIZON_SECTOR": 0.65}, "force_close": False, "seeing_multiplier": 1.25, "transparency_multiplier": 0.55, "sky_quality_multiplier": 0.65},
            "smoggy": {"count": 1, "duration_slots": [12, 30], "scope_weights": {"ALL": 0.30, "HORIZON_SECTOR": 0.70}, "force_close": False, "seeing_multiplier": 1.15, "transparency_multiplier": 0.68, "sky_quality_multiplier": 0.72},
            "cold_wave": {"count": 1, "duration_slots": [24, 48], "scope_weights": {"ALL": 1.0}, "force_close": False, "seeing_multiplier": 1.05, "transparency_multiplier": 0.96, "sky_quality_multiplier": 0.98},
            "tornado": {"count": 0, "duration_slots": [4, 8], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 2.0, "transparency_multiplier": 0.15, "sky_quality_multiplier": 0.25},
        },
    },
    "rocket_launch": {"count": 1, "duration_slots": [2, 4], "azimuth_sector_width_deg": [40.0, 80.0], "max_altitude_deg": [30.0, 60.0]},
    "earthquake": {"count": 1, "magnitude_range": [4.5, 6.0], "impact_coefficient": 0.75, "reference_magnitude": 6.8,
                   "max_degradation": 0.95, "decay_nights": 2.0, "negligible_degradation": 0.01,
                   "seeing_impact_coefficient": 0.0, "transparency_impact_coefficient": 0.0, "sky_quality_impact_coefficient": 0.0},
    "terrain_obstruction": {"sector_count": [1, 2], "width_deg_range": [30.0, 60.0], "max_altitude_deg_range": [35.0, 45.0]},
    "instrument_fault": {"count": 1, "instrument_efficiency_multiplier_range": [0.5, 0.6]},
    "publication": {"forecast_interval_days": 7, "forecast_horizon_days": 7},
    "output": {"directory": "."},
}

FIBER = {
    "schema_version": "v4-fiber-map-v1",
    "site": SITE,
    "field": {"fiber_area_deg2": 0.4, "gap_deg": 0.0, "n_fibers": 16},
    "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600},
}

PUBLIC = ("targets.csv", "footprint.csv", "v4_night_calendar.csv", "v4_bulletins.jsonl", "v4_forecasts.jsonl")
TRUTH = ("v4_slots.csv", "v4_weather_truth.csv", "v4_events.csv", "v4_earthquake_effects.csv")


def _find_source(explicit: str | None) -> Path:
    for candidate in (explicit, os.environ.get("V4_SOURCE"), str(ROOT)):
        if candidate and (Path(candidate) / "challenge" / "v4_weather_simulator.py").is_file():
            return Path(candidate).resolve()
    raise SystemExit("v4 generators not found: pass --v4-source <checkout with challenge/v4_weather_simulator.py>")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v4-source")
    parser.add_argument("--out", type=Path, default=ROOT / "starter_kit_v4" / "cards" / "demo")
    args = parser.parse_args(argv)
    source = _find_source(args.v4_source)
    sys.path.insert(0, str(source))
    from challenge.v4_catalog_generator import generate_catalog  # noqa: PLC0415
    from challenge.v4_config_check import cross_validate_generator_configs  # noqa: PLC0415
    from challenge.v4_runner import load_scenario  # noqa: PLC0415
    from challenge.v4_weather_simulator import generate  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "catalog.json").write_text(json.dumps(CATALOG), encoding="utf-8")
        (work / "weather.json").write_text(json.dumps(WEATHER), encoding="utf-8")
        generate_catalog(work / "catalog.json", work)
        generate(work / "weather.json", work)
        out = args.out.resolve()
        if out.exists():
            shutil.rmtree(out)
        for sub in ("config", "public", "truth"):
            (out / sub).mkdir(parents=True)
        for name in PUBLIC:
            shutil.copyfile(work / name, out / "public" / name)
        for name in TRUTH:
            shutil.copyfile(work / name, out / "truth" / name)
        score_path = next((path for path in (source / "challenge" / "reference" / "v4" / "v4_score_config.json",
                                             source / "config" / "v4_score_config.json") if path.is_file()), None)
        if score_path is None:
            raise SystemExit("v4_score_config.json not found in the v4 source")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        scenario = {
            "schema_version": "v4-scenario-v1",
            "name": "demo",
            "task_card": {"card_id": "demo", "scenario_slug": "v4-demo", "phase": "local"},
            "site": {**SITE, "sun_altitude_limit_deg": -18.0},
            "minimum_altitude_deg": 30.0,
            "limits": {"global_wallclock_seconds": 900},
            "fiber_config": "v4_fiber_config.json",
            "score_config": "v4_score_config.json",
            "products": {
                "targets_csv": "../public/targets.csv",
                "footprint_csv": "../public/footprint.csv",
                "night_calendar_csv": "../public/v4_night_calendar.csv",
                "bulletins_jsonl": "../public/v4_bulletins.jsonl",
                "forecasts_jsonl": "../public/v4_forecasts.jsonl",
                "slots_csv": "../truth/v4_slots.csv",
                "weather_truth_csv": "../truth/v4_weather_truth.csv",
                "events_csv": "../truth/v4_events.csv",
                "earthquake_effects_csv": "../truth/v4_earthquake_effects.csv",
            },
            "stress": {"enabled": False},
        }
        cross_validate_generator_configs(CATALOG, WEATHER, scenario, FIBER, require_hashed_seeds=True)
        dump = lambda path, data: path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # noqa: E731
        dump(out / "config" / "v4_score_config.json", score)
        dump(out / "config" / "v4_fiber_config.json", FIBER)
        dump(out / "config" / "v4_scenario.json", scenario)
        (out / "truth" / "README.md").write_text(
            "Local scoring data for the public demo card. Your agent never receives these files:\n"
            "the weather truth and events are only used by the local scorer. Do not read them from your agent.\n",
            encoding="utf-8")
        load_scenario(out / "config" / "v4_scenario.json")  # the runner's own consistency checks
    print(f"wrote demo card to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
