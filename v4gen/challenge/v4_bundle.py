#!/usr/bin/env python3
"""Build one v4 task-card bundle (the layout read by ``challenge.v4_workflow``).

A card is described by a small spec; the generator configs start from the author's
reference configs in ``challenge/reference/v4`` and are cross-validated with hashed
seeds before anything is generated. Output layout::

    <root>/config/{v4_scenario.json, v4_fiber_config.json, v4_score_config.json}
    <root>/public/{targets.csv, footprint.csv, v4_night_calendar.csv, v4_bulletins.jsonl, v4_forecasts.jsonl}
    <root>/truth/{v4_slots.csv, v4_weather_truth.csv, v4_events.csv, v4_earthquake_effects.csv[, v4_stress_events.csv]}

No seed is written anywhere in the bundle. Pure standard library.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from . import v4_catalog_generator, v4_weather_simulator
from .v4_config_check import cross_validate_generator_configs

REFERENCE = Path(__file__).resolve().parent / "reference" / "v4"
PUBLIC = ("targets.csv", "footprint.csv", "v4_night_calendar.csv", "v4_bulletins.jsonl", "v4_forecasts.jsonl")
TRUTH = ("v4_slots.csv", "v4_weather_truth.csv", "v4_events.csv", "v4_earthquake_effects.csv")
SCENARIO_CONTRACT = "v4-score-v1"  # public.scenarios.contract of every v4 card


def _load(name: str) -> dict:
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def build_card_bundle(root: Path, spec: Mapping) -> Path:
    """Generate a card bundle into ``root`` (must not exist). Returns ``root``.

    spec keys: name, card_id, scenario_slug, phase, seed (int, secret for real cards),
    start_date, end_date, targets, area_deg2, stress (bool), wallclock_seconds (optional),
    event_counts (optional {condition|rocket_launch|earthquake|instrument_fault: n}).
    """
    root = Path(root)
    if root.exists():
        raise ValueError("bundle directory already exists")
    seed = int(spec["seed"])
    stress = bool(spec.get("stress", False))
    catalog = _load("v4_catalog_config.json")
    catalog.update(seed=seed, seed_derivation="sha256-v1")
    area = float(spec.get("area_deg2", 900.0))
    if area < 3000.0:
        catalog["footprint"].update(total_area_deg2=area, n_components=2, component_area_weights=[0.6, 0.4],
                                    vertices_per_component=24)
    else:
        catalog["footprint"]["total_area_deg2"] = area
    catalog["targets"]["total_count"] = int(spec["targets"])
    catalog["observability"].update(start_date=spec["start_date"], end_date=spec["end_date"])
    weather = _load("v4_weather_stress_config.json" if stress else "v4_weather_config.json")
    # Same card secret, different stream names: catalogue and weather stay independent.
    weather.update(seed=seed, seed_derivation="sha256-v1")
    weather["survey"].update(start_date=spec["start_date"], end_date=spec["end_date"])
    weather["stress_tests"]["enabled"] = stress
    for key, count in dict(spec.get("event_counts") or {}).items():
        if key in weather["weather_events"]["conditions"]:
            weather["weather_events"]["conditions"][key]["count"] = int(count)
        elif key in ("rocket_launch", "earthquake", "instrument_fault"):
            weather[key]["count"] = int(count)
        else:
            raise ValueError(f"unknown event family {key!r}")
    scenario = _load("v4_scenario_stress.json" if stress else "v4_scenario_default.json")
    scenario["name"] = str(spec["name"])
    scenario["task_card"] = {"card_id": str(spec["card_id"]), "scenario_slug": str(spec["scenario_slug"]),
                             "phase": str(spec["phase"])}
    scenario["fiber_config"] = "v4_fiber_config.json"
    scenario["score_config"] = "v4_score_config.json"
    scenario["products"] = {
        "targets_csv": "../public/targets.csv", "footprint_csv": "../public/footprint.csv",
        "night_calendar_csv": "../public/v4_night_calendar.csv", "bulletins_jsonl": "../public/v4_bulletins.jsonl",
        "forecasts_jsonl": "../public/v4_forecasts.jsonl", "slots_csv": "../truth/v4_slots.csv",
        "weather_truth_csv": "../truth/v4_weather_truth.csv", "events_csv": "../truth/v4_events.csv",
        "earthquake_effects_csv": "../truth/v4_earthquake_effects.csv",
    }
    scenario.pop("agent_params", None)
    if stress:
        scenario["stress"]["stress_events_csv"] = "../truth/v4_stress_events.csv"
    if spec.get("wallclock_seconds") is not None:
        scenario["limits"] = {"global_wallclock_seconds": int(spec["wallclock_seconds"])}
    fiber = _load("v4_fiber_config.json")
    fiber.pop("demo", None)
    cross_validate_generator_configs(catalog, weather, scenario, fiber, require_hashed_seeds=True)

    with tempfile.TemporaryDirectory(prefix="v4-card-") as temporary:
        work = Path(temporary)
        (work / "catalog.json").write_text(json.dumps(catalog))
        (work / "weather.json").write_text(json.dumps(weather))
        v4_catalog_generator.generate_catalog(work / "catalog.json", work / "out")
        v4_weather_simulator.generate(work / "weather.json", work / "out")
        for name in ("config", "public", "truth"):
            (root / name).mkdir(parents=True)
        (root / "config" / "v4_scenario.json").write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n")
        (root / "config" / "v4_fiber_config.json").write_text(json.dumps(fiber, indent=2, sort_keys=True) + "\n")
        shutil.copyfile(REFERENCE / "v4_score_config.json", root / "config" / "v4_score_config.json")
        for name in PUBLIC:
            shutil.copyfile(work / "out" / name, root / "public" / name)
        for name in TRUTH + (("v4_stress_events.csv",) if stress else ()):
            shutil.copyfile(work / "out" / name, root / "truth" / name)
    return root
