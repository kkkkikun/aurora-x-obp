#!/usr/bin/env python3
"""A/B bench: run two agent folders against one or more v4 cards and compare totals.

    python3 bench.py cards/demo [cards/alpha ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent
AGENTS = {"aurora": KIT / "agent", "base": KIT / "baseline_ref"}


def run(agent: Path, card: Path, out: Path) -> dict:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "local_runner.py", "--card", str(card), "--agent", str(agent),
         "--wallclock", "900", "--out", str(out), "--quiet"],
        cwd=KIT, capture_output=True, text=True, timeout=1200)
    wall = time.monotonic() - started
    summary = {}
    start = proc.stdout.find("{")
    if start >= 0:
        try:
            summary = json.loads(proc.stdout[start:])
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0 or not summary:
        print(f"  !! {agent.name} on {card.name}: rc={proc.returncode}\n{proc.stderr[-1500:]}")
    return {"total": summary.get("total"), "wall": wall,
            "required_missing": summary.get("required_missing"),
            "targets_observed": summary.get("targets_observed"),
            "termination": summary.get("termination_reason", "?")}


def main() -> int:
    cards = [Path(a) for a in sys.argv[1:]] or [KIT / "cards" / "demo"]
    only = None
    if cards and cards[0].name in AGENTS:
        only = AGENTS[cards[0].name]
        cards = cards[1:] or [KIT / "cards" / "demo"]
    for card in cards:
        print(f"== {card.name} ==")
        results = {}
        for name, agent in AGENTS.items():
            if only and agent != only:
                continue
            r = run(agent, card, KIT / f"bench_{name}")
            results[name] = r
            print(f"  {name:<7} total={r['total']!s:<12} req_miss={r['required_missing']} "
                  f"obs={r['targets_observed']} wall={r['wall']:.1f}s term={r['termination']}")
        if "aurora" in results and "base" in results and results["aurora"]["total"] is not None \
                and results["base"]["total"] is not None:
            delta = results["aurora"]["total"] - results["base"]["total"]
            print(f"  delta aurora-base: {delta:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
