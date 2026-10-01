#!/usr/bin/env python3
"""Overnight regression matrix: run all cards × (aurora, base), print a summary table."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent
CARDS = ["demo", "s101", "s202", "s303", "s505", "s404stress",
         "s606", "s707", "s808", "s909", "s1010", "s1111", "s_short"]
AGENTS = {"aurora": KIT / "agent", "base": KIT / "baseline_ref"}


def run(agent: Path, card: Path, out: Path) -> dict:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "local_runner.py", "--card", str(card), "--agent", str(agent),
         "--wallclock", "900", "--out", str(out), "--quiet"],
        cwd=KIT, capture_output=True, text=True, timeout=1800)
    wall = time.monotonic() - started
    summary = {}
    start = proc.stdout.find("{")
    if start >= 0:
        try:
            summary = json.loads(proc.stdout[start:])
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0 or not summary:
        return {"total": None, "error": proc.stderr[-300:], "wall": wall}
    comp = summary.get("components") or {}
    return {"total": summary.get("total"),
            "req": summary.get("required_missing"),
            "obs": summary.get("targets_observed"),
            "unif": comp.get("uniformity_penalty"),
            "rep": comp.get("report_settlement"),
            "wall": wall}


def main() -> int:
    rows = []
    for name in CARDS:
        card = KIT / "cards" / name
        if not card.is_dir():
            print(f"MISSING card {name}", flush=True)
            continue
        r = {}
        for tag, agent in AGENTS.items():
            r[tag] = run(agent, card, KIT / f"matrix_{tag}")
        delta = (r["aurora"]["total"] or 0) - (r["base"]["total"] or 0)
        rows.append((name, r))
        a, b = r["aurora"], r["base"]
        print(f"{name:<12} aurora={a['total']!s:<12} base={b['total']!s:<12} "
              f"delta={delta:+.1f} req={a.get('req')}/{b.get('req')} "
              f"unif={a.get('unif')} rep={a.get('rep')} wallA={a['wall']:.0f}s", flush=True)
    print("\n== MATRIX DONE ==", flush=True)
    out = KIT / "matrix_result.json"
    out.write_text(json.dumps({n: r for n, r in rows}, indent=1))
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
