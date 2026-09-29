#!/usr/bin/env python3
"""扫描 _choose_quality 的阈值参数（AURORA_Q_* 环境变量），并列出分数分解。

    python3 tools/sweep_quality.py [scenario] [--sets hi1.05/step0.15/floor0.55 ...]
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAST = ROOT / "tools/fast_run.py"


def run_one(args):
    scenario, tag, env = args
    e = dict(os.environ)
    e.update(env)
    proc = subprocess.run([sys.executable, str(FAST), scenario],
                          capture_output=True, text=True, env=e, timeout=900)
    line = ""
    for ln in proc.stdout.splitlines():
        if "base=" in ln:
            line = ln.strip()
    if not line:
        return tag, None, (proc.stderr or "").strip()[-300:]
    fields = dict()
    for part in line.replace("pen={", "pen=").split():
        if "=" in part:
            k, _, v = part.partition("=")
            fields[k.strip("{")] = v.rstrip("}")
    try:
        return tag, {
            "base": float(fields["base"]), "bonus": float(fields["bonus"]),
            "req": float(fields["req"]), "pen": float(fields["pen"]),
            "total": float(fields["total"]),
        }, ""
    except Exception:
        return tag, None, line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default=str(ROOT / "agent-observer-starter-kit/scenarios/dev-reference"))
    ap.add_argument("--sets", nargs="*", default=[],
                    help="形如 hi0.90_step0.15_floor0.55")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    if not args.sets:
        args.sets = [f"w{w}_geo{g}" for w, g in itertools.product(
            (0.70, 0.80, 0.90, 1.00, 1.10, 1.20), (1.15,))]
    jobs = []
    for spec in args.sets:
        env = {}
        # 前缀按长度降序匹配，避免 "ws..." 被 "w" 抢走
        prefixes = (("bar", "AURORA_REL_BAR"), ("pick", "AURORA_PICK"), ("warm", "AURORA_WARM_K"), ("rho", "AURORA_RHO"), ("af", "AURORA_ABS_FLOOR"),
                    ("pressure", "AURORA_PRESSURE"), ("urg", "AURORA_REQ_URGENT"),
                    ("geo", "AURORA_GEO"), ("step", "AURORA_Q_STEP"), ("floor", "AURORA_Q_FLOOR"),
                    ("req", "AURORA_Q_REQ_SLACK"), ("ws", "AURORA_W_STEP"), ("wf", "AURORA_W_FLOOR"),
                    ("hi", "AURORA_Q_HI"), ("w", "AURORA_W"), ("qf", "AURORA_Q_FLOOR"))
        for part in spec.split("_"):
            if part in ("qgate", "wgate"):
                env["AURORA_GATE"] = part[:1]
                continue
            for prefix, var in prefixes:
                if part.startswith(prefix) and len(part) > len(prefix):
                    env[var] = part[len(prefix):]
                    break
        jobs.append((args.scenario, spec, env))

    print(f"{'set':34s} {'base':>10s} {'bonus':>9s} {'req':>8s} {'pen':>9s} {'TOTAL':>10s}")
    best = None
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for tag, res, err in ex.map(run_one, jobs):
            if res is None:
                print(f"{tag:34s} FAILED {err}")
                continue
            mark = ""
            if best is None or res["total"] > best[1]["total"]:
                best = (tag, res)
                mark = "  <-best"
            print(f"{tag:34s} {res['base']:10.2f} {res['bonus']:9.2f} {res['req']:8.2f} "
                  f"{res['pen']:9.2f} {res['total']:10.2f}{mark}")
    if best:
        print(f"\n最佳: {best[0]} → {best[1]['total']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
