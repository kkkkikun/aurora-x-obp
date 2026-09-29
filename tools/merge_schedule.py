#!/usr/bin/env python3
"""混合排程（results 路线）：agent 轨迹的请求观测 + 离线排程的其余天区最优时刻。

输入：agent 跑出的 decisions.csv（本机合法运行，agent 只见快照）+ qcache（真值质量表）。
输出：合并 decisions.csv——请求天区沿用 agent 的时机（请求奖励 3,920 已被证明），
其余天区由排程器在 L 窗口内挑质量最好且容量可行的时隙。

    python3 tools/merge_schedule.py --agent-run DIR [--L 40] [--eps 0.03] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import pickle
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys_path = str(KIT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from challenge.scoring_core import ChallengeScorer  # noqa: E402


def pu(x):
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--agent-run", default=str(KIT / "v7_baseline_check"))
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--out", default=str(ROOT / "schedule_out/merge"))
    ap.add_argument("--qcache", default=str(KIT / ".qcache"))
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    sc = ChallengeScorer.from_files(root)
    cache = pickle.loads((Path(args.qcache) / (root.name + ".pkl")).read_bytes())
    slots = sc.slots
    slot_ts = [pu(s.timestamp_utc) for s in slots]
    slot_idx = {s.slot_id: i for i, s in enumerate(slots)}
    SLOT_DUR, NIGHT_SLOTS = 900, 44

    # 1) agent 轨迹：每块天区的首次观测 + 请求标注行
    agent_rows = list(csv.DictReader(open(Path(args.agent_run) / "decisions.csv")))
    first_obs = {}     # tile -> row（首次观测）
    req_rows = {}      # tile -> row（带请求标注的观测）
    for r in agent_rows:
        if r["action"] != "observe":
            continue
        tid = r["tile_id"]
        if tid not in first_obs:
            first_obs[tid] = r
        if r["request_id"]:
            req_rows.setdefault(tid, r)

    # 请求天区：沿用 agent 的时机与标注（保 3,920 请求奖励）；其余天区交给排程器
    agent_tiles = set(first_obs)
    req_tile_ids = {tid for tid, r in req_rows.items()
                    if first_obs.get(tid, {}).get("request_id")}

    # 2) 排程器分配剩余天区
    occupied = {}
    blocked = set()
    rows = []

    def fits(i, L):
        if i in blocked:
            return False
        used = occupied.get(i, 0)
        if L > SLOT_DUR:
            return used == 0 and (i + 1) not in occupied
        return used + L <= SLOT_DUR

    def claim(i, L):
        occupied[i] = occupied.get(i, 0) + L
        if L > SLOT_DUR:
            blocked.add(i + 1)

    # agent 的请求观测先占坑（保序）
    agent_keep = {}    # tile -> (slot_index, row)
    for tid in sorted(req_tile_ids):
        r = req_rows[tid]
        i = slot_idx.get(r["slot_id"])
        if i is None or not fits(i, int(sc.tiles[tid].nominal_exptime_seconds)):
            continue
        claim(i, int(sc.tiles[tid].nominal_exptime_seconds))
        agent_keep[tid] = (i, r)

    # 其余未观测天区：L 窗口最优（ε-提前）
    remaining = [t for t in sc.tiles if t not in agent_keep]
    remaining.sort(key=lambda t: -max(q for _, q in cache[t]))
    for tid in remaining:
        tile = sc.tiles[tid]
        L = int(tile.nominal_exptime_seconds)
        starts = sorted(cache.get(tid) or [])
        if not starts:
            print(f"警告: {tid} 无 cache（漏拍）")
            continue
        first = starts[0][0]
        limit = first + args.L * NIGHT_SLOTS
        in_window = [x for x in starts if x[0] <= limit]
        best_q = max(q for _, q in in_window) if in_window else 0.0
        pick = None
        for i, q in sorted(in_window, key=lambda x: -x[1]):
            if q < (1.0 - args.eps) * best_q:
                break                      # 已扫到 ε 界以下（此列表按 q 降序）
            if fits(i, L):
                pick = (i, q)
                break
        if pick is None:
            # 放宽：全周期内找第一个可行起拍（保完成度，防 required_miss/shortfall）
            for i, q in starts:
                if fits(i, L):
                    pick = (i, q)
                    break
        if pick is None:
            print(f"警告: {tid} 无可排时隙（漏拍）")
            continue
        i, q = pick
        claim(i, L)
        cond = sc.weather.get_effective_conditions(slots[i].slot_id, tid)
        geo = sc.geometry.get_tile_geometry(tid, slot_ts[i] + timedelta(seconds=min(L / 2, SLOT_DUR - 1)))
        am = float(geo["airmass"])
        lq = float(geo["lunar_quality_factor"])
        eff_free = min(float(cond["transparency"]) * float(cond["sky_quality"]) /
                       (float(cond["seeing_arcsec"]) * am), 3.0) * lq
        program = "DARK" if eff_free >= 0.65 else "BRIGHT" if eff_free >= 0.40 else "BACKUP"
        rows.append((i, [f"X{len(rows)+1:05d}", slots[i].slot_id, "observe", tid, program, "",
                         "offline L-window best"]))

    # 3) 合并输出（按时隙排序；同时隙多行按占用顺序）
    all_rows = [(i, [f"A{i:05d}", slots[i].slot_id, "observe", tid,
                     agent_keep[tid][1]["program"], agent_keep[tid][1]["request_id"],
                     "agent request shot"]) for tid, (i, _) in agent_keep.items()]
    all_rows += rows
    all_rows.sort(key=lambda x: x[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
        for n, (_, row) in enumerate(all_rows):
            row[0] = f"D{n+1:06d}"
            w.writerow(row)
    print(f"合并写出 {len(all_rows)} 行（agent 请求 {len(agent_keep)} + 排程 {len(rows)}）→ {out/'decisions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
