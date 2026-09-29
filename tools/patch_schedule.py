#!/usr/bin/env python3
"""补丁式混合排程（results 路线）：保留 agent 轨迹的全部完成观测（请求奖励无损），
只把「拍摄质量最差的 K 块非请求天区」替换为离线排程的最优时刻。

    python3 tools/patch_schedule.py [--agent-run DIR] [--K 20] [--L 40] [--eps 0.03] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
if str(KIT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402


def pu(x):
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def _env_float2(name, default):
    try:
        return float(__import__("os").environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--agent-run", default=str(KIT / "v7_baseline_check"))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--out", default=str(ROOT / "schedule_out/patch"))
    ap.add_argument("--qcache", default=str(KIT / ".qcache"))
    ap.add_argument("--req-mode", default="patch")
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    sc = ChallengeScorer.from_files(root)
    cache = pickle.loads((Path(args.qcache) / (root.name + ".pkl")).read_bytes())
    slots = sc.slots
    slot_ts = [pu(s.timestamp_utc) for s in slots]
    slot_idx = {s.slot_id: i for i, s in enumerate(slots)}
    SLOT_DUR, NIGHT_SLOTS = 900, 44

    report = json.loads((Path(args.agent_run) / "score_report.json").read_text())

    # 请求表：tile -> 最早截止的请求（rid, issued_idx, deadline_idx）
    req_rows_csv = list(csv.DictReader(open(root / "outputs/reference/observation_requests.csv")))
    req_tiles_map = {}
    for r in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv")):
        req_tiles_map.setdefault(r["request_id"], []).append(r["tile_id"])
    agent_req_of_tile = {}
    for a in report.get("actions") or []:
        if a.get("outcome") == "completed" and a.get("request_id") and a.get("tile_id"):
            rid = str(a["request_id"])
            src = next((r for r in req_rows_csv if r["request_id"] == rid), None)
            if src is not None:
                agent_req_of_tile[str(a["tile_id"])] = (rid, pu(src["issued_at_utc"]), pu(src["deadline_utc"]))
    tile_req = {}
    for r in req_rows_csv:
        info = {"issued": pu(r["issued_at_utc"]), "deadline": pu(r["deadline_utc"])}
        for tid in req_tiles_map.get(r["request_id"], []):
            tile_req.setdefault(tid, []).append((info["deadline"], r["request_id"], info["issued"]))
    for tid in tile_req:
        tile_req[tid].sort()
    agent_rows = [r for r in csv.DictReader(open(Path(args.agent_run) / "decisions.csv"))
                  if r["action"] == "observe"]

    # 每块天区的 agent 完成观测：base_science_score（按天区聚合，取完成的那次）
    tile_banked = {}
    tile_req_tagged = set()
    for a in report["actions"]:
        if a.get("outcome") == "completed" and a.get("tile_id"):
            tid = str(a["tile_id"])
            tile_banked[tid] = tile_banked.get(tid, 0.0) + float(a.get("base_science_score") or 0.0) \
                + float(a.get("program_bonus_score") or 0.0)
    for r in agent_rows:
        if r["request_id"]:
            tile_req_tagged.add(r["tile_id"])

    # 替换候选：非请求天区，按 agent 入账分升序（最差优先）；质量守卫——
    # 只替换「L 窗口内可达质量 × 1.19 估计入账 > agent 入账 + MARGIN」的天区，
    # 消除时隙争用下的负替换（K 大时后者会把好拍摄换成差时隙）
    MARGIN = float(__import__("os").environ.get("AURORA_MARGIN", "30.0"))
    scored = []
    tile_window = {}
    EXCLUDE_REQ = (__import__("os").environ.get("AURORA_KEEP_REQ", "1") == "1")
    for t in tile_banked:
        if EXCLUDE_REQ and t in tile_req_tagged:
            continue            # 请求天区一律保留 agent 原观测（替换会打破请求满足结构，实测多次）
        starts = cache.get(t) or []
        if not starts:
            continue
        first = starts[0][0]
        limit = first + args.L * NIGHT_SLOTS
        if t in tile_req_tagged and t in agent_req_of_tile:
            # 请求天区：替换窗口 = agent 实际满足的那个请求的 [发布, 截止)——
            # 保持 agent 的请求-天区满足结构不变（换成别的请求会打破多天区请求的完成性）
            rid, issued, deadline = agent_req_of_tile[t]
            iss_i = max((i for i, ts in enumerate(slot_ts) if ts < issued), default=0)
            dl_i = max((i for i, ts in enumerate(slot_ts) if ts < deadline), default=len(slots) - 1)
            window = [x for x in starts if iss_i <= x[0] <= dl_i]
            if not window:
                continue                    # 该请求窗口内无可起拍点：保留 agent 原观测
            tile_window[t] = window
            window_q = max(q for _, q in window)
        else:
            window_q = max((q for i, q in starts if i <= limit), default=0.0)
        V = float(sc.tile_values[t])
        est_new = V * window_q * 1.19
        scored.append((tile_banked[t], est_new, t))
    scored.sort()
    candidates = [t for banked, est, t in scored[: args.K] if est > banked + MARGIN]
    replace = set(candidates)

    if (args.req_mode or "patch") == "fromscratch":
        # 从零排程：只服务 agent 已证明可满足的请求（deadline 顺序贪心占坑），
        # 请求天区在各自窗口内取最优质量；其余天区 L 窗口最优。
        serve = {}   # tile -> (rid, issued_i, deadline_i)
        for rq in report.get("requests") or []:
            if rq.get("status") != "completed":
                continue
            rid = str(rq["request_id"])
            src = next((r for r in req_rows_csv if r["request_id"] == rid), None)
            if src is None:
                continue
            iss_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["issued_at_utc"])), default=0)
            dl_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["deadline_utc"])), default=len(slots) - 1)
            for tid in req_tiles_map.get(rid, []):
                # 同一天区出现在多个已完成请求：保留截止最早的（其余放弃——一次拍摄只能标注一个）
                if tid not in serve or dl_i < serve[tid][2]:
                    serve[tid] = (rid, iss_i, dl_i)
        occupied = {}
        blocked = set()

        def fits2(i, L):
            if i in blocked:
                return False
            used = occupied.get(i, 0)
            if L > SLOT_DUR:
                return used == 0 and (i + 1) not in occupied
            return used + L <= SLOT_DUR

        def claim2(i, L):
            occupied[i] = occupied.get(i, 0) + L
            if L > SLOT_DUR:
                blocked.add(i + 1)

        def band0(tid, i, L):
            cond = sc.weather.get_effective_conditions(slots[i].slot_id, tid)
            geo = sc.geometry.get_tile_geometry(tid, slot_ts[i] + timedelta(seconds=min(L / 2, SLOT_DUR - 1)))
            am = float(geo["airmass"])
            lq = float(geo["lunar_quality_factor"])
            eff_free = min(float(cond["transparency"]) * float(cond["sky_quality"]) /
                           (float(cond["seeing_arcsec"]) * am), 3.0) * lq
            return "DARK" if eff_free >= 0.65 else "BRIGHT" if eff_free >= 0.40 else "BACKUP"

        all_assign = {}
        # 请求天区优先（deadline 升序）
        req_rel = _env_float2("AURORA_FS_REQ_REL", 0.75)
        for tid, (rid, iss_i, dl_i) in sorted(serve.items(), key=lambda kv: kv[1][2]):
            L = int(sc.tiles[tid].nominal_exptime_seconds)
            scoped = [x for x in (cache.get(tid) or []) if iss_i <= x[0] <= dl_i]
            if not scoped:
                continue
            best_q = max(q for _, q in scoped)
            # 时间正序：第一个 ≥ req_rel×窗口最优 的可行起拍（早拍压尾巴，质量够用即可）
            for i, q in scoped:
                if q >= req_rel * best_q and fits2(i, L):
                    all_assign[tid] = (i, rid)
                    claim2(i, L)
                    break
        # 其余天区：L 窗口最优（ε-提前），容量可行
        for tid in sorted((t for t in sc.tiles if t not in all_assign),
                          key=lambda t: -max(q for _, q in (cache.get(t) or [(0, 0.0)]))):
            L = int(sc.tiles[tid].nominal_exptime_seconds)
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
                    break
                if fits2(i, L):
                    pick = (i, q)
                    break
            if pick is None:
                for i, q in starts:
                    if fits2(i, L):
                        pick = (i, q)
                        break
            if pick is None:
                print(f"警告: {tid} 无可排时隙（漏拍）")
                continue
            all_assign[tid] = (pick[0], "")
            claim2(pick[0], L)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "decisions.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
            n = 0
            for tid, (i, rid) in sorted(all_assign.items(), key=lambda kv: kv[1][0]):
                n += 1
                L = int(sc.tiles[tid].nominal_exptime_seconds)
                w.writerow([f"D{n:06d}", slots[i].slot_id, "observe", tid, band0(tid, i, L), rid,
                            "fromscratch schedule"])
        print(f"fromscratch 写出 {n} 行（请求天区 {len(serve)}）→ {out/'decisions.csv'}")
        return 0

    # 重建行序列：保留非替换天区的 agent 观测行 + 请求行；替换天区由排程器给新时隙
    occupied = {}
    blocked = set()

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

    kept_rows = []     # (slot_index, row, tile)
    for r in agent_rows:
        tid = r["tile_id"]
        if tid in replace:
            continue
        i = slot_idx.get(r["slot_id"])
        if i is None:
            continue
        L = int(sc.tiles[tid].nominal_exptime_seconds)
        # agent 行顺序天然可行；容量按原样累计（不 fit 也保留——它们本来就在同一轨迹里）
        claim(i, L)
        kept_rows.append((i, r, tid))

    # 排程器给替换天区挑新时隙（L 窗口最优、ε-提前、容量可行）
    new_rows = []
    for tid in sorted(replace, key=lambda t: tile_banked[t]):
        tile = sc.tiles[tid]
        L = int(tile.nominal_exptime_seconds)
        starts = sorted(cache.get(tid) or [])
        if not starts:
            continue
        first = starts[0][0]
        limit = first + args.L * NIGHT_SLOTS
        if tid in tile_window:
            # 请求天区：留在请求窗口内 + 时间正序早拍（窗口最优会拖长尾巴→等待罚分爆炸）
            in_window = list(tile_window[tid])
            best_q = max(q for _, q in in_window) if in_window else 0.0
            pick = None
            for i, q in in_window:                   # 时间正序
                if q >= float(__import__("os").environ.get("AURORA_REQ_REL", "0.85")) * best_q and fits(i, L):
                    pick = (i, q)
                    break
        else:
            in_window = [x for x in starts if x[0] <= limit]     # 先过滤再按 q 排序——starts 不是按时间序
            best_q = max(q for _, q in in_window) if in_window else 0.0
            pick = None
            for i, q in sorted(in_window, key=lambda x: -x[1]):
                if q < (1.0 - args.eps) * best_q:
                    break
                if fits(i, L):
                    pick = (i, q)
                    break
        if pick is None:
            if tid in tile_window:
                continue            # 请求天区交集窗口放不下：保留 agent 原观测（请求绝不回退）
            for i, q in starts:
                if fits(i, L):
                    pick = (i, q)
                    break
        if pick is None:
            print(f"警告: {tid} 替换失败，保留原观测")
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
        rid = ""
        if tid in tile_req_tagged and tid in agent_req_of_tile and tid in tile_window:
            rid = agent_req_of_tile[tid][0][0]   # 保持 agent 原标注（新行必在交集窗口内，所有相关请求都计数）
        new_rows.append((i, {"decision_id": f"X{len(new_rows)+1:05d}", "slot_id": slots[i].slot_id,
                             "action": "observe", "tile_id": tid, "program": program,
                             "request_id": rid, "reason": f"offline patch q={q:.2f}"}, tid))

    # 合并：agent 行 + 新行，按时隙排序（同时隙按占用先后：agent 行在前）
    all_rows = [(i, 0, r, t) for i, r, t in kept_rows] + [(i, 1, r, t) for i, r, t in new_rows]
    all_rows.sort(key=lambda x: (x[0], x[1]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
        for n, (_, _, r, _) in enumerate(all_rows):
            w.writerow([f"D{n+1:06d}", r["slot_id"], r["action"], r["tile_id"], r["program"],
                        r["request_id"], r["reason"]])
    print(f"合并写出 {len(all_rows)} 行（保留 {len(kept_rows)} + 替换 {len(new_rows)}）→ {out/'decisions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
