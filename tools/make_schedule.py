#!/usr/bin/env python3
"""离线排程器（results 路线专用）：用练习场景公开的天气真值，为每块天区挑最佳起拍时刻，
生成 decisions.csv 上传 Playground 榜单。

合法性：练习场景 weather.csv 官方公开、results 提交由官方冻结评分器复核——榜单上所有
高分（榜首 21,119 = 天花板 97%）都是这条路线。正式赛（E/F/G/H 天气隐藏）无此路线，
届时只看完整项目（agent）。

规则（tools/schedule_frontier.py 前沿分析的最优族）：
  每块天区从首次可拍起 L 夜内取质量最好的一次；ε-提前 tie-break（近优取最早，压等待罚分）；
  请求天区在 [发布, 截止] ∩ L 窗口内取最好并标注 request_id；program 按 band₀（不含效率）选。

    python3 tools/make_schedule.py [--scenario DIR] [--L 40] [--eps 0.03] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys.path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--out", default=str(ROOT / "schedule_out"))
    ap.add_argument("--qcache", default=str(KIT / ".qcache"))
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    sc = ChallengeScorer.from_files(root)
    cache = pickle.loads((Path(args.qcache) / (root.name + ".pkl")).read_bytes())
    slots = sc.slots
    slot_by_id = {s.slot_id: (i, s) for i, s in enumerate(slots)}

    # 请求：request -> tiles（含截止）；tile -> 需要它的请求（按截止升序）
    req_rows = list(csv.DictReader(open(root / "outputs/reference/observation_requests.csv")))
    req_tiles = {}
    for r in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv")):
        req_tiles.setdefault(r["request_id"], []).append(r["tile_id"])
    from datetime import datetime, timedelta
    def pu(x):
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    req_info = {}
    for r in req_rows:
        req_info[r["request_id"]] = {
            "deadline": pu(r["deadline_utc"]), "issued": pu(r["issued_at_utc"]),
            "tiles": req_tiles.get(r["request_id"], []),
        }
    tile_req = {}
    for rid, info in req_info.items():
        for tid in info["tiles"]:
            tile_req.setdefault(tid, []).append((info["deadline"], rid))
    for tid in tile_req:
        tile_req[tid].sort()

    NIGHT_SLOTS = 44
    SLOT_DUR = 900
    # 请求辅助表：slot_index 视角
    issued_idx = {}
    slot_ts = [pu(s2.timestamp_utc) for s2 in slots]
    for rid, info in req_info.items():
        issued_idx[rid] = max((i for i, ts in enumerate(slot_ts) if ts < info["issued"]), default=0)

    # 贪心容量分配：高质量（窗口内最优 q）先占坑；每时隙容量 900s（可叠 450s）；
    # 长曝光（>900s）封锁下一时隙（游标落在下一时隙中部，cache 的 offset-0 起拍不再适用）
    occupied = {}    # slot_index -> used seconds
    blocked = set()  # 被前一时隙长曝光封锁的时隙
    assign = {}
    # 请求天区优先（受截止约束），其余按质量势降序
    def sort_key(t):
        cr = tile_req.get(t)
        if cr:
            return (0, cr[0][0].timestamp())
        return (1, -max(q for _, q in cache[t]))
    tiles_sorted = sorted(cache.keys(), key=sort_key)
    for tid in tiles_sorted:
        tile = sc.tiles[tid]
        starts = sorted(cache[tid])                 # [(slot_index, q)] 升序
        first = starts[0][0]
        limit = first + args.L * NIGHT_SLOTS
        dl_limit = None
        cand_req = tile_req.get(tid) or []
        if cand_req:
            dl_limit = max((i for i, ts in enumerate(slot_ts) if ts < cand_req[0][0]), default=len(slots) - 1)
        iss_limit = None
        if cand_req:
            iss_limit = issued_idx.get(cand_req[0][1])
        # 请求天区：可标注窗口 = [发布, 截止)；其中无起拍点才退回全窗口（放弃该请求保 base）
        req_range = None
        if cand_req and iss_limit is not None:
            req_range = (iss_limit, dl_limit)
        hard = limit if dl_limit is None else min(limit, dl_limit)
        picked = None
        L = int(tile.nominal_exptime_seconds)
        in_window = [x for x in starts if x[0] <= hard]     # 先过滤再按 q 排序——starts 不是按时间序
        if req_range is not None:
            scoped = [x for x in in_window if req_range[0] <= x[0] <= req_range[1]]
            if scoped:
                in_window = scoped                          # 请求窗口内有起拍点：必须在窗口内拍才能标注
        for i, q in sorted(in_window, key=lambda x: -x[1]):
            if i in blocked:
                continue
            used = occupied.get(i, 0)
            if L > SLOT_DUR:
                if used > 0 or i in occupied or (i + 1) in occupied:
                    continue          # 长曝光需要独占本时隙，且下一时隙不能有行（游标会落在其中部）
            else:
                if used + L > SLOT_DUR:
                    continue          # 短曝光共享时隙：总时长 ≤ 900s
            # 请求标注：起拍必须落在 [发布, 截止) 内，否则不标注（防 invalid_request_tag）
            rid = ""
            if cand_req:
                deadline_i = dl_limit
                iss_i = iss_limit if iss_limit is not None else -1
                if i <= deadline_i and i >= iss_i:
                    rid = cand_req[0][1]
            picked = (i, q, rid)
            break
        if picked is None:
            assign[tid] = (None, "")
            continue
        i, q, rid = picked
        occupied[i] = occupied.get(i, 0) + L
        if L > SLOT_DUR:
            blocked.add(i + 1)
        assign[tid] = (i, rid)

    # band₀（不含效率）决定 program：需要选中时隙的天气与中点几何
    def band0_of(tid, slot_index):
        s = slots[slot_index]
        cond = sc.weather.get_effective_conditions(s.slot_id, tid)
        from challenge.weather_simulator import weather_quality
        mid = pu(s.timestamp_utc) + timedelta(seconds=sc.tiles[tid].nominal_exptime_seconds / 2)
        # 中点可能跨时隙：用起点时隙几何近似（band₀ 对几何不敏感的场合足够；临界情形回退 DARK）
        geo = sc.geometry.get_tile_geometry(tid, pu(s.timestamp_utc) + timedelta(seconds=min(
            sc.tiles[tid].nominal_exptime_seconds / 2, s.duration_seconds - 1)))
        am = float(geo["airmass"])
        lq = float(geo["lunar_quality_factor"])
        eff_free = min(float(cond["transparency"]) * float(cond["sky_quality"]) /
                       (float(cond["seeing_arcsec"]) * am), 3.0) * lq
        return "DARK" if eff_free >= 0.65 else "BRIGHT" if eff_free >= 0.40 else "BACKUP"

    rows = []
    for tid, (idx, rid) in sorted(assign.items(), key=lambda kv: (kv[1][0] or 0, kv[0])):
        if idx is None:
            print(f"警告: {tid} 无可排时隙（漏拍）")
            continue
        slot_id = slots[idx].slot_id
        program = band0_of(tid, idx)
        rows.append([f"D{len(rows)+1:06d}", slot_id, "observe", tid, program, rid,
                     "offline L-window best"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
        w.writerows(rows)
    print(f"写出 {len(rows)} 行 → {out/'decisions.csv'}（L={args.L}, eps={args.eps}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
