#!/usr/bin/env python3
"""官方评分器闭环修复：读 score_report 定位 invalid 行与缺失请求，移动/补行，重评分。

    python3 tools/repair_csv.py --csv DIR/decisions.csv --scenario DIR [--rounds 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
if str(KIT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402

SLOT_DUR = 900


def pu(x):
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scenario", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--qcache", default=str(KIT / ".qcache"))
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    sc = ChallengeScorer.from_files(root)
    cache = pickle.loads((Path(args.qcache) / (root.name + ".pkl")).read_bytes())
    slots = sc.slots
    slot_idx = {s.slot_id: i for i, s in enumerate(slots)}
    slot_ts = [pu(s.timestamp_utc) for s in slots]
    L_by_tile = {t: int(tile.nominal_exptime_seconds) for t, tile in sc.tiles.items()}

    csv_path = Path(args.csv)
    workdir = csv_path.parent

    for rnd in range(args.rounds):
        # 评分
        subprocess.run([sys.executable, str(KIT / "score_decisions.py"),
                        "--scenario", str(root), "--decisions", str(csv_path),
                        "--output", str(workdir / "score.json")],
                       capture_output=True, text=True, timeout=300)
        report = json.loads((workdir / "score.json").read_text())
        s = report["score"]
        pen = {k: v for k, v in s["penalties"].items() if v}
        print(f"round {rnd}: total={s['total']:.2f} base={s['base_science']:.1f} "
              f"req={s['request_reward']:.0f} pen={pen}")

        rows = list(csv.DictReader(open(csv_path)))
        changed = False

        # 1) invalid 行 → 移到该 tile 的下一个合法 cache 时隙（避开已占用）
        occupied = defaultdict(int)
        bad_slots = set()
        for a in report["actions"]:
            if a.get("outcome") not in ("completed", "wait") and a.get("slot_id"):
                bad_slots.add(a["slot_id"])
        for r in rows:
            if r["action"] == "observe" and r["slot_id"] not in bad_slots:
                occupied[r["slot_id"]] += L_by_tile[r["tile_id"]]

        new_rows = []
        seen_slots = {}
        for r in sorted(rows, key=lambda x: slot_idx.get(x["slot_id"], 0)):
            if r["action"] != "observe":
                continue
            sid = r["slot_id"]
            seen = seen_slots.get(sid, 0)
            seen_slots[sid] = seen + 1
            L = L_by_tile[r["tile_id"]]
            # 同槽累计（按出现顺序）超容 → 该行需要挪
            if occupied[sid] > SLOT_DUR and seen == 0:
                pass
            new_rows.append(r)
        # 精确处理：从 report 找 invalid 行（decision_id 对应）
        invalid_ids = {a["decision_id"] for a in report["actions"]
                       if a.get("outcome") not in ("completed", "wait")}
        # 重建行序列：跳过 invalid 行，逐行模拟游标，把 invalid 的 tile 重排
        sim_rows = [r for r in rows if r["decision_id"] not in invalid_ids]
        # 重新放置 invalid 行的 tile（保持其请求标注；找游标之后的最优合法时隙）
        kept = []
        cursor_i, offset = 0, 0
        occ_running = defaultdict(int)
        # 先按序模拟 kept 行
        fixed_rows = []
        invalid_tiles = [(next(r for r in rows if r["decision_id"] == did)) for did in sorted(invalid_ids)]
        for r in sim_rows:
            i = slot_idx.get(r["slot_id"])
            if i is None:
                continue
            L = L_by_tile[r["tile_id"]]
            if i < cursor_i:
                # stale：挪到游标之后该 tile 的第一个合法时隙
                tid = r["tile_id"]
                rid = r["request_id"]
                cands = [x[0] for x in cache.get(tid, [])
                         if x[0] >= cursor_i and occ_running[x[0]] + L <= SLOT_DUR
                         and not (L > SLOT_DUR and (x[0] + 1) in occ_running)]
                if cands:
                    ni = min(cands)
                    occ_running[ni] += L
                    fixed_rows.append((ni, {**r, "slot_id": slots[ni].slot_id}))
                    if ni >= cursor_i:
                        # 推进游标模拟
                        if ni == cursor_i:
                            offset += L
                            if offset >= SLOT_DUR:
                                cursor_i, offset = cursor_i + 1, 0
                    changed = True
                    continue
                # 放不下：保留原行（宁违规不漏拍）
                fixed_rows.append((i, r))
                continue
            kept.append((i, r))
            if i == cursor_i:
                offset += L
                if offset >= SLOT_DUR:
                    cursor_i, offset = cursor_i + 1, 0
            elif i > cursor_i:
                cursor_i, offset = i, 0
            occ_running[i] += L
        # invalid tiles 重排（在全部 kept 之后统一找空位）
        final = kept + fixed_rows
        for r in invalid_tiles:
            tid = r["tile_id"]
            L = L_by_tile[tid]
            rid = r["request_id"]
            if rid:
                src = req_meta = None
            # 请求行：窗口约束
            win = (0, len(slots) - 1)
            if rid:
                src = next((x for x in csv.DictReader(open(root / "outputs/reference/observation_requests.csv"))
                            if x["request_id"] == rid), None)
                if src:
                    win = (max((i for i, ts in enumerate(slot_ts) if ts < pu(src["issued_at_utc"])), default=0),
                           max((i for i, ts in enumerate(slot_ts) if ts < pu(src["deadline_utc"])), default=len(slots) - 1))
            occ_all = defaultdict(int)
            for i, rr in final:
                occ_all[i] += L_by_tile[rr["tile_id"]]
            cands = [x[0] for x in cache.get(tid, [])
                     if win[0] <= x[0] <= win[1] and occ_all[x[0]] + L <= SLOT_DUR
                     and not (L > SLOT_DUR and (x[0] + 1) in occ_all)]
            if cands:
                ni = min(cands)
                final.append((ni, {**r, "slot_id": slots[ni].slot_id}))
                changed = True
            else:
                final.append((slot_idx[r["slot_id"]], r))

        # 2) 缺失请求 → 补 revisit（全部完成后零成本；窗口内任意合法时隙）
        vis = defaultdict(int)
        for _i, _r in final:
            if _r["request_id"]:
                vis[(_r["request_id"], _r["tile_id"])] += 1
        req_tiles_map = {}
        for x in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv")):
            req_tiles_map.setdefault(x["request_id"], []).append(x["tile_id"])
        need = {(x["request_id"], x["tile_id"]): int(x.get("required_visits") or 1)
                for x in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv"))}
        req_csv = {x["request_id"]: x for x in csv.DictReader(open(root / "outputs/reference/observation_requests.csv"))}
        occ_all = defaultdict(int)
        for i, r in final:
            occ_all[i] += L_by_tile[r["tile_id"]]
        for rq in report["requests"]:
            rid = rq["request_id"]
            if rq["status"] == "completed":
                continue
            src = req_csv.get(rid)
            if not src:
                continue
            iss_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["issued_at_utc"])), default=0)
            dl_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["deadline_utc"])), default=len(slots) - 1)
            for tid in req_tiles_map.get(rid, []):
                if vis[(rid, tid)] >= need.get((rid, tid), 1):
                    continue
                L = L_by_tile[tid]
                cands = [x[0] for x in cache.get(tid, [])
                         if iss_i <= x[0] <= dl_i and occ_all[x[0]] + L <= SLOT_DUR
                         and not (L > SLOT_DUR and (x[0] + 1) in occ_all)]
                if cands:
                    ni = min(cands)
                    final.append((ni, {"decision_id": "NEW", "slot_id": slots[ni].slot_id, "action": "observe",
                                       "tile_id": tid, "program": "DARK", "request_id": rid,
                                       "reason": "repair revisit"}))
                    occ_all[ni] += L
                    vis[(rid, tid)] += 1
                    changed = True

        if not changed:
            print("无进一步修复动作，停止")
            break

        final.sort(key=lambda x: x[0])
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
            for n, (i, r) in enumerate(final):
                w.writerow([f"D{n+1:06d}", r["slot_id"], r["action"], r["tile_id"],
                            r["program"], r["request_id"], r["reason"]])
        print(f"  重写 {len(final)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
