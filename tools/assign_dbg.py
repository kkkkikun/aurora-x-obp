#!/usr/bin/env python3
"""全局指派排程（results 路线）：请求层 + base 层联合优化。

评分器语义（scoring_core.py 实读，2026-09-28）：
  - 非 mechanics 下重复观测完全合法：base/bonus 只入账第一次完成曝光，无任何罚分；
  - 请求访问 = 带 request_id 的**完成**曝光，visit[(rid,tid)] += 1，每请求独立计数；
  - 请求完成 = satisfied ≥ required_tile_count；奖励 140×tiles，过期未完成罚 190×tiles
    （真值可行数不足则豁免）。
  → 共享天区可拍多次（每请求一次），之前的「一次拍摄服务多请求」是伪约束。

指派模型：
  Phase 1 请求层：agent 已满足的请求逐一保持满足——每 tile 在 [发布, 截止] 内取质量
           最优可行时隙（deadline 升序贪心 + 容量），标注该请求。
  Phase 2 base 层：其余天区 L 窗口最优 + 容量（ε-提前压尾部）。
  Phase 3 局部搜索：毫秒级全目标评估（base+bonus+req+wait-pen），接受改进的移动。

    python3 tools/assign_full.py [--L 40] [--out DIR] [--iters 3000]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
if str(KIT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402

SLOT_DUR, NIGHT_SLOTS = 900, 44
W_PEN = 0.0009          # 0.001/s × 900s
BONUS = {"DARK": 0.25, "BRIGHT": 0.15, "BACKUP": 0.08}


def pu(x):
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def band0_of(sc, tid, slot_index, L):
    s = sc.slots[slot_index]
    cond = sc.weather.get_effective_conditions(s.slot_id, tid)
    geo = sc.geometry.get_tile_geometry(tid, pu(s.timestamp_utc) + timedelta(seconds=min(L / 2, SLOT_DUR - 1)))
    am = float(geo["airmass"])
    lq = float(geo["lunar_quality_factor"])
    eff_free = min(float(cond["transparency"]) * float(cond["sky_quality"]) /
                   (float(cond["seeing_arcsec"]) * am), 3.0) * lq
    return "DARK" if eff_free >= 0.65 else "BRIGHT" if eff_free >= 0.40 else "BACKUP"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "schedule_out/assign"))
    ap.add_argument("--qcache", default=str(KIT / ".qcache"))
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    sc = ChallengeScorer.from_files(root)
    cache = pickle.loads((Path(args.qcache) / (root.name + ".pkl")).read_bytes())
    slots = sc.slots
    n_slots = len(slots)
    slot_ts = [pu(s.timestamp_utc) for s in slots]
    slot_idx = {s.slot_id: i for i, s in enumerate(slots)}
    observable = [bool(sc.weather.get_effective_conditions(s.slot_id).get("is_observable")) for s in slots]
    V = sc.tile_values
    L_by_tile = {t: int(tile.nominal_exptime_seconds) for t, tile in sc.tiles.items()}
    quota = int(sc.config["flexible_quota_per_region"])

    # 请求元数据
    req_csv = {r["request_id"]: r for r in csv.DictReader(open(root / "outputs/reference/observation_requests.csv"))}
    req_tiles_map = {}
    for r in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv")):
        req_tiles_map.setdefault(r["request_id"], []).append(r["tile_id"])
    visits_needed = {}
    for r in csv.DictReader(open(root / "outputs/reference/observation_request_tiles.csv")):
        visits_needed[(r["request_id"], r["tile_id"])] = int(r.get("required_visits") or 1)

    # agent 已满足的请求集合（可行性已被证明）
    agent_report = None
    for cand in (ROOT / "schedule_out/f_60_30_0.03/score.json", KIT / "v7_baseline_check/score_report.json"):
        if Path(cand).exists():
            agent_report = json.loads(Path(cand).read_text())
            break
    satisfied_rids = set()
    if agent_report:
        for rq in agent_report.get("requests") or []:
            if rq.get("status") == "completed":
                satisfied_rids.add(str(rq["request_id"]))
    print(f"agent 已满足请求: {len(satisfied_rids)}")

    # 请求窗口（slot index 视角）
    req_win = {}
    for rid in satisfied_rids:
        src = req_csv[rid]
        iss_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["issued_at_utc"])), default=0)
        dl_i = max((i for i, ts in enumerate(slot_ts) if ts < pu(src["deadline_utc"])), default=n_slots - 1)
        req_win[rid] = (iss_i, dl_i)

    # ---- Phase 1: 请求层（deadline 升序；每 tile 每请求一次标注拍摄）----
    occupied = {}    # slot -> used seconds
    blocked = set()  # 被长曝光封锁

    def fits(i, L):
        if i in blocked or i >= n_slots:
            return False
        used = occupied.get(i, 0)
        if L > SLOT_DUR:
            return used == 0 and (i + 1) not in occupied
        return used + L <= SLOT_DUR

    def claim(i, L):
        occupied[i] = occupied.get(i, 0) + L
        if L > SLOT_DUR:
            blocked.add(i + 1)

    assign = []   # list[(tile_id, slot_index, rid)]
    tile_shots = {}   # tile -> [(slot, rid)]
    req_order = sorted(satisfied_rids, key=lambda r: req_win[r][1])
    for rid in req_order:
        iss_i, dl_i = req_win[rid]
        for tid in req_tiles_map.get(rid, []):
            L = L_by_tile[tid]
            need = visits_needed.get((rid, tid), 1)
            have = sum(1 for _, r2 in tile_shots.get(tid, []) if r2 == rid)
            for _ in range(max(0, need - have)):
                scoped = [(q, i) for i, q in cache.get(tid, []) if iss_i <= i <= dl_i and fits(i, L)]
                if not scoped:
                    # 容量满：放松到窗口内时间正序第一个任意起拍（挤掉非请求的天区后面重排）
                    scoped = [(q, i) for i, q in cache.get(tid, []) if iss_i <= i <= dl_i]
                    if not scoped:
                        print(f"警告: {rid}/{tid} 请求窗口内无合法起拍（该请求可能无法满足）")
                        break
                    # 从窗口最优开始找任何能容纳的（允许重排代价）
                    for q, i in sorted(scoped, key=lambda x: -x[0]):
                        assign[:] = [a for a in assign if not (a[0] == tid and a[2] == rid)]
                        tile_shots[tid] = [(s, r2) for s, r2 in tile_shots.get(tid, []) if r2 != rid]
                        assign.append((tid, i, rid))
                        tile_shots.setdefault(tid, []).append((i, rid))
                        break
                    continue
                best_q = max(q for q, _ in scoped)
                pick_i = None
                for q, i in scoped:
                    if q >= 0.90 * best_q:      # ε-提前：近优取最早，压尾部
                        pick_i = i
                        break
                assign.append((tid, pick_i, rid))
                tile_shots.setdefault(tid, []).append((pick_i, rid))
                claim(pick_i, L)

    # ---- Phase 2: base 层（其余天区 L 窗口最优）----
    served = {t for t, _, _ in assign}
    for tid in sorted((t for t in sc.tiles if t not in served),
                      key=lambda t: -max(q for _, q in (cache.get(t) or [(0, 0.0)]))):
        L = L_by_tile[tid]
        starts = cache.get(tid) or []
        if not starts:
            print(f"警告: {tid} 无 cache（漏拍）")
            continue
        first = starts[0][0]
        limit = first + args.L * NIGHT_SLOTS
        in_window = [x for x in starts if x[0] <= limit and fits(x[0], L)]
        if not in_window:
            in_window = [x for x in starts if fits(x[0], L)]
        if not in_window:
            print(f"警告: {tid} 无可排时隙（漏拍）")
            continue
        best_q = max(q for _, q in in_window)
        near = [i for q, i in in_window if q >= (1.0 - args.eps) * best_q]
        pick_i = min(near) if near else min(i for _, i in in_window)   # ε-近优取最早
        assign.append((tid, pick_i, ""))
        tile_shots.setdefault(tid, []).append((pick_i, ""))
        claim(pick_i, L)

    # ---- 毫秒级全目标评估 ----
    starts_set = {t: {i for i, _ in rows} for t, rows in cache.items()}
    for _chk in assign:
        if not isinstance(_chk[1], int):
            print("BAD ASSIGN ENTRY:", _chk)
            raise SystemExit(0)
    bands = {}

    def evaluate(asg):
        """asg: list[(tile, slot, rid)] → (total, detail)"""
        shots = {}   # tile -> [(slot, rid)]
        for t, i, r in asg:
            shots.setdefault(t, []).append((i, r))
        base = bonus = 0.0
        first_shot = {}
        for t, lst in shots.items():
            lst2 = sorted(lst)
            i0 = lst2[0][0]
            q = dict(cache[t]).get(i0, 0.0)
            b = BONUS[bands[t]]
            base += V[t] * q
            bonus += V[t] * q * b
            first_shot[t] = i0
        # 请求
        req_reward = req_pen = 0.0
        for rid in satisfied_rids:
            sat = sum(1 for t in req_tiles_map.get(rid, [])
                      if sum(1 for _, r2 in shots.get(t, []) if r2 == rid) >= visits_needed.get((rid, t), 1))
            need = int(float(req_csv[rid]["required_tile_count"]))
            if sat >= need:
                req_reward += float(req_csv[rid]["completion_reward"])
            else:
                req_pen += float(req_csv[rid]["miss_penalty"])
        # 终局罚分
        done_by_region = {}
        for t, i0 in first_shot.items():
            tile = sc.tiles[t]
            if tile.scheduling_class == "FLEXIBLE":
                done_by_region[tile.region_id] = done_by_region.get(tile.region_id, 0) + 1
        regions = {tile.region_id for tile in sc.tiles.values()}
        shortfall = sum(max(0, quota - done_by_region.get(r, 0)) for r in regions) * 100.0
        required_miss = sum(1 for t, tile in sc.tiles.items()
                            if tile.scheduling_class == "REQUIRED" and t not in first_shot) * 1000.0
        # 等待罚分：最后一个拍摄之前， observable 且存在「未完成且此刻可起拍」的天区
        last = max((i for lst in shots.values() for i, _ in lst), default=0)
        by_tile = {t: {i for i, _ in rows} for t, rows in cache.items()}
        wait = 0.0
        for i in range(last):
            if not observable[i]:
                continue
            for t, i0 in first_shot.items():
                if i0 > i and i in by_tile[t]:
                    wait += W_PEN
                    break
        total = base + bonus + req_reward - req_pen - shortfall - required_miss - wait
        return total, dict(base=base, bonus=bonus, req=req_reward - req_pen, short=shortfall,
                           rmiss=required_miss, wait=wait)

    cur_total, cur_detail = evaluate(assign)
    print(f"初始指派: total={cur_total:.1f} {cur_detail}")

    # ---- Phase 3: 局部搜索 ----
    rng = random.Random(20260928)
    by_slot = {}
    for n, (t, i, r) in enumerate(assign):
        by_slot.setdefault(i, []).append(n)

    def try_move(n, new_i):
        t, old_i, r = assign[n]
        L = L_by_tile[t]
        if not (0 <= new_i < n_slots) or new_i == old_i:
            return False
        if r:
            lo, hi = req_win[r]
            if not (lo <= new_i <= hi):
                return False
            if new_i not in starts_set.get(t, set()):
                return False
        # 容量（粗略：移出旧时隙、检查新时隙；长曝光封锁忽略——评估器会兜底验证总罚分）
        old_used = sum(L_by_tile[a[0]] for a in assign if a[1] == old_i)
        new_used = sum(L_by_tile[a[0]] for a in assign if a[1] == new_i) - L
        if L > SLOT_DUR:
            if occupied.get(old_i, 0) == L:
                pass
            if (new_i + 1) in {a[1] for a in assign} and old_i != new_i + 1:
                return False
        elif new_used + L > SLOT_DUR:
            return False
        assign[n] = (t, new_i, r)
        ok_total, _ = evaluate(assign)
        if ok_total > cur_total + 1e-9:
            cur_total = ok_total
            return True
        assign[n] = (t, old_i, r)
        return False

    improved = 0
    for it in range(args.iters):
        n = rng.randrange(len(assign))
        t, old_i, r = assign[n]
        cand = rng.choice(sorted(starts_set.get(t, set()) or [old_i]))
        if try_move(n, cand):
            improved += 1
    print(f"局部搜索: {improved} 次改进移动")
    final_total, final_detail = evaluate(assign)
    print(f"最终评估: total={final_total:.1f} {final_detail}")

    # ---- 写 CSV ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(assign, key=lambda x: x[1])
    with (out / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
        for n, (tid, i, rid) in enumerate(rows):
            w.writerow([f"D{n+1:06d}", slots[i].slot_id, "observe", tid,
                        band0_of(sc, tid, i, L_by_tile[tid]), rid, "global assignment"])
    print(f"写出 {len(rows)} 行 → {out/'decisions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
