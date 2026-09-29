#!/usr/bin/env python3
"""离线调度前沿：给定「每块天区何时开拍」的分配规则，直接算 base / bonus / 等待罚分 / 总分。

非 mechanics：每块只有第一次完成的曝光入账 → base = Σ tile_value × q。
等待罚分 = 0.001/s × ∫ 1[存在未完成天区此时可开拍] dt（scoring_core._has_actionable_tile）。

规则族：
  best        : 每块取全周期最好的一次
  bestK       : 每块取前 K 夜内最好的一次
  firstQ      : 每块取第一次 q ≥ Q 的时刻
  bestQ       : 每块取 q ≥ Q 中最早的一次（同 firstQ，但 Q 随进度调整）

    python3 tools/schedule_frontier.py
"""
from __future__ import annotations

import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys.path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402

CACHE = KIT / ".qcache/dev-reference.pkl"
SLOT_SECONDS = 900
IDLE = 0.001
BONUS_DARK = 0.25
BONUS_OTHER = {"BRIGHT": 0.15, "BACKUP": 0.08}
DARK_TH = 0.65
BRIGHT_TH = 0.40


def load():
    sc = ChallengeScorer.from_files(KIT / "scenarios/dev-reference")
    cache = pickle.loads(CACHE.read_bytes())
    slots = sc.slots
    night_start = {}
    for i, s in enumerate(slots):
        night_start.setdefault(s.night_id, i)
    nights = list(night_start)
    return sc, cache, slots, nights, night_start


def evaluate(sc, cache, slots, nights, night_start, assign):
    """assign: tile_id -> slot_index（None = 漏拍）"""
    tv = sc.tile_values
    base = bonus = 0.0
    required_miss = flexible = 0
    quota = int(sc.config["flexible_quota_per_region"])
    done_by_region = {}
    for tid, idx in assign.items():
        if idx is None:
            tile = sc.tiles[tid]
            if tile.scheduling_class == "REQUIRED":
                required_miss += 1
            continue
        q = dict(cache[tid])[idx]
        base += tv[tid] * q
        band = "DARK" if q >= DARK_TH else "BRIGHT" if q >= BRIGHT_TH else "BACKUP"
        bonus += tv[tid] * q * (BONUS_DARK if band == "DARK" else BONUS_OTHER[band])
        if sc.tiles[tid].scheduling_class == "FLEXIBLE":
            done_by_region[sc.tiles[tid].region_id] = done_by_region.get(sc.tiles[tid].region_id, 0) + 1
    regions = {t.region_id for t in sc.tiles.values()}
    shortfall = sum(max(0, quota - done_by_region.get(r, 0)) for r in regions)
    flexible = shortfall * float(sc.config["penalties"]["flexible_shortfall_per_tile"])

    # 等待罚分：从头到最后一块拍完为止，任一未完成天区「此刻可开拍」即计罚
    by_tile_start = {}
    for tid, rows in cache.items():
        by_tile_start[tid] = {i for i, _ in rows}
    shots = {tid: idx for tid, idx in assign.items() if idx is not None}
    last = max(shots.values()) if shots else 0
    avoidable = 0.0
    open_slots = [i for i, s in enumerate(slots) if sc.weather.get_effective_conditions(s.slot_id).get("is_observable")]
    open_set = set(open_slots)
    for i in range(0, last + 1):
        if i not in open_set:
            continue
        for tid, sidx in shots.items():
            if sidx >= i and i in by_tile_start[tid]:
                avoidable += SLOT_SECONDS * IDLE
                break
    pen = avoidable + required_miss * 1000.0 + flexible
    total = base + bonus + 3920.0 - pen
    return dict(base=base, bonus=bonus, pen=pen, avoidable=avoidable, total=total,
                missing=required_miss, shortfall=shortfall)


def main() -> int:
    sc, cache, slots, nights, night_start = load()
    tv = sc.tile_values
    print(f"{'规则':26s} {'base':>9s} {'bonus':>8s} {'pen':>9s} {'avoid':>9s} {'TOTAL':>9s} {'缺':>3s}")
    rows_by_tile = {t: dict(r) for t, r in cache.items()}

    def first_q(th):
        out = {}
        for tid, d in rows_by_tile.items():
            hit = [(i, q) for i, q in d.items() if q >= th]
            out[tid] = min(hit)[0] if hit else None
        return out

    def best_in(n_nights):
        limit = night_start[nights[min(n_nights, len(nights)) - 1]] + 44
        out = {}
        for tid, d in rows_by_tile.items():
            cand = [(q, i) for i, q in d.items() if i <= limit]
            out[tid] = max(cand)[1] if cand else None
        return out

    def best_all():
        return {tid: max((q, i) for i, q in d.items())[1] for tid, d in rows_by_tile.items()}

    cases = [("贪心首个可拍(≈基线)", first_q(0.0))]
    for q in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        cases.append((f"首个 q≥{q}", first_q(q)))
    for k in (30, 60, 90, 120, 150, 166, 180):
        cases.append((f"前{k}夜内最好", best_in(k)))
    cases.append(("全周期最好(oracle)", best_all()))

    def within_L(L):
        """每块：从首次可拍起 L 夜内取最好的一次（等待被窗口截断 → 控制罚分）"""
        out = {}
        for tid, d in rows_by_tile.items():
            if not d:
                out[tid] = None
                continue
            first = min(d)
            limit = first + L * 44
            cand = [(q, i) for i, q in d.items() if i <= limit]
            out[tid] = max(cand)[1] if cand else None
        return out

    def within_L_firstQ(L, Q):
        out = {}
        for tid, d in rows_by_tile.items():
            if not d:
                out[tid] = None
                continue
            first = min(d)
            limit = first + L * 44
            cand = [(i, q) for i, q in sorted(d.items()) if i <= limit and q >= Q]
            out[tid] = cand[0][0] if cand else None
        return out

    for L in (5, 10, 20, 40, 80, 180):
        cases.append((f"首拍起 L={L} 夜内最好", within_L(L)))
    for Q in (0.9, 1.0, 1.1):
        for L in (10, 20, 40):
            cases.append((f"L={L} 内首个 q≥{Q}", within_L_firstQ(L, Q)))
    for name, assign in cases:
        r = evaluate(sc, cache, slots, nights, night_start, assign)
        print(f"{name:26s} {r['base']:9.0f} {r['bonus']:8.0f} {r['pen']:9.0f} {r['avoidable']:9.0f} "
              f"{r['total']:9.0f} {r['missing']:3d}")
    print("\n目标: base 15469  bonus 3853  pen 2123  TOTAL 21119（请求奖励 3920 已含）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
