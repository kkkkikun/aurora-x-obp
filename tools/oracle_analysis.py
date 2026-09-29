#!/usr/bin/env python3
"""离线分析器：量化「非 mechanics 场景里每块天区只有第一次完成的曝光入账」的择时空间。

A) 回放我们实际跑出来的曝光质量分解（airmass / 大气 / 月光 / 综合 / 程序带）
B) 每块天区在**真实天气**下的单块 oracle：合法、可在同一夜完成、逐段可观测的
   最佳起拍时刻，按最优程序带计（忽略天区间的时间冲突 → 上界）
C) 与实际得分对比，给出 base / program_bonus 的可提升空间

用法：
    python3 tools/oracle_analysis.py [scenario_dir] [--run DIR]
默认场景 agent-observer-starter-kit/scenarios/dev-reference，
默认回放 agent-observer-starter-kit/v64_dev-reference。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys.path.insert(0, str(KIT))

from challenge.scoring_core import ChallengeScorer  # noqa: E402
from challenge.weather_simulator import weather_quality  # noqa: E402


def pct(values, *ps):
    s = sorted(values)
    return [s[min(len(s) - 1, int(len(s) * p))] for p in ps]


def analyze_run(run_dir: Path) -> None:
    report = json.loads((run_dir / "score_report.json").read_text())
    obs = [a for a in report["actions"] if a["action"] == "observe"]
    atm, lun, com, air = [], [], [], []
    bands: dict[str, int] = {}
    for a in obs:
        for seg in a.get("segments", []):
            atm.append(seg["atmospheric_quality"])
            lun.append(seg["lunar_quality_factor"])
            com.append(seg["combined_quality"])
            air.append(seg["airmass"])
            bands[seg["quality_band"]] = bands.get(seg["quality_band"], 0) + 1
    print(f"== A) 实际跑法分解（{run_dir.name}，{len(obs)} 次曝光）==")
    for name, vals in (("airmass", air), ("atmospheric", atm), ("lunar", lun), ("combined", com)):
        if not vals:
            continue
        p10, p50, p90 = pct(vals, 0.1, 0.5, 0.9)
        print(f"  {name:12s} mean={statistics.fmean(vals):.3f} p10={p10:.3f} p50={p50:.3f} p90={p90:.3f} max={max(vals):.3f}")
    print("  band 分布:", bands)
    s = report["score"]
    done = report["completion"]["completed_tiles"]
    n = max(1, len(done))
    print(f"  base={s['base_science']:.2f}（{len(done)} 块 → {s['base_science']/n:.2f}/块） "
          f"bonus={s['program_bonus']:.2f}（占 base {s['program_bonus']/max(1e-9,s['base_science']):.4f}） "
          f"penalty={sum(v for v in s['penalties'].values()):.2f} total={s['total']:.2f}")


def start_quality(scorer: ChallengeScorer, tile_id: str, start_index: int, offset: int):
    """从 (slot_index, offset) 起拍一次曝光，返回 (按最优程序的得分, 加权综合质量, 各段带)。"""
    tile = scorer.tiles[tile_id]
    slots = scorer.slots
    if start_index >= len(slots):
        return None
    first_night = slots[start_index].night_id
    remaining = float(tile.nominal_exptime_seconds)
    offset_seconds = offset
    segs = []
    quality_sum = 0.0
    idx = start_index
    while remaining > 1e-9:
        if idx >= len(slots) or slots[idx].night_id != first_night:
            return None
        slot = slots[idx]
        start = slot.timestamp_utc + dt.timedelta(seconds=offset_seconds)
        seconds = min(remaining, slot.duration_seconds - offset_seconds)
        mid = start + dt.timedelta(seconds=seconds / 2.0)
        if not scorer._tile_legal(tile, start) or not scorer._tile_legal(tile, mid):
            return None
        cond = scorer.weather.get_effective_conditions(slot.slot_id, tile_id)
        if not cond["is_observable"]:
            return None
        geo = scorer.geometry.get_tile_geometry(tile_id, mid)
        airmass = float(geo["airmass"])
        atm_q = weather_quality(cond, airmass, scorer.weather.config)
        band_q = weather_quality(cond, airmass, scorer.weather.config, include_efficiency=False)
        lunar = float(geo["lunar_quality_factor"])
        combined = atm_q * lunar
        band = scorer._quality_band((band_q if scorer.mechanics else atm_q) * lunar)
        base = scorer.tile_values[tile_id] * seconds / tile.nominal_exptime_seconds * combined
        segs.append((base, band, seconds, airmass, combined, atm_q, lunar))
        quality_sum += combined * seconds
        remaining -= seconds
        idx += 1
        offset_seconds = 0
    if remaining > 1e-9:
        return None
    bonuses = scorer.config["program_bonus"]
    best_program, best_score = None, -1.0
    for program in bonuses:
        score = sum(b * (1.0 + float(bonuses[program]) if band == program else 1.0) for b, band, *_ in segs)
        if score > best_score:
            best_program, best_score = program, score
    weighted_quality = quality_sum / float(scorer.tiles[tile_id].nominal_exptime_seconds)
    return best_score, weighted_quality, best_program, segs


def oracle(root: Path) -> None:
    scorer = ChallengeScorer.from_files(root)
    print(f"\n== B/C) 单块 oracle（{root.name}，mechanics={scorer.mechanics}，"
          f"coverage={scorer.config.get('coverage_bonus_weight')}）==")
    slots = scorer.slots
    observable = [
        i for i, s in enumerate(slots)
        if bool(scorer.weather.get_effective_conditions(s.slot_id).get("is_observable"))
    ]
    print(f"  slots={len(slots)} observable={len(observable)} ({len(observable)/max(1,len(slots)):.1%})")

    results = {}
    opportunities: dict[str, tuple[int, list[float]]] = {}
    for tile_id in scorer.tiles:
        best = None
        n_opt = 0
        quals = []
        for i in observable:
            got = start_quality(scorer, tile_id, i, 0)
            if got is None:
                continue
            n_opt += 1
            quals.append(got[1])
            if best is None or got[0] > best[0]:
                best = (got[0], got[1], got[2], i, got[3])
        if best:
            results[tile_id] = best
            opportunities[tile_id] = (n_opt, quals)

    print(f"  有合法曝光的天区: {len(results)}/{len(scorer.tiles)}")
    if not results:
        return
    scores = [v[0] for v in results.values()]
    quals = [v[1] for v in results.values()]
    base_like = sum(v[1] * scorer.tile_values[t] for t, v in results.items())
    bands: dict[str, int] = {}
    for v in results.values():
        bands[v[2]] = bands.get(v[2], 0) + 1
    print(f"  单块最高综合质量: mean={statistics.fmean(quals):.3f} min={min(quals):.3f} max={max(quals):.3f}")
    print(f"  最优程序带分布: {bands}")
    print(f"  Σ(质量×tile_value) ≈ {base_like:.2f}   （我们实际 base 见 A）")
    counts = sorted(v[0] for v in opportunities.values())
    if counts:
        print(f"  每块可起拍次数: min={counts[0]} p50={counts[len(counts)//2]} max={counts[-1]}（合计 {sum(counts)}）")
    # 只用「前 k 好」机会的天花板：模拟不能预知天气时按阈值等待的收益
    for k in (1, 3, 5, 10, 10**9):
        vals = [sorted(v[1], reverse=True)[min(k, len(v[1])) - 1] for v in opportunities.values() if v[1]]
        if not vals:
            continue
        approx = sum(q * scorer.tile_values[t] for (t, v), q in zip(
            [(t, v) for t, v in opportunities.items() if v[1]], vals))
        print(f"  取每块前 {k if k < 10**9 else '∞'} 好机会的质量均值={statistics.fmean(vals):.3f} → base 上限≈{approx:.2f}")
    print(f"  忽略冲突的 base+bonus 上限 ≈ {sum(scores):.2f}")
    print(f"  榜一实测 base+bonus = {15469.61 + 3853.29:.2f}；我们 = {7878.59 + 1313.26:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--run", default=str(KIT / "v64_dev-reference"))
    args = ap.parse_args()
    run = Path(args.run)
    if (run / "score_report.json").exists():
        analyze_run(run)
    root = Path(args.scenario)
    if not (root / "config").exists():
        print(f"场景不存在: {root}")
        return 1
    oracle(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
