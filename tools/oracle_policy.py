#!/usr/bin/env python3
"""上帝视角（truth-aware）策略验证器：假设我们能拿到未来天气，最高能打多少分。

用途：判断榜一 dev-reference 21119.49 是「只看当前信息就能做到」还是「需要未来天气」。
  - 只看当前信息：我们已扫到最好 14457.95
  - 上帝视角：本脚本给出上界

策略（每块天区只有第一次完成的曝光入账，所以要挑最好的一次）：
  预计算每块天区在**每个时隙起拍**能拿到的质量 q（真实天气+真实几何，须同夜完成），
  以及「从第 i 个时隙起还能拿到的最好 q」（后缀最大值）。
  决策时：当前质量 q_now ≥ γ × 剩余最好 q → 拍；否则等。
  γ 随进度衰减（剩的机会越少越迁就），并有硬截止夜强制收工（控制等待罚分）。

    python3 tools/oracle_policy.py [scenario] [--gamma 0.92] [--deadline 0.75] [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys.path.insert(0, str(KIT))

from challenge.challenge_workflow import ChallengeWorkflow  # noqa: E402
from challenge.weather_simulator import weather_quality  # noqa: E402

CACHE_DIR = KIT / ".qcache"


def build_cache(workflow) -> dict:
    """tile_id -> list[(slot_index, q)]：在该时隙起拍能完成（同夜、逐段可观测、合法）的质量。"""
    scorer = workflow.scorer
    slots = scorer.slots
    cache = {}
    started = time.time()
    for tid, tile in scorer.tiles.items():
        rows = []
        for i, slot in enumerate(slots):
            if not scorer.weather.get_effective_conditions(slot.slot_id).get("is_observable"):
                continue
            if not scorer._tile_legal(tile, slot.timestamp_utc):
                continue
            remaining, idx, off = float(tile.nominal_exptime_seconds), i, 0
            first_night, ok, qsum = slot.night_id, True, 0.0
            while remaining > 1e-9:
                if idx >= len(slots) or slots[idx].night_id != first_night:
                    ok = False
                    break
                cur = slots[idx]
                start = cur.timestamp_utc + dt.timedelta(seconds=off)
                seconds = min(remaining, cur.duration_seconds - off)
                mid = start + dt.timedelta(seconds=seconds / 2.0)
                if not scorer._tile_legal(tile, start) or not scorer._tile_legal(tile, mid):
                    ok = False
                    break
                cond = scorer.weather.get_effective_conditions(cur.slot_id, tid)
                if not cond["is_observable"]:
                    ok = False
                    break
                geo = scorer.geometry.get_tile_geometry(tid, mid)
                am = float(geo["airmass"])
                q = weather_quality(cond, am, scorer.weather.config) * float(geo["lunar_quality_factor"])
                qsum += q * seconds
                remaining -= seconds
                idx += 1
                off = 0
            if ok and remaining <= 1e-9:
                rows.append((i, qsum / float(tile.nominal_exptime_seconds)))
        cache[tid] = rows
    print(f"  cache built in {time.time()-started:.1f}s, "
          f"entries={sum(len(v) for v in cache.values())}", flush=True)
    return cache


def suffix_max(cache, n_slots: int, horizon: int | None = None) -> dict:
    """tile -> list，arr[i] = 从时隙 i 起（可到 horizon，None=全部）能拿到的最好 q。"""
    if horizon is None:
        horizon = n_slots - 1
    best = {}
    for tid, rows in cache.items():
        arr = [0.0] * (n_slots + 1)
        cur = 0.0
        k = len(rows) - 1
        for i in range(n_slots - 1, -1, -1):
            if i <= horizon:
                while k >= 0 and rows[k][0] >= i:
                    if rows[k][0] <= horizon:
                        cur = max(cur, rows[k][1])
                    k -= 1
                arr[i] = cur
            else:
                arr[i] = 0.0
        best[tid] = arr
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--gamma", type=float, default=0.92)
    ap.add_argument("--gamma-end", type=float, default=0.70, help="进度 100% 时的 γ")
    ap.add_argument("--deadline", type=float, default=0.75, help="进度超过该比例后线性压低 γ")
    ap.add_argument("--req-urgent", type=float, default=5.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    root = Path(args.scenario).resolve()
    workflow = ChallengeWorkflow(root=root)
    scorer = workflow.scorer
    slot_index = {s.slot_id: i for i, s in enumerate(scorer.slots)}
    n_slots = len(scorer.slots)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / (root.name + ".pkl")
    if cache_path.exists() and not args.rebuild:
        cache = pickle.loads(cache_path.read_bytes())
        print(f"  cache loaded {cache_path.name}", flush=True)
    else:
        cache = build_cache(workflow)
        cache_path.write_bytes(pickle.dumps(cache))
    best_all = suffix_max(cache, n_slots, None)
    best_until_h = suffix_max(cache, n_slots, max(1, int(n_slots * args.deadline)))

    first_night_idx = 0
    progress_base = float(max(1, n_slots - 1))

    th = scorer.config["quality_thresholds"]
    quota = int(scorer.config["flexible_quota_per_region"])
    deadlines: dict = {}
    horizon_index = max(1, min(n_slots - 1, int(n_slots * args.deadline)))

    def band_of(q):
        return ("DARK" if q >= float(th["dark"]) else
                "BRIGHT" if q >= float(th["bright"]) else "BACKUP")

    def request_need(snapshot):
        """tid -> [(req_id, remaining_visits, deadline, reward_share)]，只含还没满足的访问。"""
        need = {}
        for req in snapshot.get("active_requests") or []:
            if req.get("is_complete"):
                continue
            rid = str(req["request_id"])
            dl = deadlines.get(rid)
            left_tiles = max(1, int(req.get("required_tile_count", 1)) - int(req.get("satisfied_tile_count", 0)))
            share = (float(req.get("completion_reward") or 0) + float(req.get("miss_penalty") or 0)) / left_tiles
            for m in req.get("tile_requirements") or []:
                rem = int(m.get("remaining_visits", m.get("required_visits", 0)))
                if rem <= 0:
                    continue
                need.setdefault(str(m["tile_id"]), []).append((rid, rem, dl, share))
        return need

    def provider(snapshot, _deadline):
        cursor = snapshot.get("cursor") or {}
        i = slot_index.get(str(cursor.get("slot_id")), 0)
        try:
            offset = float(cursor.get("slot_offset_seconds") or 0.0)
        except (TypeError, ValueError):
            offset = 0.0
        now = None
        try:
            now = dt.datetime.fromisoformat(str(cursor.get("timestamp_utc")).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            now = None
        if not deadlines:
            for req in snapshot.get("active_requests") or []:
                try:
                    deadlines[str(req["request_id"])] = dt.datetime.fromisoformat(
                        str(req["deadline_utc"]).replace("Z", "+00:00"))
                except Exception:  # noqa: BLE001
                    pass

        prog = min(1.0, i / progress_base)
        gamma = (args.gamma if prog < args.deadline else
                 args.gamma - (args.gamma - args.gamma_end) * (prog - args.deadline) / max(1e-9, 1 - args.deadline))

        progress = snapshot.get("progress") or {}
        completed = set(progress.get("completed_tile_ids") or [])
        flex_done = dict(progress.get("flexible_completed_by_region") or {})
        need = request_need(snapshot)

        best_choice, best_key = None, None
        for c in snapshot.get("candidate_tiles") or []:
            tid = str(c["tile_id"])
            weather = c.get("effective_weather") or {}
            if not weather.get("is_observable"):
                continue
            geo = c.get("geometry") or {}
            try:
                am = float(geo["airmass"])
                lunar = float(geo.get("lunar_quality_factor") or 1.0)
            except Exception:  # noqa: BLE001
                continue
            tile = scorer.tiles.get(tid)
            if tile is None or not scorer._can_complete_from(tile, i, offset):
                continue
            q_now = weather_quality(weather, am, scorer.weather.config) * lunar
            if q_now <= 0:
                continue
            hits = need.get(tid, [])
            urgent = bool(hits) and now is not None and any(
                h[2] is not None and (h[2] - now).total_seconds() <= args.req_urgent * 86400 for h in hits)
            is_done = tid in completed
            if is_done and not hits:
                continue                      # 非 mechanics：完成后再拍没有分
            science = 0.0 if is_done else float(c["tile_science_value"]) * q_now
            req_value = sum(h[3] for h in hits) if (urgent or not is_done) else 0.0
            terminal = 0.0
            if not is_done:
                if c["scheduling_class"] == "REQUIRED":
                    terminal = 1000.0
                elif int(flex_done.get(str(c["region_id"]), 0)) < quota:
                    terminal = 100.0
            # 质量门：对未来（截止夜之前）最好质量的达成率
            arr = best_until_h.get(tid)
            fut = 0.0
            if arr and i + 1 < len(arr):
                fut = arr[i + 1]
            if fut <= 0:
                arr_all = best_all.get(tid)
                fut = arr_all[i + 1] if arr_all and i + 1 < len(arr_all) else 0.0
            ratio = (q_now / fut) if fut > 0 else 1.0
            eligible = urgent or (ratio >= gamma)
            if not eligible:
                continue
            gain = science + req_value + terminal * min(1.0, 0.2 + prog)
            key = (1 if urgent else 0, ratio if not urgent else 0.0, gain)
            if best_key is None or key > best_key:
                best_choice, best_key = (tid, q_now, hits), key

        if best_choice is None:
            return {"action": "wait", "reason": f"oracle wait gamma={gamma:.3f}"}
        tid, q_now, hits = best_choice
        req_id = ""
        if hits:
            urgent_hits = [h for h in hits if now is not None and h[2] is not None
                           and (h[2] - now).total_seconds() <= args.req_urgent * 86400]
            pool = urgent_hits or hits
            req_id = pool[0][0]
        return {"action": "observe", "tile_id": tid, "program": band_of(q_now),
                "request_id": req_id, "reason": f"oracle gamma={gamma:.3f}"}

    started = time.time()
    result = workflow.run(provider, wallclock_seconds=86400.0)
    rep = result["score_report"]
    s = rep["score"]
    pen = sum(s["penalties"].values())
    print(f"scenario={root.name} gamma={args.gamma}->{args.gamma_end} deadline={args.deadline} "
          f"wall={time.time()-started:.1f}s term={rep['termination_reason']}")
    print(f"  base={s['base_science']:.2f} bonus={s['program_bonus']:.2f} req={s['request_reward']:.2f} "
          f"pen={pen:.2f} total={s['total']:.2f} tiles={len(rep['completion']['completed_tiles'])} "
          f"missing={rep['completion']['required_missing']}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        import csv as _csv
        with (out / "decisions.csv").open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
            for row in workflow.committed:
                w.writerow([row.decision_id, row.slot_id, row.action, row.tile_id,
                            row.program, row.request_id, row.reason])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
