"""你的策略 · v6.11 — v6.8 + 中天爬升段加成（仅非 mechanics 场景启用，唯一改动，未实测·待三场景回归）。

每次决策，平台把「现在可以观测的候选」按公开评分公式排好序交给你（第 1 个是估计收益最高的）。
你返回要观测的候选（可附 reason），或 None 表示本时隙等待。

=== 三层策略（按场景类型自动切换）===
  1. 未观测 REQUIRED 硬优先（所有场景）：每块一次曝光即可消除 -1000 终局罚分风险。
  2. 正式赛型（coverage_bonus_weight > 0）：覆盖感知边际排序——
     marginal = estimated_total_gain + W×( base_so_far×ΔE + est_base×E_after )，
     ΔE 为该候选对 Jain 覆盖均匀度的边际贡献（正式赛 demand>100% 拍不完，
     均匀分布值 ~1/5 总分）。叠加异常调整（见 3）与 nova 确认调度。
  3. 宽松场景（W = 0）：质量择时——贪心总在天区刚升过 30°（airmass≈2.0）时开拍，
     而 airmass 改善集中在升起后头几段；等到 高度≥45° 或 时角≥−30° 再拍（质量 ×1.2-1.4），
     等待罚金仅 0.9/时隙。**松弛门**：剩余夜×40时隙 ÷ 剩余天区 < 12 时禁止择时
     （180 夜场景松弛 ~114 → +457；14 夜 ~8 / 7 夜 ~4 → 强制回到默认排序）。
     带 request_id 的候选不推迟（截止期风险）。

=== 异常检测（替换管线检测器 _classify；报告仍由协议发出）===
  分层归一化：读数比 = tile_last_finished / 承诺时公开基线，÷ 当夜全场中位数
  （全球事件如漏报的 cold_wave 会整夜压低所有读数，绝对阈值必炸）。
  - 故障：分区滚动读数中位数 / 全场 ≤0.70 且跨 ≥2 夜（真故障乘数 0.4-0.7 → rel≈0.45-0.65）。
  - 标签：≥2 次有效读数跨 ≥2 夜；nova rel≥1.18（首读须 ≥1.00）、reddening rel∈[0.30,0.84]
    （首读须在带内——标签是永久属性）；分区塌陷期与已污染（已知故障区）读数不计入。
  - nova 确认调度：首读 rel≥1.30（实测全场零误报）→ 候选边际 +120 促使复拍；
    已上报 nova 的天区估值 ×1.5（重复观测合法、按最高分入账）。
  - 整个 _classify 套兜底：任何异常只丢一次检测机会，绝不终止运行（agent_error 教训）。
  - v6.7 故障报告等待期规避：平台 fault_status 只在夜初发布且有 1 天响应延迟
    （challenge_workflow._fault_status / decision_snapshot），平台自带的
    filter_fault_scope 要到公布后才生效；而本策略的故障报告在分区 4 读中位数
    ≤0.70 时即发出。这个空档里塌陷分区读数只实现 ~0.45-0.65×（v64r_compano 实测
    10-21 报告后至 10-23 滤除生效前还有 8 个时隙打在 R02 上、单读 ~30-78 vs 健康
    ~120-160）。改动：报告发出后、平台 fault_status（fault 或 normal=误报答复）
    到达前，选择时跳过触发分区的候选（仍有其他分区候选时）；无候选可换则照旧。
    无故障事件场景（comp-like/-b 无 fault 事件）此路径永不触发，零影响。
  - v6.8 REQUIRED 末班车游标预留：预览层只放行「现在起拍能在窗口内完成」的候选
    （scoring_preview._known_window_can_finish），而贪心/边际排序从不主动等待，
    曝光动辄跨时隙 → 当一块未观测 REQUIRED 的当夜唯一可完成起点恰好落在某个
    时隙边界 t* 时，若 t* 前的曝光跨过它，t* 就被跳过、该块永远漏拍（comp-like
    实测 T00817：窗口 12:15-12:45、曝光 1350s，只有 12:15:00 整点起拍能完成；
    基线在 12:00 拍了 1200s 的 T00659 → 下个决策 12:20 → −1000 required_miss；
    另查实它在最后 4 夜里前 3 夜 12:15 高度 <30°，本就无解，唯独末夜可捕获）。
    改动：用 night_start.tile_windows（每夜发布）缓存当夜 REQUIRED 窗口，用
    weekly.tile_windows（每 7 夜发布）缓存已知未来窗口；当「某未观测 REQUIRED
    当夜可完成起点集合 == {下一时隙边界} 且已知未来无窗口，且当前已报价的未观测
    REQUIRED 都可延期」时，把本时隙的选择池限制为曝光 ≤ 剩余时隙的候选（贴合
    曝光，时隙不空转，下个决策恰好落在/不越过 t*）；无贴合候选才整段等待。
    硬优先分支同时给「错过今晚就没机会」的 REQUIRED 最高优先。触发面极窄：
    comp-like-b / comp-ano 的 REQUIRED 全部按时捕获（基线 required_miss=0），
    预计全程 0-8 次触发且每次只是把同夜的捕获时点挪到 t*。
    本地重放验证（基线决策流上静态评估 + 分叉点端到端模拟，非完整评测）：
    comp-like 触发 2 次（夜 27 恰逢天气整夜阻塞=零成本、末夜 12:05 制胜），
    端到端 +420.0（+1000 罚分消除 + T00817 入账 118.5 − T00659 被挤占的时隙）；
    comp-like-b 触发 3 次均为同夜重排（基线本就捕获、required_miss 保持 0）；
    comp-ano 触发 0 次，v6.7 行为完全不受影响。
  - v6.11 中天爬升段加成（boost-only，仅非 mechanics 场景启用）：预览用决策瞬间的
    airmass 评估整段曝光，权威计分按段取中点几何 → 起拍于中天前、曝光跨入中天的
    候选实现质量高于预览估计（comp-like 实测爬升段中位 realized/est = 1.024；
    airmass_exponent=1.0）。v6.9 合并修正（boost+discount）A/B 净 −62.6 符号随种子
    翻转；v6.10 boost-only 三场景评测 588514.2（−190.1）——分解归因：comp-like/-b
    精确兑现 +86.0/+243.1（harness 已校准至逐场景分毫不差复现官方接受分），损失
    全部来自 comp-ano ≈ −519（机制推断：boost 扰动观测流 → 故障 4 读累积时序
    漂移 → 检出/修复推迟；mechanics 场景的 realized/est 混入隐藏效率因子，几何-only
    修正前提被破坏）。本版改动：harvest 扩展收 night_start 行 best_time_utc/
    best_airmass（全类别）+ _rise_factor（割线外推曝光均值，factor=(am_now/am_avg)^k
    夹 [1.0,1.15]，只缩放科学分），并由公开 score_config 的异常段（anomaly_tags/
    reporting）做 mechanics 门控——mechanics 场景（comp-ano、真实正式赛同构）禁用
    加成、行为精确等同 v6.8；非 mechanics（comp-like/-b）保留已验证的 +86.0/+243.1。
    任何异常 factor=1 退化为 v6.8 行为。

=== 实测记录（v6.4 定版，2026-09-26，全部零误报零回归）===
  comp-like   (30夜1600天区 覆盖0.35):        203716.8（真基线 201930.7，+1786）
  comp-like-b (同上 seed 777):                201520.4（真基线 198567.3，+2953）
  comp-ano    (同上 + 异常全开 seed 999):      182351.0（真基线 182228.3，+122.7；故障 1/1）
  finals-preview (7晚异常全开):                8243.6（基线 8214.26，+29；T00054 reddening +100；故障 1/1）
  dev-reference (180夜 宽松):                  12744.6（基线 12287.5，+457；质量择时生效）
  dev-fortnight (14夜):                        7693.6（基线 7328.5，+365；末段安全择时）
  demo-week (7夜):                             5909.6（基线 5909.1，+0.5；松弛门正确阻断）
  已知局限: 30 夜场景每块天区平均 0.73 次读数，标签转化靠运气；reddening 与漂移灰区
  重叠只做被动检测；正式赛（W>0）的质量择时（与覆盖权衡）尚未实现。
"""
import os
import statistics
import sys
from datetime import date, datetime, timedelta

import anomaly_detection as _ad

try:
    import llm_planner as _lp
except Exception:  # noqa: BLE001
    _lp = None


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return float(default)

# ---------------- 异常检测状态（模块级，跨决策持久） ----------------
_S = {
    "tile_region": {},    # tile_id -> region_id（从候选学习）
    "night_reads": {},    # night -> [ratio]
    "tile_reads": {},     # tile_id -> [(ratio, night, realized)]
    "region_reads": {},   # region -> [(ratio, night)]
    "reported_tags": set(),   # (tile_id, kind)
    "reported_nova": set(),
    "reported_red": set(),
    "fault_cleared_for": None,
    "fault_regions_known": set(),
    "pending_fault_region": None,  # v6.7: 已发故障报告、等平台 fault_status 公布的分区（期间规避）
    "pending_fault_since": None,   # v6.7: 报告发出时刻（超过 3 天未获答复则自行解除）
    "slot_seconds": None,          # v6.8: 时隙长度（initial_publication.calendar）
    "night_windows_req": {},       # v6.8: 当夜 REQUIRED 窗口 tile_id -> [{ws,we,ex}]
    "night_windows_for": None,     # v6.8: 上述缓存所属 night_id
    "llm_night_for": None,         # LLM 夜计划已规划到的 night_id（每夜 ≤1 次）
    "future_windows": {},          # v6.8: weekly 已知未来窗口 tile_id -> [(night_id, ws, we, ex)]
    "night_windows_meta": {},      # v6.11: 当夜全类别窗口 tile_id -> [(ws, we, best_time, best_am)]
    "v": {},              # tile_id -> tile_science_value（首拍时学习）
    "banked": {},         # tile_id -> 已入账最高分
}
NOVA_REL_MIN = 1.18
RED_REL_MAX = 0.84
RED_REL_MIN = 0.30
NOVA_ANTI_MAX = 0.92    # nova 天区不允许出现明显偏低的读数
RED_ANTI_MIN = 1.10     # reddening 天区不允许出现明显偏高的读数
TAG_MIN_READS = 2
FAULT_REL = 0.70
FAULT_MIN_REGION_READS = 4
CONFIRM_BONUS = 90.0    # 一次确认读数的期望奖励（P(确认)×100 的近似）
NOVA_FACTOR = 1.5
RED_FACTOR = 0.8
_BONUS = {"DARK": 0.25, "BRIGHT": 0.15, "BACKUP": 0.08}
_GATE_MODE = "w"


def _night_ref(night):
    """当夜全场基准：当夜 ≥5 条读数用其中位数；否则用历史各夜中位数的中位数。"""
    cur = _S["night_reads"].get(night) or []
    if len(cur) >= 5:
        return statistics.median(cur)
    meds = [statistics.median(v) for n, v in _S["night_reads"].items()
            if n != night and len(v) >= 5]
    return statistics.median(meds) if meds else None


def _parse_utc(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _classify_hierarchical(self, tile_id, ratio, snapshot):
    """替换管线检测器的分类：分层归一化 + 标签/故障判定，返回要附带的报告。

    注意：process_snapshot 不在 try/except 内，这里抛异常 = agent_error = 整场终止，
    所以整个函数体套了兜底保护：任何意外只丢一次检测机会，绝不终止运行。"""
    try:
        return _classify_impl(self, tile_id, ratio, snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"detector fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return []


def _sync_fault_regions(self):
    """故障状态公布后：把作用分区的全部历史读数追溯标记为污染（标签证据剔除）。"""
    scope = getattr(self, "_fault_scope", None)
    if not scope:
        return
    if scope.get("spatial_scope_type") != "REGION_SET":
        return
    regions = set((scope.get("spatial_scope_payload") or {}).get("region_ids") or [])
    if regions <= _S["fault_regions_known"]:
        return
    _S["fault_regions_known"] |= regions
    for tid, reads in _S["tile_reads"].items():
        if _S["tile_region"].get(tid) in regions:
            for entry in reads:
                entry[3] = True


def _classify_impl(self, tile_id, ratio, snapshot):
    reports = []
    night = (snapshot.get("cursor") or {}).get("night_id", "?")
    region = _S["tile_region"].get(tile_id)
    _S["night_reads"].setdefault(night, []).append(ratio)
    _S["tile_reads"].setdefault(tile_id, []).append([ratio, night, 0.0, False])
    if region:
        _S["region_reads"].setdefault(region, []).append((ratio, night))
    _sync_fault_regions(self)

    ref = _night_ref(night)
    if ref is None or ref <= 0.05:
        return reports
    rel = ratio / ref

    # ---- 标签判定（相对读数需跨 ≥2 夜、≥TAG_MIN_READS 次、无反向读数）----
    reads = _S["tile_reads"][tile_id]
    # 分区塌陷态（故障未公布时的窗口期）：滚动最近 4 条分区读数的中位数明显低于全场基准，
    # 该区所有天区的读数一律不作标签证据（防 T00011 类故障区误报）。
    region = _S["tile_region"].get(tile_id)
    region_depressed = False
    if region:
        recent = [x for x, _ in _S["region_reads"].get(region, [])][-4:]
        nref_now = _night_ref(night)
        if len(recent) >= 2 and nref_now and statistics.median(recent) / nref_now <= 0.85:
            region_depressed = True
    rels = []
    if not region_depressed:
        for r, n, _, bad in reads:
            if bad:
                continue
            nref = _night_ref(n)
            if nref:
                rels.append((r / nref, n))
    # 永久属性门：标签从第一次有效读数就该在带内。首读正常的天区永远不可能是标签天区
    #（T00011 教训：故障期读数骤降的普通天区）。
    first_ok_red = bool(rels) and rels[0][0] <= RED_REL_MAX
    first_ok_nova = bool(rels) and rels[0][0] >= 1.00
    hi = sum(1 for x, _ in rels if x >= NOVA_REL_MIN)
    lo = sum(1 for x, _ in rels if RED_REL_MIN <= x <= RED_REL_MAX)
    nights_hi = len({n for x, n in rels if x >= NOVA_REL_MIN})
    nights_lo = len({n for x, n in rels if RED_REL_MIN <= x <= RED_REL_MAX})
    if (tile_id, "NOVA") not in _S["reported_tags"] and hi >= TAG_MIN_READS and nights_hi >= 2 \
            and first_ok_nova and all(x[0] >= NOVA_ANTI_MAX for x in rels):
        _S["reported_tags"].add((tile_id, "NOVA"))
        _S["reported_nova"].add(tile_id)
        reports.append({"kind": "NOVA", "tile_id": tile_id})
    elif (tile_id, "Reddening") not in _S["reported_tags"] and lo >= TAG_MIN_READS and nights_lo >= 2 \
            and first_ok_red and all(x[0] <= RED_ANTI_MIN for x in rels):
        _S["reported_tags"].add((tile_id, "Reddening"))
        _S["reported_red"].add(tile_id)
        reports.append({"kind": "Reddening", "tile_id": tile_id})

    # ---- 故障判定（分区最近读数中位数 ÷ 夜基准）----
    now = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
    repair_until = getattr(self, "_fault_repair_until", None)
    if region and repair_until is not None and now is not None and now >= repair_until \
            and _S["fault_cleared_for"] != region:
        _S["region_reads"][region] = []          # 修复完成后重置该区基线
        _S["fault_cleared_for"] = region
    if region and not getattr(self, "_fault_pending", False):
        recent = [r for r, n in _S["region_reads"].get(region, [])][-FAULT_MIN_REGION_READS:]
        if len(recent) >= FAULT_MIN_REGION_READS:
            rmed = statistics.median(recent)
            nref = _night_ref(night)
            if nref and rmed / nref <= FAULT_REL:
                repair_active = repair_until is not None and (now is None or now < repair_until)
                if not repair_active:
                    self._fault_pending = True
                    _S["pending_fault_region"] = region   # v6.7: 等待期规避在该分区生效
                    _S["pending_fault_since"] = now
                    reports.append({"kind": "Instrument_Failure"})

    # ---- 兼容：维护原检测器的 _tag_reads（top_suspect 使用）----
    band = None
    if rel >= NOVA_REL_MIN:
        band = "NOVA"
    elif RED_REL_MIN <= rel <= RED_REL_MAX:
        band = "Reddening"
    tag_reads = self._tag_reads.setdefault(tile_id, [])
    tag_reads.append((band, night))
    return reports


_ad.AnomalyDetector._classify = _classify_hierarchical

_orig_detector_init = _ad.AnomalyDetector.__init__


def _detector_init_hook(self, initial_publication):
    _orig_detector_init(self, initial_publication)
    try:
        cal = (initial_publication.get("calendar") or {})
        _S["night_count"] = int(cal.get("night_count") or 0)
        _S["tile_count"] = int((initial_publication.get("tile_catalog") or {}).get("tile_count") or 0)
        slot_dur = int(cal.get("slot_duration_seconds") or 0)
        if slot_dur > 0:
            _S["slot_seconds"] = slot_dur
        # 任务卡自适应：分区清单与评分常数一律从官方 initialize 合约读取（E/F/G/H 未知卡的泛化前提）
        region_ids = (initial_publication.get("tile_catalog") or {}).get("region_ids") or []
        if region_ids:
            _S["regions"] = [str(r) for r in region_ids]
        sc = (initial_publication.get("scoring_contract") or {}).get("score_config") or {}
        if sc.get("program_bonus"):
            _S["program_bonus"] = {str(k): float(v) for k, v in sc["program_bonus"].items()}
        tags = sc.get("anomaly_tags") or {}
        if tags.get("nova_factor") is not None:
            _S["nova_factor"] = float(tags["nova_factor"])
        if tags.get("reddening_factor") is not None:
            _S["red_factor"] = float(tags["reddening_factor"])
        # v7 择优等待需要的天区可用期与日历边界（tile_catalog.csv_row 自带）
        _S["first_night"] = None
        _S["last_night"] = None
        try:
            if cal.get("first_night"):
                _S["first_night"] = date.fromisoformat(str(cal["first_night"])[:10])
            if cal.get("last_night"):
                _S["last_night"] = date.fromisoformat(str(cal["last_night"])[:10])
        except ValueError:
            pass
        avail = {}
        for t in (initial_publication.get("tile_catalog") or {}).get("tiles") or []:
            avail[str(t.get("tile_id"))] = (t.get("available_from_utc"), t.get("available_until_utc"))
        _S["tile_avail"] = avail
    except Exception:  # noqa: BLE001
        pass


_ad.AnomalyDetector.__init__ = _detector_init_hook


# ---------------- v6.8 REQUIRED 末班车游标预留 ----------------

def _harvest_publications(snapshot, memory):
    """从 night_start / weekly 发布里缓存窗口知识（只存 REQUIRED，量小且够用）。"""
    cursor = snapshot.get("cursor") or {}
    night = cursor.get("night_id")
    night_start = snapshot.get("night_start")
    if isinstance(night_start, dict) and night:
        rows = night_start.get("tile_windows") or []
        if rows and _S["night_windows_for"] != night:
            cache_req = {}
            cache_meta = {}
            for row in rows:
                try:
                    tid = str(row.get("tile_id") or "")
                    cls = str(row.get("scheduling_class") or "")
                    ws = _parse_utc(row.get("window_start_utc"))
                    we = _parse_utc(row.get("window_end_utc"))
                    ex = int(row.get("nominal_exptime_seconds") or 0)
                    bt = _parse_utc(row.get("best_time_utc"))
                    bam = float(row.get("best_airmass") or 0.0)
                except Exception:  # noqa: BLE001
                    continue
                if not tid or ws is None or we is None or ex <= 0:
                    continue
                if cls == "REQUIRED":
                    cache_req.setdefault(tid, []).append({"ws": ws, "we": we, "ex": ex})
                if bt is not None and bam > 0:
                    cache_meta.setdefault(tid, []).append((ws, we, bt, bam))
            _S["night_windows_req"] = cache_req
            _S["night_windows_meta"] = cache_meta
            _S["night_windows_for"] = night
    # LLM 任务规划：每夜首次决策触发（cursor.night_id 变化即触发，与 night_start 发布
    # 与否解耦——实测 dev-fortnight 的 night_start 只在前几夜出现）。无平台注入凭据时
    # llm_planner 整层惰性，零调用零行为差异。
    if _lp is not None and night and _S.get("llm_night_for") != night:
        _S["llm_night_for"] = night
        try:
            tiles_known = set(_S.get("tile_avail") or {})
            regions_known = set(_S.get("regions") or set())
            if not regions_known:
                regions_known = {t.region_id for t in _S.get("tile_objs", [])} or None
            if tiles_known and regions_known:
                plan = _lp.night_plan(snapshot, memory, tiles_known, regions_known)
                if plan:
                    memory["plan"] = plan
        except Exception as exc:  # noqa: BLE001  规划失败只丢计划，绝不影响决策流
            print(f"llm night-plan fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    weekly = snapshot.get("weekly")
    if isinstance(weekly, dict) and night:
        for row in (weekly.get("tile_windows") or []):
            try:
                if str(row.get("scheduling_class") or "") != "REQUIRED":
                    continue
                tid = str(row.get("tile_id") or "")
                rid = str(row.get("night_id") or "")
                ws = _parse_utc(row.get("window_start_utc"))
                we = _parse_utc(row.get("window_end_utc"))
                ex = int(row.get("nominal_exptime_seconds") or 0)
            except Exception:  # noqa: BLE001
                continue
            if not tid or not rid or ws is None or we is None or ex <= 0:
                continue
            bucket = _S["future_windows"].setdefault(tid, [])
            entry = (rid, ws, we, ex)
            if entry not in bucket:
                bucket.append(entry)


def _grid_starts(ws, we, ex, now, slot_start, slot_dur):
    """当夜时隙网格上、晚于 now 且能让曝光在窗口内完成的所有起点。"""
    starts = []
    span = timedelta(seconds=slot_dur)
    k = int((now - slot_start).total_seconds() // slot_dur) + 1
    while True:
        g = slot_start + span * k
        if g >= we:
            break
        if g >= ws and (g + timedelta(seconds=ex)) <= we:
            starts.append(g)
        k += 1
    return starts


def _has_future_window(tile, night):
    """weekly 视野内是否还有（容得下整段曝光的）未来夜窗口。"""
    for rid, ws, we, ex in _S["future_windows"].get(tile, []):
        if rid > night and (we - ws).total_seconds() >= ex:
            return True
    return False


def _required_deferrable(tile, night, t_star, now, slot_start, slot_dur):
    """该 REQUIRED 之后（严格晚于 t*）还有没有已知的捕获机会。"""
    for w in _S["night_windows_req"].get(tile) or []:
        for g in _grid_starts(w["ws"], w["we"], w["ex"], now, slot_start, slot_dur):
            if g > t_star:
                return True
    return _has_future_window(tile, night)


def _is_last_chance(tile, night, now, slot_start, slot_dur):
    """错过现在这次就再无已知机会（今晚已无更晚起点且未来无已知窗口）。"""
    try:
        wins = _S["night_windows_req"].get(tile) or []
        if not wins:
            return False
        for w in wins:
            if _grid_starts(w["ws"], w["we"], w["ex"], now, slot_start, slot_dur):
                return False
        return not _has_future_window(tile, night)
    except Exception:  # noqa: BLE001
        return False


def _reserve_wait_needed(candidates, completed, snapshot):
    """v6.8 触发判定：是否应把本时隙让给「下一时隙边界才能起拍」的末班车 REQUIRED。"""
    cursor = snapshot.get("cursor") or {}
    night = cursor.get("night_id")
    slot_dur = _S.get("slot_seconds")
    if not slot_dur or not night or _S["night_windows_for"] != night or not _S["night_windows_req"]:
        return False, 0.0
    now = _parse_utc(cursor.get("timestamp_utc"))
    if now is None:
        return False, 0.0
    try:
        offset = float(cursor.get("slot_offset_seconds") or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    slot_start = now - timedelta(seconds=offset)
    t_star = slot_start + timedelta(seconds=slot_dur)
    for c in candidates:
        if c["scheduling_class"] == "REQUIRED" and str(c["tile_id"]) not in completed:
            if not _required_deferrable(str(c["tile_id"]), night, t_star, now, slot_start, slot_dur):
                return False, 0.0   # 有濒死 REQUIRED 现在就能拍：走硬优先，不预留
    for tile, wins in _S["night_windows_req"].items():
        if tile in completed:
            continue
        starts = []
        for w in wins:
            starts.extend(_grid_starts(w["ws"], w["we"], w["ex"], now, slot_start, slot_dur))
        if len(starts) == 1 and starts[0] == t_star and not _has_future_window(tile, night):
            return True, slot_dur - offset   # 贴合曝光上限（时隙剩余秒数）
    return False, 0.0


def _rise_factor(raw_map, tile_id, now, exptime, exponent):
    """v6.11：中天前起拍、曝光跨入中天的候选的均值质量加成（boost-only）。

    预览用决策瞬间的 airmass 评估整段曝光，权威计分按段取中点几何；comp-like 实测
    爬升段中位 realized/est = 1.024（被系统性低估）。用 night_start 行的
    best_time_utc/best_airmass 做割线外推曝光末端 airmass，取曝光均值，factor =
    (am_now/am_avg)^exponent，max(1.0, ·) 只保留加成侧（恶化段不动——饱和系统中
    延后恶化段候选无益），夹 [1.0, 1.15]。任何异常或数据缺失返回 1.0。"""
    try:
        rows = _S["night_windows_meta"].get(tile_id) or []
        if not rows:
            return 1.0
        raw = raw_map.get(tile_id)
        if not raw:
            return 1.0
        am_now = (raw.get("geometry") or {}).get("airmass")
        if am_now is None:
            return 1.0
        am_now = float(am_now)
        if am_now < 1.0:
            return 1.0
        row = None
        for r in rows:
            if r[0] <= now < r[1]:
                row = r
                break
        if row is None:
            return 1.0
        best_t, best_am = row[2], row[3]
        if best_t is None or best_am is None or best_am <= 0:
            return 1.0
        if now >= best_t:
            return 1.0                     # 已过中天：无爬升加成
        end = now + timedelta(seconds=exptime)
        s_rise = (am_now - best_am) / max(1.0, (best_t - now).total_seconds())
        if end <= best_t:
            am_end = best_am + s_rise * (best_t - end).total_seconds()
        else:
            am_end = best_am + s_rise * (end - best_t).total_seconds()
        am_end = max(1.0, am_end)
        am_avg = 0.5 * (am_now + am_end)
        if am_avg <= 0 or am_avg >= am_now:
            return 1.0
        factor = (am_now / am_avg) ** exponent
        return max(1.0, min(1.15, factor))
    except Exception:  # noqa: BLE001
        return 1.0

# ---------------- 决策策略 ----------------
REQUIRED_RATE_FLOOR = 0.5


def _bonus_of(program):
    """项目加成从官方评分合约读取（任务卡可变），回退内置默认。"""
    return float(_S.get("program_bonus", {}).get(program, _BONUS.get(program, 0.0)))


def _nova_factor():
    return float(_S.get("nova_factor", NOVA_FACTOR))


def _red_factor():
    return float(_S.get("red_factor", RED_FACTOR))


def _jain_evenness(counts, n_regions):
    s = sum(counts.values())
    q = sum(v * v for v in counts.values())
    if s == 0 or q == 0:
        return 0.0
    return (s * s) / (n_regions * q)


def _remember(memory, choice):
    if choice.get("tile_id"):
        memory.setdefault("tile_base", {})[choice["tile_id"]] = float(choice.get("estimated_science_score") or 0.0)


def _tile_value(memory, c):
    """从首次观测学习 tile_science_value（重复观测时 preview 估值会被压低）。"""
    vmap = memory.setdefault("_v", {})
    tid = c["tile_id"]
    if tid not in vmap and c.get("combined_quality"):
        est = float(c.get("estimated_science_score") or 0.0)
        if est > 0:
            vmap[tid] = est / float(c["combined_quality"])
    return vmap.get(tid)


def _choose_quality(candidates, snapshot, memory, reserve_ctx, reserve_fit_rem, top):
    """v7 择优等待（W=0 且非 mechanics）：每块天区只有**第一次完成的曝光**入账
    （scoring_core.apply_decision：非 mechanics 下重复观测 = duplicate_tile 无效，
    且完成前不入账），因此得分完全由「那一次的综合质量」决定——
        base = Σ tile_value × (大气质量 × 月光因子)，质量 = 效率×透射×天光/(seeing×airmass)
    实测（dev-reference，180 夜）：贪心首拍 mean_q=0.555（airmass 均值 1.44），
    而单块 oracle mean_q=1.210、榜一实测 1.0986（15469.61/Σtile_value 14081.10）。
    天空 95% 时隙可观测、每块中位 2456 个合法起拍点 → 缺的不是机会是门槛。

    策略：按「该天区剩余可用夜数占比 frac」给质量阈值——
    还有很多夜就死等好质量（等待罚金仅 0.001/s = 0.9/时隙），机会将尽才放松，
    最后一夜无条件拍（防 1000/100 终局罚分）。请求带截止期，临近即拍。
    """
    if not candidates:
        return None
    now = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
    night_id = (snapshot.get("cursor") or {}).get("night_id") or ""
    # 日期基准一律用**时间戳的 UTC 日期**：天区可用期/窗口都是按时间戳给的，
    # 而 night_id 记的是「夜的编号日期」（夜从 02:00 UTC 跨到次日 12:00 UTC），
    # 用 night_id 会让「最后一夜」判定整整晚一天 → T00001 这类 14 夜短窗口天区漏拍。
    now_date = now.date() if now is not None else None

    # ---- 双门阈值：天气因子 W + 中天几何 ----
    # 质量 = w / airmass，其中 w = 效率×透射×天光/seeing × 月光因子（与几何无关）。
    # 实测 dev-reference：w 的 p50=0.662、p90=0.981、p99=1.263；各天区中天 airmass
    # 中位 1.084、最差 1.870。用绝对 q 阈值会把「中天 airmass 高」的天区永远饿死
    # （T00002 的 q 天花板只有 0.805），所以阈值改在 w 上（天气好坏人人平等），
    # 几何单独用「airmass ≤ 当夜 best_airmass × 系数」把关 → 两门都过才拍，
    # 拍的时候仍按公开收益排序。剩余可用夜数不足时按档放松 w 阈值（防漏拍）。
    global _GATE_MODE
    _GATE_MODE = (os.environ.get("AURORA_GATE") or "w").strip().lower()
    w_hi = _env_float("AURORA_W", 0.90)
    w_step = _env_float("AURORA_W_STEP", 0.15)
    w_floor = _env_float("AURORA_W_FLOOR", 0.55)
    geo_slack = _env_float("AURORA_GEO", 1.15)
    q_floor = _env_float("AURORA_Q_FLOOR", 0.0)
    req_slack = _env_float("AURORA_Q_REQ_SLACK", 0.15)
    urgent_days = _env_float("AURORA_REQ_URGENT", 5.0)

    deadlines = {}
    for req in snapshot.get("active_requests") or []:
        dl = _parse_utc(req.get("deadline_utc"))
        if dl is not None:
            deadlines[str(req.get("request_id"))] = dl

    geo_by_tile = {}
    for sc_row in snapshot.get("candidate_tiles") or []:
        geo_by_tile.setdefault(str(sc_row.get("tile_id")), sc_row.get("geometry") or {})

    # 日历压力守门（沿用 v6.4 松弛门）：剩余夜×40时隙 不足以覆盖剩余天区×12 时，
    # 说明没有挑剔的余地——直接退回公开排序，避免短场景（14 夜/7 夜）漏拍。
    night_count = _S.get("night_count") or 0
    tile_count = _S.get("tile_count") or 0
    progress = snapshot.get("progress") or {}
    completed = set(progress.get("completed_tile_ids") or [])
    if night_count and tile_count:
        seq = _S.setdefault("night_seq", {})
        if night_id and night_id not in seq:
            seq[night_id] = len(seq)
        nights_left = night_count - (seq.get(night_id, 0) + 1)
        tiles_left = tile_count - len(completed)
        if nights_left * 40 < tiles_left * _env_float("AURORA_PRESSURE", 12.0):
            _remember(memory, top)
            return top

    last_night = _S.get("last_night")
    avail_map = _S.get("tile_avail") or {}

    def frac_of(tile_id):
        """剩余可用日历占比（按时间戳日期；无信息时返回 1.0 = 最挑剔）。"""
        if now_date is None:
            return 1.0
        raw = avail_map.get(tile_id)
        end = None
        if raw and raw[1]:
            try:
                end = date.fromisoformat(str(raw[1])[:10])
            except ValueError:
                end = None
        if last_night is not None:
            last_day = last_night + timedelta(days=1)   # 最后一夜的时间戳落在次日
            if end is None or end > last_day:
                end = last_day
        if end is None:
            return 1.0
        nights_left = (end - now_date).days + 1
        if nights_left <= 1:
            return 0.0                                    # 最后一夜：无条件拍
        start = None
        if raw and raw[0]:
            try:
                start = date.fromisoformat(str(raw[0])[:10])
            except ValueError:
                start = None
        if start is None:
            first_night = _S.get("first_night")
            if first_night is not None:
                start = first_night + timedelta(days=1)
        total = (end - start).days + 1 if start else max(nights_left, 1)
        return nights_left / max(total, 1)

    # 放松基准 = 该天区**剩余可用夜数占比**（frac_of）：实测比「距首次可拍的夜数」更好——
    # 后者会让天区在 L 夜后一路掉到地板价（dev-reference base 10009→8356），而按可用期占比
    # 让 180 夜天区前 99 夜都保持高门槛（base 10009、罚分 1628），短窗口天区自然提前放松。
    def w_gate_for(tile_id):
        frac = frac_of(tile_id)
        if frac >= 1.0:
            # 等待窗口已用尽 → 再看该天区自身可用期是否也快结束（提前进入地板价）
            frac = min(frac, max(frac_of(tile_id), 0.0))
        if frac >= 0.55:
            return w_hi
        if frac >= 0.30:
            return w_hi - w_step
        if frac >= 0.12:
            return w_hi - 2 * w_step
        if frac >= 0.04:
            return w_hi - 3 * w_step
        return w_floor

    def geo_ok_for(tile_id, c):
        """几何门：要求已接近当夜中天（airmass ≤ best_airmass×slack）。
        两种情况必须放行，否则会把天区饿死：
          1) 最佳时刻已过（中天之后 airmass 只会变差，继续等没有意义）；
          2) 窗口即将关闭（等不到更好的几何了）。"""
        rows = _S.get("night_windows_meta", {}).get(tile_id)
        if not rows:
            return True
        geo = geo_by_tile.get(tile_id) or {}
        try:
            am = float(geo.get("airmass") or 0.0)
        except (TypeError, ValueError):
            am = 0.0
        if am <= 0:
            return True
        try:
            best_am = min(r[3] for r in rows if r and r[3])
        except Exception:  # noqa: BLE001
            return True
        if am <= best_am * geo_slack:
            return True
        if now is None:
            return True
        # 已过中天：放行
        past_best = any(r[2] is not None and now >= r[2] for r in rows if r)
        if past_best:
            return True
        # 窗口将闭：放行
        exposure = float(c.get("nominal_exptime_seconds") or 0.0)
        closing_soon = any(r[1] is not None and (r[1] - now).total_seconds() <= 1.5 * exposure + 900
                           for r in rows if r)
        return closing_soon

    passing, urgent = [], []
    for c in candidates:
        q = float(c.get("combined_quality") or 0.0)
        if q <= 0.0:
            continue
        tid = str(c["tile_id"])
        geo = geo_by_tile.get(tid) or {}
        try:
            am = float(geo.get("airmass") or 0.0)
        except (TypeError, ValueError):
            am = 0.0
        w = q * am if am > 0 else q
        if not geo_ok_for(tid, c):
            continue
        if q < q_floor:
            continue
        req_id = str(c.get("request_id") or "")
        if frac_of(tid) <= 0.0:
            passing.append(c)               # 该天区最后一夜：无条件拍，防 1000/100 终局罚分
            continue
        gate = w_gate_for(tid)
        # ---- v7.2 相对质量门（FLEXIBLE）：q ≥ ρ×该天区历史最大 q（预热 K 夜后启用）----
        # 每块天区只入账第一次曝光 → 得分由那一次的质量决定。天气是站点级的：
        # 好夜多块天区同时过门、坏夜集体等待（等待罚金 0.9/时隙，预算内）。
        # 绝对 w 门的问题：把「高天花板天区」（可到 q≥1.2）和「低天花板天区」
        # （T00002 上限 0.805）用同一把 w 尺子量——前者拍早了，后者仍被饿。
        # 相对门按各自上限 ρ 比例设bar；预热期只积累不拍，避免首夜低bar锁死质量。
        seen = _S.setdefault("q_seen", {})
        ent = seen.setdefault(tid, [0.0, 0])       # [max_q, seen_nights]
        if q > ent[0]:
            ent[0] = q
        # 记录出现夜数（每夜一次）
        if len(ent) < 3:
            ent.append(night_id)
            ent[1] = 1
        elif ent[2] != night_id:
            ent[1] += 1
            ent[2] = night_id
        warm_k = _env_float("AURORA_WARM_K", 8.0)
        rel_bar = _env_float("AURORA_REL_BAR", 0.80)
        absfloor = _env_float("AURORA_ABS_FLOOR", 0.55)
        is_required = str(c.get("scheduling_class")) == "REQUIRED"
        dl0 = deadlines.get(req_id) if req_id else None
        urgent_now = dl0 is not None and now is not None and (dl0 - now).total_seconds() <= urgent_days * 86400
        if not is_required and ent[1] >= warm_k and ent[0] > 0:
            # ---- v7.3 相对质量指派（q_rel 贪心）----
            # 榜首画像（mean_q 0.98 / 62 夜拍完 / 罚 2228）不是「每块天区等自己的记录夜」，
            # 而是全局指派：每夜拍「此刻相对质量 q/max_seen 最高的天区」。
            # q_rel = q / 该天区历史最大 q —— 低天花板天区轻松到 0.9，高天花板天区只在
            # 顶级夜过线；站点级天气让好夜多块同过、坏夜集体等待（罚金预算内）。
            if req_id and urgent_now:
                urgent.append(c)              # 截止期临近：无条件拍（必须在此处，之前的 continue 会让它不可达）
                continue
            bar = rel_bar
            if req_id and dl0 is not None and now is not None:
                # 请求天区按截止期临近度递减 bar：远期按质量择机，临期放松，urgent 兜底
                dl_days = (dl0 - now).total_seconds() / 86400.0
                if 10 < dl_days <= 21:
                    bar *= 0.93
                elif urgent_days < dl_days <= 10:
                    bar *= 0.85
            # 只在「已过当夜中天」后出手：过峰后 q 单调下降，首个达标时隙就是本夜最好价
            at_peak = False
            if now is not None:
                for r in _S.get("night_windows_meta", {}).get(tid) or []:
                    if r[2] is not None and now >= r[2]:
                        at_peak = True
                        break
            q_rel = q / ent[0]
            if q_rel >= bar and q >= absfloor and at_peak:
                c["_q_rel"] = q_rel
                passing.append(c)
            continue
        # REQUIRED / 请求 / 预热期：沿用 w 门 + 梯度放松（保完成度）
        if _GATE_MODE == "q":
            if q < gate:
                continue
            if req_id:
                dl = deadlines.get(req_id)
                if dl is not None and now is not None and (dl - now).total_seconds() <= urgent_days * 86400:
                    urgent.append(c)
                    continue
                if q < max(w_floor, gate - req_slack):
                    continue
            passing.append(c)
            continue
        if req_id and urgent_now:
            urgent.append(c)                  # 截止期临近：无条件拍
            continue
        if req_id:
            continue                          # 未到 urgent 且没过相对门：继续等更好的夜
        if w >= gate:
            passing.append(c)

    pool = urgent or passing
    if not pool:
        if os.environ.get("AURORA_Q_DEBUG"):
            scored = []
            for c in candidates:
                tid = str(c["tile_id"])
                geo = geo_by_tile.get(tid) or {}
                try:
                    am = float(geo.get("airmass") or 0.0)
                except (TypeError, ValueError):
                    am = 0.0
                q = float(c.get("combined_quality") or 0.0)
                scored.append((q * am if am > 0 else q, am, tid, w_gate_for(tid)))
            best = max(scored)
            print(f"QDBG wait  n_cand={len(candidates)} best_w={best[0]:.3f} am={best[1]:.3f} "
                  f"tile={best[2]} gate={best[3]:.3f} urgent={len(urgent)} night={night_id}",
                  file=sys.stderr, flush=True)
        return None
    if reserve_fit_rem is not None:
        fitted = [c for c in pool
                  if float(c["nominal_exptime_seconds"] or 0.0) <= reserve_fit_rem + 0.5]
        if fitted:
            pool = fitted
    # LLM 夜计划加权：只在已通过质量门的 pool 内起作用（priority/focus 提升排序偏好、
    # avoid 在有替代时剔除）——计划永远不能绕过质量门，只能决定「门内谁先拍」。
    plan = memory.get("plan") or {}
    rev = memory.get("plan_revision") or {}
    prio_ids = {str(t) for t in (plan.get("priority_tiles") or [])} | {str(t) for t in (rev.get("boost_tiles") or [])}
    focus_regions = {str(r) for r in (plan.get("focus_regions") or [])} | {str(r) for r in (rev.get("focus_regions") or [])}
    avoid_regions = {str(r) for r in (rev.get("avoid_regions") or [])}
    if avoid_regions:
        kept = [c for c in pool if str(c.get("region_id")) not in avoid_regions]
        if kept:
            pool = kept

    def _plan_boost(c) -> float:
        if str(c["tile_id"]) in prio_ids:
            return 0.15
        if str(c.get("region_id")) in focus_regions:
            return 0.06
        return 0.0

    if (_env_str("AURORA_PICK", "rel") or "rel").lower() == "rel":
        pick = max(pool, key=lambda c: (c.get("_q_rel", 0.0) + _plan_boost(c),
                                        c.get("estimated_gain_per_second") or 0.0))
    else:
        pick = max(pool, key=lambda c: (c.get("estimated_gain_per_second") or 0.0,
                                        c.get("estimated_total_gain") or 0.0))
    if urgent and pick not in urgent:
        pick = max(urgent, key=lambda c: (c.get("estimated_gain_per_second") or 0.0,
                                          c.get("estimated_total_gain") or 0.0))
    q = float(pick.get("combined_quality") or 0.0)
    pgeo = geo_by_tile.get(str(pick["tile_id"])) or {}
    try:
        p_am = float(pgeo.get("airmass") or 0.0)
    except (TypeError, ValueError):
        p_am = 0.0
    pick["reason"] = (f"quality gate v7 w={q * p_am if p_am > 0 else q:.2f} q={q:.2f} "
                      f"gate={w_gate_for(str(pick['tile_id'])):.2f}" +
                      (" urgent request" if pick in urgent else ""))
    if os.environ.get("AURORA_Q_DEBUG"):
        print(f"QDBG shoot n_cand={len(candidates)} pick_q={q:.3f} am={p_am:.3f} "
              f"w={q * p_am if p_am > 0 else q:.3f} gate={w_gate_for(str(pick['tile_id'])):.3f} "
              f"urgent={len(urgent)} pass={len(passing)} night={night_id}",
              file=sys.stderr, flush=True)
    _remember(memory, pick)
    return pick


def choose_action(candidates, snapshot, memory):
    if not candidates:
        return None

    top = candidates[0]
    top_rate = top["estimated_gain_per_second"] or 0.0
    feedback = snapshot.get("tile_last_finished")
    if feedback and feedback.get("tile_id"):
        tid = str(feedback["tile_id"])
        score = float(feedback.get("score") or 0.0)
        if score > 0:
            _S["banked"][tid] = max(_S["banked"].get(tid, 0.0), score)
            for entry in _S["tile_reads"].get(tid, []):
                if entry[0] > 0 and not entry[2]:
                    entry[2] = score
                    break
    progress = snapshot.get("progress") or {}
    completed = set(progress.get("completed_tile_ids") or [])
    cfg = snapshot.get("score_config") or {}
    coverage_weight = float(cfg.get("coverage_bonus_weight") or 0.0)
    mechanics = snapshot.get("schema_version") == "decision-snapshot-v3"

    # 1.5) v6.8 REQUIRED 末班车游标预留：收割窗口知识并判定是否需要让出本时隙。
    #    任何异常只丢一次预留机会，绝不影响后续决策（教训 2）。
    reserve_fit_rem = None      # 非 None → 预留生效；数值 = 贴合曝光上限（时隙剩余秒）
    reserve_ctx = None          # (now, slot_start, slot_dur) 当夜缓存可用时用于 last_chance 判定
    reserve_night = (snapshot.get("cursor") or {}).get("night_id")
    try:
        _harvest_publications(snapshot, memory)
    except Exception as exc:  # noqa: BLE001
        print(f"pub-harvest fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    try:
        fire, fit_rem = _reserve_wait_needed(candidates, completed, snapshot)
        if fire:
            reserve_fit_rem = fit_rem
    except Exception as exc:  # noqa: BLE001
        print(f"reserve-check fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    try:
        if _S.get("night_windows_for") == reserve_night and _S.get("slot_seconds"):
            now_lc = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
            if now_lc is not None:
                off_lc = float((snapshot.get("cursor") or {}).get("slot_offset_seconds") or 0.0)
                reserve_ctx = (now_lc, now_lc - timedelta(seconds=off_lc), _S["slot_seconds"])
    except Exception:  # noqa: BLE001
        reserve_ctx = None

    # v7 路由：宽松场景（W=0 且非 mechanics）改走「择优等待」。
    # 这类场景每块天区只入账第一次完成的曝光（重复观测 = 无效），
    # 所以 base/bonus 完全由那一次的质量决定 → 质量阈值 + 剩余机会放松。
    # 覆盖型（W>0）与异常型（mechanics）场景保持 v6.11 行为不变。
    # 短场景（夜数 < AURORA_V7_MIN_NIGHTS，默认 30）不走 v7：实测 14 夜 dev-fortnight
    # 走 v7 会漏 T00041（-1000）与 flexible 配额（-400），7 夜 demo-week 也无择时余地；
    # 这些场景沿用 v6.4 的「REQUIRED 硬优先 + 几何择时」。
    if (coverage_weight <= 0 and not mechanics
            and (_S.get("night_count") or 0) >= _env_float("AURORA_V7_MIN_NIGHTS", 30.0)):
        try:
            pick = _choose_quality(candidates, snapshot, memory, reserve_ctx, reserve_fit_rem, top)
        except Exception as exc:  # noqa: BLE001  任何异常回退到公开排序，绝不等待成灾
            print(f"quality gate fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return top
        return pick

    # 1) 未观测 REQUIRED 硬优先（预留期间只允许贴合曝光；末班车者最高优先）
    best_required, best_key = None, None
    for c in candidates:
        if c["scheduling_class"] != "REQUIRED" or c["tile_id"] in completed:
            continue
        if reserve_fit_rem is not None and \
                float(c["nominal_exptime_seconds"] or 0.0) > reserve_fit_rem + 0.5:
            continue
        rate = c["estimated_gain_per_second"] or 0.0
        last_chance = False
        if reserve_ctx is not None:
            last_chance = _is_last_chance(str(c["tile_id"]), reserve_night,
                                          reserve_ctx[0], reserve_ctx[1], reserve_ctx[2])
        key = (last_chance, rate)
        if best_key is None or key > best_key:
            best_required, best_key = c, key
    if best_required is not None:
        best_required["reason"] = "unobserved REQUIRED, hard priority" + \
            (" (last chance)" if best_key[0] else "")
        _remember(memory, best_required)
        return best_required
    if reserve_fit_rem is not None and coverage_weight <= 0:
        return None   # 宽松场景：整段等待换 t* 边界决策

    # 2) 宽松场景（覆盖权重 0）：质量择时。实测贪心在天区刚升过 30°（airmass≈2.0、
    #    全夜最差几何）就开拍；等它过中天（HA≥0），airmass 降到 ~1.15，质量提升 ~1.7 倍，
    #    而等待罚金仅 0.9/时隙。天区按最高分入账 → 推迟到中天附近是纯赚。
    if coverage_weight <= 0:
        # 松弛门：日历压力（剩余夜×约40时隙 ÷ 剩余天区）不足时不得择时——
        # 实测 180 夜场景松弛 ~114（择时 +457），14 夜 ~8 / 7 夜 ~4（择时分别 -130/-426）。
        night_count = _S.get("night_count") or 0
        tile_count = _S.get("tile_count") or 0
        if night_count and tile_count:
            seq = _S.setdefault("night_seq", {})
            nid = (snapshot.get("cursor") or {}).get("night_id", "")
            if nid and nid not in seq:
                seq[nid] = len(seq)
            nights_left = night_count - (seq.get(nid, 0) + 1)
            tiles_left = tile_count - len(completed)
            if nights_left * 40 < tiles_left * _env_float("AURORA_PRESSURE", 12.0):
                _remember(memory, top)
                return top
        cursor_ts = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
        geo = {}
        for sc in snapshot.get("candidate_tiles") or []:
            geo[str(sc.get("tile_id"))] = sc.get("geometry") or {}
        ready = []
        for c in candidates:
            if c.get("request_id"):
                ready.append(c)   # 请求有截止期，不推迟
                continue
            g = geo.get(c["tile_id"]) or {}
            ha = g.get("hour_angle_deg")
            if ha is None:
                ready.append(c)
                continue
            alt = g.get("altitude_deg")
            # airmass 改善集中在升起后头几段（30°→45°，airmass 2.0→1.41）；中天附近几乎不再改善。
            # 等待有 0.9/时隙的成本 → 只等快速爬升段：高度 ≥45° 或时角 ≥−30° 即拍。
            if (alt is not None and alt >= 45.0) or (ha is not None and ha >= -30.0):
                ready.append(c)
                continue
            we = _parse_utc(c.get("window_end_utc") or g.get("window_end_utc"))
            if cursor_ts and we and (we - cursor_ts).total_seconds() <= 1.5 * float(c["nominal_exptime_seconds"]):
                ready.append(c)                 # 窗口将闭，最后机会
        if ready:
            pick = max(ready, key=lambda c: c["estimated_gain_per_second"])
            if pick is not top:
                pick["reason"] = "quality timing (waited for culmination)"
            _remember(memory, pick)
            return pick
        return None   # 全部还在东升：等待质量改善

    # 3) 正式赛型场景：覆盖感知边际 + 异常调整
    tile_region = memory.setdefault("tile_region", {})
    for c in candidates:
        tile_region.setdefault(c["tile_id"], c["region_id"])
        _S["tile_region"].setdefault(c["tile_id"], c["region_id"])
    region_list = _S.get("regions") or sorted({r for r in tile_region.values() if r})
    counts = {r: 0 for r in region_list}
    for tile_id in completed:
        region = tile_region.get(tile_id)
        if region is not None:
            counts[region] = counts.get(region, 0) + 1
    n_regions = max(1, len(region_list))
    base_so_far = sum(memory.get("tile_base", {}).get(t, 0.0) for t in completed)
    s_total = sum(counts.values())
    q_total = sum(v * v for v in counts.values())
    e_now = _jain_evenness(counts, n_regions)

    cur_night = night_id_of(snapshot)
    suspects = set()          # nova 类强嫌疑：rel >= 1.30 的读数几乎必真，值得花一个时隙确认
    for tid, reads in _S["tile_reads"].items():
        if any((tid, k) in _S["reported_tags"] for k in ("NOVA", "Reddening")):
            continue
        strong = False
        for r, n, _, bad in reads:
            if bad:
                continue
            nref = _night_ref(n)
            if nref and r / nref >= 1.30 and n != cur_night:
                strong = True
        if strong:
            suspects.add(tid)

    # 4) v6.7 故障报告等待期分区规避：报告已发出、平台 fault_status 尚未公布的空档里
    #    （fault_status 只在夜初发布、响应延迟 1 天；平台自带 filter_fault_scope 在公布
    #    后才把区内候选从快照里滤掉），塌陷分区每次观测只实现 ~0.5×，把时隙让给健康
    #    分区几乎纯赚（天区按最高分入账，修复后补拍不吃亏）。误报时收到 normal 答复
    #    同样解除；3 天未获答复自动解除（响应 1 天+修复 2 天已到，等下去没有意义）。
    pool = candidates
    try:
        # LLM 计划自适应：fault_status 首次公布 / nova 确认 → 重规划（≤3 次）
        if _lp is not None and mechanics:
            status_now = snapshot.get("fault_status")
            ev_key = None
            if isinstance(status_now, dict) and status_now.get("status") == "fault":
                ev_key = "fault:" + str(status_now.get("event_id", ""))
            elif _S["reported_nova"] and "nova:" not in _S:
                ev_key = "nova:" + ",".join(sorted(_S["reported_nova"])[:3])
                _S["nova:"] = True
            if ev_key and _S.get("last_replan_key") != ev_key:
                _S["last_replan_key"] = ev_key
                tiles_known = set(_S.get("tile_avail") or {})
                regions_known = set(_S.get("regions") or set())
                detail = {"event": ev_key}
                if isinstance(status_now, dict) and status_now.get("status") == "fault":
                    detail["scope"] = status_now.get("spatial_scope_payload")
                    detail["repair_by"] = status_now.get("repair_complete_utc")
                rev = _lp.replan_event("anomaly", detail, memory, tiles_known, regions_known)
                if rev:
                    memory["plan_revision"] = rev
    except Exception as exc:  # noqa: BLE001
        print(f"llm replan fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    try:
        pending_region = _S.get("pending_fault_region")
        if pending_region:
            status = snapshot.get("fault_status")
            now_ts = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
            since = _S.get("pending_fault_since")
            timed_out = now_ts is not None and since is not None \
                and (now_ts - since).total_seconds() > 3 * 86400
            if (isinstance(status, dict) and status.get("status") in ("fault", "normal")) or timed_out:
                _S["pending_fault_region"] = None
            elif now_ts is not None:
                alt = [c for c in candidates if c.get("region_id") != pending_region]
                if alt:
                    pool = alt
    except Exception as exc:  # noqa: BLE001  choose_action 抛异常=整层策略静默失效（教训 2）
        print(f"pending-fault avoidance fallback after {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        pool = candidates

    # v6.8 预留生效：本时隙只允许贴合曝光（下个决策落在/不越过 t* 边界）；
    # 没有贴合候选才整段等待。溢出跨时隙的曝光会跳过 t*，正是漏拍的成因。
    if reserve_fit_rem is not None:
        fitted = [c for c in pool
                  if float(c["nominal_exptime_seconds"] or 0.0) <= reserve_fit_rem + 0.5]
        if not fitted:
            return None
        pool = fitted

    # v6.11 修正上下文：原始几何快照 + airmass 指数 + 决策时刻；mechanics 门控——
    #    异常机制场景（comp-ano/真实正式赛同构）的 realized/est 混入隐藏效率因子且
    #    boost 会扰动故障检测流（v6.10 实测 comp-ano ≈ −519），禁用加成。
    raw_map = {}
    for rc in (snapshot.get("candidate_tiles") or []):
        try:
            raw_map[str(rc.get("tile_id"))] = rc
        except Exception:  # noqa: BLE001
            continue
    exponent = 1.0
    now_v11 = _parse_utc((snapshot.get("cursor") or {}).get("timestamp_utc"))
    boost_on = now_v11 is not None and _S.get("night_windows_for") == cur_night \
        and not (cfg.get("anomaly_tags") or cfg.get("reporting"))
    try:
        exponent = float((((snapshot.get("scoring_contract") or {}).get("weather_score_interface")
                           or {}).get("airmass_exponent")) or 1.0)
    except Exception:  # noqa: BLE001
        pass

    best, best_rate = None, None
    for c in pool:
        gain = c["estimated_total_gain"]
        sci = c["estimated_science_score"]
        tid = c["tile_id"]
        boost_f = 1.0
        if boost_on:
            boost_f = _rise_factor(raw_map, tid, now_v11,
                                   float(c["nominal_exptime_seconds"] or 0.0), exponent)
            if boost_f != 1.0:
                gain = gain - sci * (1.0 - boost_f)
                sci = sci * boost_f
        if tid in completed:
            # 重复观测价值：mechanics 下按最高分入账——当前质量能把入账抬高多少就是净赚；
            # 非 mechanics 只有已上报 nova 才有（且语义不同：仅访问计数）
            if mechanics or tid in _S["reported_nova"]:
                v = _tile_value(memory, c)
                if v:
                    tf = _nova_factor() if tid in _S["reported_nova"] else 1.0
                    potential = v * float(c["combined_quality"]) * (1.0 + _bonus_of(c["program"])) * tf * boost_f
                    banked = _S["banked"].get(tid, 0.0)
                    gain = max(gain, potential - banked)
            rate = gain / max(1.0, c["nominal_exptime_seconds"])
        else:
            e_after_c = (s_total + 1) ** 2 / (n_regions * (q_total + 2 * counts.get(c["region_id"], 0) + 1))
            marginal = gain + coverage_weight * (base_so_far * (e_after_c - e_now)
                                                 + sci * min(1.0, e_after_c))
            factor = 1.0
            if tid in _S["reported_nova"]:
                factor *= _nova_factor()
            elif tid in _S["reported_red"]:
                factor *= _red_factor()
            bonus = 60.0 if tid in suspects else 0.0
            # LLM 夜计划加权：priority_tiles +80、focus_regions 内 +40（计划只动权重不产生动作）
            plan = memory.get("plan") or {}
            rev = memory.get("plan_revision") or {}
            if tid in (plan.get("priority_tiles") or []) or tid in (rev.get("boost_tiles") or []):
                bonus += 80.0
            elif c["region_id"] in (rev.get("avoid_regions") or []):
                bonus -= 120.0
            elif c["region_id"] in (plan.get("focus_regions") or []) or c["region_id"] in (rev.get("focus_regions") or []):
                bonus += 40.0
            rate = (marginal * factor + bonus) / max(1.0, c["nominal_exptime_seconds"])
        if best_rate is None or rate > best_rate:
            best, best_rate = c, rate

    chosen = best if best is not None else top
    if chosen is top:
        _remember(memory, top)
        return top
    chosen["reason"] = f"v5 marginal (region {chosen['region_id']}, suspects={len(suspects)})"
    _remember(memory, chosen)
    return chosen


def night_id_of(snapshot):
    return (snapshot.get("cursor") or {}).get("night_id", "?")
