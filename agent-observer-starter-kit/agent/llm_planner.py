"""LLM 智能体技术层：任务规划 + 计划自适应（官方六环节中的两项）。

官方要求（2026-09《关于比赛的一些说明》§四）：参赛系统必须显著使用智能体技术，
六环节（自然语言理解/数据解析/任务规划/行动决策/工具调用/计划自适应）至少两个，
由智能体技术评审团队评审。本模块落在其中两项：

  任务规划    night_plan()  —— 每夜夜初调用一次 LLM，把「未观测 REQUIRED 清单、
                活动请求与截止期、分区覆盖进度」压缩成结构化 JSON 夜计划
                （priority_tiles / focus_regions / strategy_note），
                校验后作为边际排序的加权输入。
  计划自适应  replan_event() —— 异常事件（fault_status 首次公布、nova 标签确认、
                请求临近截止）触发 LLM 重规划，产出对既有计划的修订
                （avoid_regions / boost_tiles）。

铁律（评审可核）：
  1. LLM 输出永远只是「计划/权重」，动作由确定性引擎产生——幻觉计划只会被
     丢弃，永远不会变成非法动作（invalid 罚分与 LLM 无关）。
  2. 任何异常（无 key / 超时 / 非 JSON / 字段非法）→ 静默回退确定性，分数下限
     = 无 LLM 版本。
  3. 调用预算：每夜 ≤1 次规划 + 异常事件 ≤3 次重规划（全局墙钟 3600s 内
     30 夜 × ~1-2s 完全可承受；AURORA_LLM_BUDGET 可再压）。
  4. AURORA_LLM_PROVIDER 未设置时整层惰性（零调用、零行为差异）。

配置（agent/.env 或平台环境变量）：
  AURORA_LLM_PROVIDER   例 moonshot / openai / anthropic（缺省 = 关闭）
  AURORA_LLM_MODEL      模型名（缺省按 provider 选默认）
  AURORA_LLM_KEY        API key（也可用各 provider 惯例的 *_API_KEY）
  AURORA_LLM_BASE_URL   OpenAI 兼容端点（moonshot 等需要）
  AURORA_LLM_BUDGET     全场最大调用次数（缺省 40）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import sys

PROVIDER_DEFAULTS = {
    "moonshot": ("kimi-k2-0905-preview", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "openai": ("gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("deepseek-chat", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "anthropic": ("claude-3-5-haiku-latest", "", "ANTHROPIC_API_KEY"),
}

_STATE = {
    "client": None,        # 惰性构建的 (call_fn, model_name)
    "night_done": set(),   # 已规划过的夜
    "event_count": 0,      # 重规划次数
    "calls": 0,            # 总调用数
    "failures": 0,
}


def _env(name, default=""):
    return (os.environ.get(name) or default).strip()


def _budget_left():
    budget = 40
    try:
        budget = int(_env("AURORA_LLM_BUDGET", "40"))
    except ValueError:
        pass
    return budget - _STATE["calls"] > 0


def _client():
    """惰性构建 LLM 调用函数（返回 call_fn(prompt) -> str）。未配置返回 None。"""
    if _STATE["client"] is not None:
        return _STATE["client"][0]
    provider = _env("AURORA_LLM_PROVIDER")
    if not provider or not _budget_left():
        _STATE["client"] = (None, "")
        return None
    model, base_url, key_env = PROVIDER_DEFAULTS.get(provider, ("", "", ""))
    model = _env("AURORA_LLM_MODEL") or model
    base_url = _env("AURORA_LLM_BASE_URL") or base_url
    key = _env("AURORA_LLM_KEY") or _env(key_env)
    if not model or not key:
        _STATE["client"] = (None, "")
        return None
    import urllib.request

    def call(prompt: str) -> str:
        if not _budget_left():
            raise RuntimeError("LLM budget exhausted")
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是巡天望远镜的夜间观测规划助手。只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _STATE["calls"] += 1
        return data["choices"][0]["message"]["content"]

    _STATE["client"] = (call, model)
    return call


def _extract_json(text: str):
    """从回复中提取第一个 JSON 对象（容忍 ```json 围栏与前后闲话）。"""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:]
            if p.startswith("{"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _validate_plan(plan, known_tiles, known_regions):
    """计划字段白名单化：priority_tiles/boost_tiles 里的天区必须真实存在，
    focus_regions/avoid_regions 里的分区必须真实存在；超长截断。非法 → None。"""
    if not isinstance(plan, dict):
        return None
    out = {}
    tiles = set(known_tiles)
    regions = set(known_regions)
    for key, pool, limit in (("priority_tiles", tiles, 24), ("boost_tiles", tiles, 24)):
        vals = plan.get(key)
        if isinstance(vals, list):
            cleaned = [str(v) for v in vals[:limit] if str(v) in pool]
            if cleaned:
                out[key] = cleaned
    for key, pool, limit in (("focus_regions", regions, 8), ("avoid_regions", regions, 8)):
        vals = plan.get(key)
        if isinstance(vals, list):
            cleaned = [str(v) for v in vals[:limit] if str(v) in pool]
            if cleaned:
                out[key] = cleaned
    note = plan.get("strategy_note")
    if isinstance(note, str) and note.strip():
        out["strategy_note"] = note.strip()[:200]
    return out or None


def night_plan(snapshot, memory, known_tiles, known_regions):
    """任务规划：夜初调用。返回校验后的计划 dict（或 None=不干预）。"""
    if not _budget_left():
        return None
    cursor = (snapshot.get("cursor") or {})
    night = cursor.get("night_id") or ""
    if not night or night in _STATE["night_done"]:
        return None
    _STATE["night_done"].add(night)
    call = _client()
    if call is None:
        return None
    try:
        progress = snapshot.get("progress") or {}
        completed = set(progress.get("completed_tile_ids") or [])
        flex_done = progress.get("flexible_completed_by_region") or {}
        # 素材：未观测 REQUIRED + 活动请求 + 分区进度（压缩成紧凑文本）
        unobserved_req = []
        for row in (snapshot.get("night_start") or {}).get("tile_windows") or []:
            if str(row.get("scheduling_class")) == "REQUIRED" and str(row.get("tile_id")) not in completed:
                unobserved_req.append({
                    "tile": str(row.get("tile_id")),
                    "window": [str(row.get("window_start_utc", ""))[11:16],
                               str(row.get("window_end_utc", ""))[11:16]],
                    "exptime": row.get("nominal_exptime_seconds"),
                })
        requests = [{
            "id": str(r.get("request_id")), "deadline": str(r.get("deadline_utc", ""))[5:16],
            "need": r.get("required_tile_count"), "have": r.get("satisfied_tile_count"),
            "tiles": [m.get("tile_id") for m in (r.get("tile_requirements") or [])],
        } for r in (snapshot.get("active_requests") or []) if not r.get("is_complete")]
        prompt = (
            "巡天夜计划。输出 JSON：{\"priority_tiles\": [tile_id...], "
            "\"focus_regions\": [region_id...], \"strategy_note\": \"≤40字\"}。\n"
            f"夜={night} 已完成={len(completed)}天区 分区进度={json.dumps(flex_done)}\n"
            f"今晚未观测必做天区={json.dumps(unobserved_req[:12])}\n"
            f"活动请求={json.dumps(requests[:8])}\n"
            "规则：必做天区漏一块扣1000、请求按时完成每块+140/过期-190、"
            "覆盖越均匀加分越多（各区完成数应均衡）、圆顶关闭或质量差(综合质量<0.4)时等待。"
        )
        t0 = time.time()
        reply = call(prompt)
        plan = _validate_plan(_extract_json(reply), known_tiles, known_regions)
        if plan and os.environ.get("AURORA_LLM_DEBUG"):
            print(f"llm-planner night {night}: {plan} ({time.time()-t0:.1f}s)",
                  file=sys.stderr, flush=True)
        return plan
    except Exception as exc:  # noqa: BLE001  任何失败 = 静默回退确定性
        _STATE["failures"] += 1
        print(f"llm-planner night fallback after {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


def replan_event(kind: str, detail: dict, memory, known_tiles, known_regions):
    """计划自适应：异常事件触发的重规划。返回修订计划（或 None）。"""
    if not _budget_left() or _STATE["event_count"] >= 3:
        return None
    call = _client()
    if call is None:
        return None
    try:
        prev = memory.get("plan") or {}
        prompt = (
            "巡天计划修订。输出 JSON：{\"avoid_regions\": [...], \"boost_tiles\": [...], "
            "\"strategy_note\": \"≤40字\"}。\n"
            f"事件类型={kind} 事件详情={json.dumps(detail)}\n"
            f"当前计划={json.dumps(prev)}\n"
            "规则：仪器故障区在修复完成前效率≈0.1-0.7 应避开并把观测转向其他分区；"
            "确认的天区标签（nova ×1.5 / reddening ×0.8）值得复拍刷最高分。"
        )
        _STATE["event_count"] += 1
        reply = call(prompt)
        plan = _validate_plan(_extract_json(reply), known_tiles, known_regions)
        if plan:
            plan["event_kind"] = kind
            memory["plan_revision"] = plan
        return plan
    except Exception as exc:  # noqa: BLE001
        _STATE["failures"] += 1
        print(f"llm-planner event fallback after {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


def status():
    return {"calls": _STATE["calls"], "failures": _STATE["failures"],
            "events": _STATE["event_count"]}
