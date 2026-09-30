"""Optional LLM advice for the baseline agent (an example of where a model can help).

The platform gives every run two environment variables:
  OPENAI_BASE_URL   an OpenAI-compatible endpoint (the platform's model proxy)
  OPENAI_API_KEY    a temporary credential for this run
OPENAI_MODEL (or MODEL_NAME) picks the model; without it the platform placeholder "team-model" is used,
which the proxy maps to your team's configured model.

The hook is used only when USE_LLM=1 and both variables are present. Every call has a short timeout and
the whole run has a small total budget. When anything fails, the agent falls back to its own rules, so a
missing key or a slow model never breaks a run.

Two example uses (each one is a single short call, never one call per decision):
  * night_plan():      read tonight's forecast/bulletin notices and suggest directions to avoid and an
                       exposure-length scale (plan adaptation from natural-language-like notices);
  * confirm_report():  look at the evidence for an instrument problem and confirm or veto a report
                       (action decision).
Standard library only (urllib).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


class LLMAdvisor:
    def __init__(self, log=lambda text: None, timeout_seconds: float = 20.0, budget_seconds: float = 240.0, max_calls: int = 40):
        self.log = log
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.model = (os.environ.get("OPENAI_MODEL", "").strip() or os.environ.get("MODEL_NAME", "").strip() or "team-model")
        wanted = os.environ.get("USE_LLM", "0").strip().lower() in ("1", "true", "yes", "on")
        self.night_plan_on = os.environ.get("LLM_NIGHT_PLAN", "1").strip().lower() in ("1", "true", "yes", "on")
        self.confirm_on = os.environ.get("LLM_CONFIRM_REPORT", "1").strip().lower() in ("1", "true", "yes", "on")
        self.enabled = wanted and bool(self.base_url and self.api_key)
        self.timeout = timeout_seconds
        self.budget = budget_seconds
        self.max_calls = max_calls
        self.spent = 0.0
        self.calls = 0
        self.failures = 0
        if wanted and not self.enabled:
            self.log("llm: USE_LLM is on but OPENAI_BASE_URL / OPENAI_API_KEY are missing; using rules only")

    @property
    def status(self) -> str:
        return "on" if self.enabled else "off"

    def _chat(self, system: str, user: str, wallclock_left: float):
        """One chat completion -> parsed JSON object, or None on any problem."""
        if not self.enabled or self.calls >= self.max_calls or self.failures >= 3:
            return None
        timeout = min(self.timeout, self.budget - self.spent, max(0.0, wallclock_left - 60.0))
        if timeout < 2.0:
            return None
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 200,
        }).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.api_key})
        started = time.monotonic()
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"] or ""
            match = re.search(r"\{.*\}", text, re.S)
            return json.loads(match.group(0)) if match else None
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            self.failures += 1
            self.log(f"llm: call failed ({type(exc).__name__}); falling back to rules")  # never log the key
            return None
        finally:
            self.spent += time.monotonic() - started

    def night_plan(self, night_date: str, forecast_notices: list, bulletin_notices: list, wallclock_left: float):
        """-> {"avoid_directions": [...], "duration_scale": float} or None."""
        system = (
            "You advise a robotic telescope survey. Tonight's weather notices name event kinds "
            "(rain, overcast, haze, cold_snap, storm, rocket_launch, earthquake, terrain_obstruction) "
            "and a compass direction (N/NE/E/SE/S/SW/W/NW or ALL). "
            "Rules, in priority order: "
            "(1) avoid_directions lists ONLY the directions that tonight's notices explicitly name "
            "with a sight-blocking kind (rain, storm, overcast, rocket_launch, terrain_obstruction); "
            "(2) haze, cold_snap and earthquake do NOT block pointing: never list them; "
            '(3) direction ALL means the whole sky: return an empty list (nothing to dodge); '
            "(4) empty or absent notices -> empty list and duration_scale 1.0; "
            "(5) never list more than the explicitly named directions; listing all eight is always wrong. "
            "duration_scale: 1.0 normally; 1.2-1.4 only when a quality-dimming kind "
            "(overcast/haze/cold_snap) is announced for tonight; 0.8-0.9 when the sky is clear and "
            "you want fast target cycling. Reply with ONE JSON object only: "
            '{"avoid_directions": [...], "duration_scale": number}. '
            'Example: notices [{"event_kind":"overcast","direction":"SW"}] -> '
            '{"avoid_directions": ["SW"], "duration_scale": 1.3}. '
            'Example: notices [] -> {"avoid_directions": [], "duration_scale": 1.0}.')
        user = json.dumps({"night": night_date,
                           "forecast_notices_for_tonight": [
                               {k: n.get(k) for k in ("event_kind", "direction")}
                               for n in forecast_notices],
                           "current_bulletin_notices": [
                               {k: n.get(k) for k in ("event_kind", "direction")}
                               for n in bulletin_notices]})
        answer = self._chat(system, user, wallclock_left)
        if not isinstance(answer, dict):
            return None
        avoid = {str(d).upper() for d in answer.get("avoid_directions", []) if str(d).upper() in DIRECTIONS}
        try:
            scale = min(1.4, max(0.7, float(answer.get("duration_scale", 1.0))))
        except (TypeError, ValueError):
            scale = 1.0
        return {"avoid_directions": sorted(avoid), "duration_scale": scale}

    def confirm_report(self, evidence: dict, wallclock_left: float):
        """-> True (report), False (do not report) or None (no opinion: keep the rule's decision)."""
        system = ("You check telescope data quality. A false instrument-fault report costs points, a correct one "
                  'earns points. Reply with one JSON object only: {"report": true|false}.')
        answer = self._chat(system, json.dumps(evidence), wallclock_left)
        if isinstance(answer, dict) and isinstance(answer.get("report"), bool):
            return answer["report"]
        return None
