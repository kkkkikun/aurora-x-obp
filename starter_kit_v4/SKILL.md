# SKILL: build, test and submit an Agent Observer v4 agent

For coding agents (Claude Code, Codex, Cursor, ...) working in this kit. Follow the steps in order.
Python 3.9+ standard library only. Run commands from the kit folder. Read `README.md` for the full rules.

## 1. Check the kit works

```
python3 --version
python3 local_runner.py --quiet
python3 local_runner.py --agent examples/idle_agent.py --quiet
```

Expected on `cards/demo` (the public demo card at Paranal, Chile (virtual), 7 nights, 2,400 targets):
baseline `"termination_reason": "survey_complete"`, `"total": 849.000608`, `"required_missing": 3`;
idle agent `"total": -6200.0`. Exit code 2 means `agent_error`: read `"error"` and `run_output/agent.log`.

## 2. Know the contract (`participant-agent-protocol-v4`)

- stdin/stdout, one JSON object per line, flush after each line. Logs go to stderr only.
- Messages in: `initialize` (once, no reply), `decision_request` (reply once, same `decision_sequence`),
  `finish` (once, no reply; stdin then closes; exit within 30 s).
- Every reply: `{"protocol_version": "participant-agent-protocol-v4", "message_type": "decision_response",
  "decision_sequence": <same int>, "action": ..., ...}`.
- Actions:
  - `observe`: `pointing {alt_deg, az_deg}`, `assignments {"<fibre 0-15>": "<target_id>"}` (each target
    once, each fibre once), `duration_seconds` 60–3600 (integer), `program` `DARK|BRIGHT|BACKUP` (optional,
    default `BACKUP`). No other keys. The exposure may cross slots within a night, but stops at
    that night's final slot end; its score uses the actual elapsed seconds, even if fewer than 60.
  - `wait`: `duration_seconds` 60–3600, or `until_utc` (UTC ending in `Z`, later than `now_utc`). Use
    `until_utc` for daytime.
  - `report`: the instrument has an unrepaired fault now (+100 if true; the first 2 false reports
    after each correct report are free on the demo card, then −150 each). The next request arrives
    at the same simulated time and gives `correct`, `repaired`, and `score_delta` in `last_result`
    and a `report_result` notice in `new_messages`. Read the free threshold from
    `scoring.reporting.false_report_free_allowance` and the action cap from
    `scoring.reporting.max_consecutive_reports`. A further report after the cap ends as
    `agent_error`. `observe` or `wait` resets only the consecutive-action count; only a correct
    report resets the false-report count.
  - `finish`: end the survey.
- Any invalid reply (including an unknown key) ends the run as `agent_error` (the score so far still
  counts). Validate before sending.
- Still computing when the wall clock runs out: the process is stopped at once, with no `finish`.
- Wall clock: 900 s per card on the platform, starting at the first request. `payload.wallclock.remaining_seconds`.
  A full season can be about 5,000 decisions: keep each decision under about 100 ms. No model call per decision.

## 3. Know the score

- factor = min(feature_flux × duration × q / 450, 1); score = science_weight × factor × program_bonus.
- q is the hidden sky/instrument quality; learn it from `last_result.hits` (each hit has its score).
- program_bonus: DARK 1.20, BRIGHT 1.12, BACKUP 1.06 if the declared program matches the sky band, else 1.0.
- Only the best exposure per target counts. Required target with best factor < 0.5: −50 each.
- −200 × (1 − Jain index) over 10° RA bands. Spread observations across right ascension.

## 4. Improve the agent

Edit files in `agent/` only (`baseline_agent.py` is the entry point; `planner.py` holds the strategy and
its tunable constants; `skymath.py` has the exact geometry). After each change:

```
python3 local_runner.py --quiet --out run_output
```

Compare `total`, `required_missing`, `targets_observed` and `wall_seconds`. Keep `wall_seconds` far below
the wall clock: a formal card can be much larger than the demo card.

Do not read `cards/` from the agent. On the platform the agent only has its own folder.

## 5. Optional: use a model

`agent/llm_hook.py` calls `OPENAI_BASE_URL/chat/completions` with `OPENAI_API_KEY` (the platform injects
both). Turn it on with `"USE_LLM": "1"` in `agent/observer.project.json`. Keep calls rare (once per night
or for rare decisions), with short timeouts and a rule-based fallback. For local tests put the variables
in `agent/.env`. Never commit or pack `.env`.

## 6. Package and submit

```
python3 pack_agent.py --out ../my-agent.zip
```

The ZIP has `observer.project.json` at its root with `"protocol": "jsonl-v4"` and
`"run": ["python3", "-u", "baseline_agent.py"]`. Upload it on the website's Participate page
(or push the `agent/` folder to a GitHub repository with `observer.project.json` at its root).
If you add packages, add a `build` step, e.g. `["pip", "install", "-r", "requirements.txt"]`.
