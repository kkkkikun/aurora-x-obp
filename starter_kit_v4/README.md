# Agent Observer v4 starter kit

Build an agent that runs a four-month spectroscopic survey from a virtual telescope at
**Paranal, Chile (virtual)**. Your agent chooses where to point, which target goes on each of the 16
fibres, how long to expose, and which program to declare. The platform scores the result.

This kit runs everything on your computer, exactly like the platform:

| Path | What it is |
|---|---|
| `agent/` | **Your submission.** A working baseline agent (`baseline_agent.py`), its planner, sky maths, an optional LLM hook, and `observer.project.json`. |
| `examples/idle_agent.py` | The smallest valid agent. It observes nothing (the "do nothing" score). |
| `cards/demo/` | A small public demo card: 7 nights, 2,400 targets. |
| `local_runner.py` | Runs an agent against a card, like the platform, and prints the score. |
| `pack_agent.py` | Zips `agent/` into a ZIP you can upload. |
| `challenge/` | The simulator, scorer and the platform's own engine adapter `v4_workflow.py` (read-only). |
| `SKILL.md` | Step-by-step instructions for coding agents. |

Only the Python standard library is needed (Python 3.9 or newer; the platform uses 3.12).

## Quick start

```bash
python3 local_runner.py                                   # baseline agent on the demo card
python3 local_runner.py --agent examples/idle_agent.py    # the "do nothing" score, for comparison
python3 pack_agent.py --out ../my-agent.zip               # ZIP for the Participate page
```

Expected on the demo card (a few seconds):

| Agent | `total` | Required missing |
|---|---|---|
| `examples/idle_agent.py` | −6200 | 120 of 120 |
| `agent/` (baseline) | 849.000608 | 3 of 120 |

The last line of the output is a JSON summary. Files are in `run_output/`: `decisions.csv`,
`observations.csv`, `messages.jsonl`, `score_report.json`, `workflow_result.json`, `actions.jsonl`, `agent.log`
(your agent's stderr).

## The task in 10 lines

1. The survey runs night by night. Each night has an observing window (sun below −18°).
2. At the start you get the whole target list: position, class, brightness (`feature_flux`), weight,
   and a `required` flag. About 5% of targets are required.
3. At each decision you send one action: `observe`, `wait`, `report` or `finish`.
4. `observe` = one pointing (alt/az) + up to 16 fibre assignments + a duration (60–3600 s) + a program.
   It can cross weather slots within one night. At the final slot's end, an unfinished exposure
   stops and scores using only its actual elapsed time, even if that is under 60 s.
5. A target scores only if it is assigned to a fibre **and** falls in that fibre's cell at the start
   of the exposure, and it stays at or above 30° altitude for the whole exposure. The 30° scoring
   limit applies to each target, not to the field centre.
6. Longer exposures and better sky give a higher score per target, capped at 1 × weight × program bonus.
7. Only each target's **best** exposure counts. Exposures do not add up.
8. Every required target that never reaches an exposure factor of 0.5 costs **50 points**.
9. Your agent never sees the weather itself. During a run it only gets short bulletins and forecasts
   (event kind + compass direction). Some cards publish their weather files for local scoring; your
   agent must not read them.
10. One wall clock per card (900 s on the platform). When it runs out, the survey stops there.

## Protocol: `participant-agent-protocol-v4`

One JSON object per line. The platform writes to your **stdin**; you answer on **stdout**. Print logs to
**stderr** only (they end up in `agent.log`). Always flush after each line.

### 1. `initialize` (once, no reply)

```json
{"protocol_version":"participant-agent-protocol-v4","message_type":"initialize","payload":{
  "schema_version":"v4-initialize-v1",
  "task_card":{"card_id":"demo","scenario_slug":"v4-demo","phase":"local"},
  "site":{"name":"Paranal, Chile (virtual)","latitude_deg":-24.6157,"longitude_deg":-70.3976,
          "utc_offset_hours":-4.0,"sun_altitude_limit_deg":-18.0,"minimum_altitude_deg":30.0},
  "survey":{"start_utc":"2026-10-02T00:00:00Z","end_utc":"2026-10-08T08:45:00Z","slot_seconds":900,
            "nights":[{"night_id":"N20261001","night_date":"2026-10-01","observing_start_utc":"2026-10-02T00:00:00Z",
                       "observing_end_utc":"2026-10-02T09:00:00Z","slot_count":36}]},
  "instrument":{"n_fibers":16,"grid_side":4,"fiber_area_deg2":0.4,"gap_deg":0.0,"glass_side_deg":0.632456,
                "pitch_deg":0.632456,"fov_side_deg":2.529822,"layout":"row-major, fiber 0 bottom-left; ...",
                "exposure":{"min_duration_seconds":60,"max_duration_seconds":3600}},
  "scoring":{"q0":0.68,"flux_zero_point":0.5,"exposure_zero_point_seconds":900,"...":"full public score config"},
  "footprint":[{"component_id":"C00","vertices":[[335.0,-5.2],[339.3,-6.3]]}],
  "targets":{"columns":["target_id","ra_deg","dec_deg","target_class","feature_flux","science_weight","required"],
             "rows":[["V4T000001",347.43,-35.96,"BGS",1.48,0.45,false]]},
  "limits":{"global_wallclock_seconds":900,"max_consecutive_reports":32,
            "response_max_bytes":524288,"decision_timeout":"global only (no per-decision timeout)"}}}
```

### 2. `decision_request` → your `decision_response`

```json
{"protocol_version":"participant-agent-protocol-v4","message_type":"decision_request","decision_sequence":17,"payload":{
  "schema_version":"v4-decision-snapshot-v1",
  "now_utc":"2026-10-02T03:30:00Z","survey_end_utc":"2026-10-08T08:45:00Z",
  "observe_action_index":12,"running_total":41.27,
  "wallclock":{"elapsed_seconds":2.4,"remaining_seconds":897.6},
  "latest_bulletin":{"record_type":"bulletin","slot_id":"N20261001-S015","issued_at_utc":"2026-10-02T03:30:00Z",
                     "initial":false,"notices":[{"event_kind":"overcast","direction":"SW"}]},
  "latest_forecast":{"record_type":"forecast","coverage_start_utc":"...","coverage_end_utc":"...",
                     "notices":[{"event_kind":"rain","direction":"ALL","nights":["2026-10-02"]}]},
  "new_messages":[],
  "last_result":{"action":"observe","observe_index":11,"assigned_count":16,"hit_count":13,
                 "hits":[{"target_id":"V4T001234","score":0.8123}]}}}
```

- `new_messages`: bulletins and forecasts published since your last request, plus an immediate
  `report_result` after a report (and, on some cards, one `state_resync`, see below).
- `last_result.hits`: the targets of your previous observe that landed on their fibre, with their score
  (0 when the sky was closed or blocked there). Assigned targets that are missing were not hits.
  Fibre ids are not returned.
- `running_total`: the sum of best scores so far (no penalties).

Answer with the same `decision_sequence` and one action. Optional `reason` and `decision_source`
strings are only logged.

| `action` | Fields | Time used |
|---|---|---|
| `observe` | `pointing: {"alt_deg": 0–90, "az_deg": [0, 360)}`, `assignments: {"0": "V4T000123", ..., "15": ...}` (fibre id → target id, each fibre and each target once), `duration_seconds` (whole number, 60–3600), `program` (`DARK`, `BRIGHT` or `BACKUP`; optional, default `BACKUP`) | the requested duration, capped at the current night's final slot end |
| `wait` | `duration_seconds` (60–3600) **or** `until_utc` (a later UTC time ending in `Z`, e.g. the next night's start) | until then |
| `report` | none: "the instrument has a fault now" | 0 s (up to the configured consecutive-report limit) |
| `finish` | none: end the survey now | ends the run |

```json
{"protocol_version":"participant-agent-protocol-v4","message_type":"decision_response","decision_sequence":17,
 "action":"observe","pointing":{"alt_deg":62.5,"az_deg":201.3},
 "assignments":{"0":"V4T000123","5":"V4T004567"},"duration_seconds":900,"program":"DARK","reason":"dense field"}
```

```json
{"protocol_version":"participant-agent-protocol-v4","message_type":"decision_response","decision_sequence":18,
 "action":"wait","until_utc":"2026-10-03T00:00:00Z","reason":"daytime"}
```

Send only the fields in the table (plus the envelope, `reason` and `decision_source`). A wrong answer
ends the run as `agent_error`: bad JSON, a wrong `decision_sequence`, an unknown or extra field, an unknown
target, a fibre used twice (`"5"` and `"05"` are the same fibre), a value out of range, a line over
512 KiB, or another `report` after `scoring.reporting.max_consecutive_reports` consecutive reports.
The demo card allows 32. An over-limit attempt ends as `agent_error` without a new report score.
`observe` or `wait` resets the report count; a correct report still counts toward it. The separate
`scoring.reporting.false_report_free_allowance` gives the number of false reports allowed without
penalty after each correct report (2 on the demo card). `observe` and `wait` do not reset that count;
only a correct report does.
The score still counts everything done before termination.

### 3. `finish` (once, no reply)

```json
{"protocol_version":"participant-agent-protocol-v4","message_type":"finish",
 "payload":{"schema_version":"v4-finish-v1","termination_reason":"survey_complete","decisions":266,
            "observe_actions":160,"last_decision_sequence":180,"grace_seconds":30}}
```

`termination_reason` is `survey_complete`, `agent_finished`, `global_wallclock_expired` or `agent_error`.
`decisions` counts the rows of `decisions.csv` (a long `until_utc` wait becomes several rows).
Then stdin closes. You have 30 seconds to write a summary to stderr and exit. The score is already fixed.
If your agent is still computing when the wall clock runs out, it is stopped at once and gets no `finish`.

## Geometry

- Pointing is the field centre in alt/az (azimuth 0 = north, 90 = east).
- 16 square assignable regions in a 4 × 4 grid. Each region covers 0.4 deg² and is
  about 0.632° wide. They touch without gaps, making the field 6.4 deg² and about
  2.530° across. A region is a target-assignment rule, not a physical fibre aperture.
- Fibre 0 is bottom-left. Rows follow increasing altitude; columns follow increasing azimuth
  in the local tangent plane. These directions rotate with the pointing, so the diagram's
  right side is not always geographic east. Fibre id = row × 4 + column.
- Target positions are projected on a flat (gnomonic) plane centred on the pointing, at the start of the
  exposure. The telescope then tracks, so targets do not drift during the exposure.
- `agent/skymath.py` has the exact formulas (sidereal time, alt/az, projection, fibre lookup).

## Scoring

For each target *i* and exposure *e*:

```
q      = time-average[instrument × transparency × sky × moon(i,t)
                      / (seeing × airmass(i,t)^0.6)] / q0                 (hidden)
factor = min( feature_flux × duration × q / (0.5 × 900), 1 )
score  = science_weight × factor × program_bonus
```

The time average is evaluated at the midpoints of pieces no longer than 120 seconds.
Each piece uses that target's altitude-dependent airmass and lunar factor at its midpoint.

- `program_bonus`: 1.20 (DARK), 1.12 (BRIGHT), 1.06 (BACKUP) when the declared program matches the
  sky's band for that target; 1.0 when it does not. The band comes from the sky quality (transparency,
  sky brightness, seeing), the Moon and the airmass. The instrument does not affect the band.
  Band limits: DARK ≥ 0.65, BRIGHT ≥ 0.40, else BACKUP.
- The Moon term uses the public lunar model in `scoring.lunar_model`.
- Only the best exposure of each target counts.

Final total for the card:

```
total = Σ best score
        − 50 × (required targets whose best factor < 0.5)
        − 200 × (1 − Jain index over 10° right-ascension bands of the fraction of targets with factor ≥ 0.5)
        + reports (+100 if an unrepaired instrument fault is active; first 2 false reports after
                   each correct report: 0, then −150 per false report on the demo card)
```

The Jain index is 1 when every RA band is observed to the same fraction, so spread your effort.
After a `report`, the next request arrives at the same simulated time. Its `last_result` gives
`correct`, `repaired`, and `score_delta`; `new_messages` also contains a `report_result` notice
with those fields and `issued_at_utc`. A correct report repairs the fault immediately. A false
or repeated report gives `correct: false` and `repaired: false`; `score_delta` is 0 within the
free threshold and −150 afterward.

**Worked example.** An ELG target with `feature_flux` 0.60 and `science_weight` 1.0. You expose 900 s
and declare DARK. The sky gives q = 0.75 and the band is DARK.
factor = min(0.60 × 900 × 0.75 / 450, 1) = 0.90. Score = 1.0 × 0.90 × 1.20 = **1.08**.
If you had declared BRIGHT, the program would not match: 1.0 × 0.90 × 1.0 = 0.90.
With 300 s instead, factor = 0.30. That is below 0.5, so a required target would still count as missing.

## Weather, bulletins and forecasts

- A **bulletin** is published every 900 s slot of the night. It lists what is active now:
  `{"event_kind": ..., "direction": ...}`. Kinds: `rain`, `overcast`, `haze`, `cold_snap`, `storm`,
  `rocket_launch`, `earthquake`, `terrain_obstruction`. Direction: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`,
  `NW`, or `ALL` (the whole sky).
- The first bulletin (`"initial": true`) also lists directions with **terrain** that blocks low altitudes
  all survey long.
- A **forecast** is published about once a week. It lists expected events and the nights they touch.
- A `rocket_launch` closes its announced sector for a short time and ends no later than the last slot
  of the night in which it starts. Any unused scheduled duration does not carry into the next night.
- There are no numbers: no cloud cover, no seeing values. Learn the real quality from your own hits.
- Some sky losses are never announced. Your scores are the ground truth.

### `state_resync` (some cards only)

On some cards part of your recent observation data can be lost. You then get one message in
`new_messages` with `record_type: "state_resync"`. It lists `observed_target_ids` and `best_scores`:
the targets that still count and their best score now. Rebuild your "already done" list from it. The
time already spent is not returned.

## Time

- One global wall clock per card: 900 s on the platform. It starts at the first `decision_request` and
  counts your thinking time and the simulator's time (about 10 ms per decision).
  `initialize.limits.global_wallclock_seconds` is your budget. Locally, `--wallclock` can only lower it.
- There is no per-decision limit. `payload.wallclock.remaining_seconds` tells you what is left.
- A full season can need about 5,000 decisions. That is about 150 ms per decision. Do not call a model
  on every decision.
- Use `wait` with `until_utc` to skip the day in one step.

## The baseline agent

`agent/baseline_agent.py` + `agent/planner.py`, pure standard library. On each decision:

1. Daytime: `wait` `until_utc` the next night.
2. `rain`/`storm` over `ALL` in the bulletin: wait one slot.
3. Find targets that are up now and stay above 30° long enough.
4. Rank them by what they can still gain **tonight**: a required target counts as a big gain only if
   its factor can reach 0.5 with the longest exposure it allows now (faint targets wait for a better
   sky). Targets that set soon or have few nights left rank higher. Targets that failed before rank lower.
5. For the best targets, try 16 pointings each (the target centred on each fibre). Fill every fibre with
   the target that can gain most there. Keep the best pointing.
6. Pick the duration with the best expected score per second. Pick the program most targets will match.
7. Learn the sky quality from recent hits. Avoid announced directions and terrain at low altitude.
8. Report a fault only after a large drop in quality that stays on three different nights and that the
   program bands do not explain. At most twice per run.

Ideas to beat it: plan whole nights ahead, balance RA bands, use forecasts, handle `state_resync`
better, and use a model for the few decisions where judgement matters.

## Optional LLM hook

`agent/llm_hook.py` shows how to call a model through the platform's proxy. On the platform your agent
gets `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The hook is off by default (`"USE_LLM": "0"` in
`observer.project.json`). Set it to `"1"` to turn it on. It makes one short call per night and one before
a report, with a 12 s timeout and a 90 s total budget. If the model is missing or slow, the agent keeps
its own rules.

To try it locally, put these lines in `agent/.env` (never upload `.env`):

```
USE_LLM=1
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL=your-model
```

## Submitting

1. `python3 pack_agent.py --out ../my-agent.zip`. The ZIP has `observer.project.json` at its root
   (`"protocol": "jsonl-v4"`, `python3 -u baseline_agent.py`). `.env` is never packed.
2. Upload it on the website's Participate page, or push `agent/` to a GitHub repository.
3. If you need packages, add a `build` step to `observer.project.json` (for example
   `["pip", "install", "-r", "requirements.txt"]`).

## Card folder

```
cards/demo/config/   v4_scenario.json (card id, site, limits), v4_fiber_config.json, v4_score_config.json
cards/demo/public/   targets.csv, footprint.csv, v4_night_calendar.csv, v4_bulletins.jsonl, v4_forecasts.jsonl
cards/demo/truth/    hidden weather and events, for local scoring only
```

Bulletins and forecasts reach your agent one by one during the run, as they are published. Your agent
must not read the card folder at all: on the platform it only has its own folder. Run another public card
with `python3 local_runner.py --card <folder>`.

## Troubleshooting

- `agent_error`: read the `error` field in the summary and the end of `run_output/agent.log`.
- Nothing on stdout except JSON lines. `print(..., file=sys.stderr)` for logs.
- Behind a local HTTP proxy, add `NO_PROXY=127.0.0.1,localhost` to `agent/.env` when you test against a
  model server on your own machine.
