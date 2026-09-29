#!/usr/bin/env python3
"""进程内快跑：和 local_runner 完全同一套 ChallengeWorkflow + MinimalDecisionAgent，
只是省掉 JSONL 子进程传输，用于秒级迭代策略参数。

    python3 tools/fast_run.py [scenario] [--out DIR] [--agent DIR]

校验：对同一场景，分数与 decisions.csv 必须与 local_runner 逐位一致。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "agent-observer-starter-kit"
sys.path.insert(0, str(KIT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default=str(KIT / "scenarios/dev-reference"))
    ap.add_argument("--agent", default=str(KIT / "agent"))
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    agent_dir = Path(args.agent).resolve()
    sys.path.insert(0, str(agent_dir))
    from decision_graph import MinimalDecisionAgent  # noqa: E402
    from challenge.challenge_workflow import ChallengeWorkflow  # noqa: E402

    root = Path(args.scenario).resolve()
    workflow = ChallengeWorkflow(root=root)
    agent = MinimalDecisionAgent(workflow.initial_publication(), model=None, top_k=12)

    def provider(snapshot, _deadline):
        return agent.decide(snapshot)

    started = time.time()
    result = workflow.run(provider, wallclock_seconds=86400.0)
    elapsed = time.time() - started
    report = result["score_report"]
    score = report["score"]
    pen = score["penalties"]
    if not args.quiet:
        print(f"scenario={root.name} term={report['termination_reason']} "
              f"wall={elapsed:.1f}s decisions={result['committed_action_count']}")
        print(f"  base={score['base_science']:.6f} bonus={score['program_bonus']:.6f} "
              f"req={score['request_reward']:.6f} cov={score['coverage_bonus']:.6f} "
              f"pen={sum(pen.values()):.6f} total={score['total']:.6f}")
        print(f"  penalties={ {k: round(v, 3) for k, v in pen.items() if v} } "
              f"tiles={len(report['completion']['completed_tiles'])} "
              f"req_miss={report['completion']['required_missing']}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
        (out / "workflow_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        with (out / "decisions.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["decision_id", "slot_id", "action", "tile_id", "program", "request_id", "reason"])
            for row in workflow.committed:
                writer.writerow([row.decision_id, row.slot_id, row.action, row.tile_id,
                                 row.program, row.request_id, row.reason])
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
