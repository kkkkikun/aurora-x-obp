#!/usr/bin/env python3
"""The smallest valid agent: it reads every message and answers each decision_request with `finish`.

It observes nothing, so its score is the "do nothing" floor: every required target counts as missing.
Use it as a protocol skeleton. Run:  python3 local_runner.py --agent examples/idle_agent.py
"""
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    kind = message.get("message_type")
    if kind == "initialize":
        targets = message["payload"]["targets"]["rows"]
        print(f"idle agent: {len(targets)} targets received", file=sys.stderr, flush=True)
    elif kind == "decision_request":
        print(json.dumps({
            "protocol_version": message["protocol_version"],
            "message_type": "decision_response",
            "decision_sequence": message["decision_sequence"],
            "action": "finish",
            "reason": "idle agent: observe nothing",
        }), flush=True)
    elif kind == "finish":
        print("idle agent: finish " + message["payload"]["termination_reason"], file=sys.stderr, flush=True)
