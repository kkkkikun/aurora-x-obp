#!/usr/bin/env python3
import json
import sys
import time


for line in sys.stdin:
    payload = json.loads(line)
    if payload.get("message_type") == "initialize":
        continue
    time.sleep(10)
    print(
        json.dumps(
            {
                "protocol_version": payload["protocol_version"],
                "message_type": "decision_response",
                "decision_sequence": payload["decision_sequence"],
                "action": "wait",
                "reason": "slow fixture",
            }
        ),
        flush=True,
    )
