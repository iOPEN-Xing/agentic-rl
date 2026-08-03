#!/usr/bin/env python3
"""Validate terminal-outcome/judge correlation from trajectory JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.envs.reward_fusion import reward_consistency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--environment-key", default="outcome_reward")
    parser.add_argument("--judge-key", default="judge_mean_score")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    paired_records = [
        record
        for record in records
        if record.get(args.environment_key) is not None
        and record.get(args.judge_key) is not None
    ]
    environment = [
        float(record[args.environment_key]) for record in paired_records
    ]
    judge = [float(record[args.judge_key]) for record in paired_records]
    result = reward_consistency(
        environment,
        judge,
        warning_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(result["conflict"]))


if __name__ == "__main__":
    raise SystemExit(main())
