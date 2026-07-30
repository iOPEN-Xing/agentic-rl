#!/usr/bin/env python3
"""Validate environment/judge correlation from a JSON list of reward records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.envs.reward_fusion import reward_consistency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    environment = [float(record["environment_reward"]) for record in records]
    judge = [float(record["judge_reward"]) for record in records]
    result = reward_consistency(
        environment,
        judge,
        warning_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(result["conflict"]))


if __name__ == "__main__":
    raise SystemExit(main())
