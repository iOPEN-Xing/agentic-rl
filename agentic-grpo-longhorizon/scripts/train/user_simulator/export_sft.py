#!/usr/bin/env python3
"""Export accepted teacher generations as drop-in user-simulator chat SFT."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Optional eval JSONL; historical trial 07 is held out from train.",
    )
    parser.add_argument(
        "--stop-repeat",
        type=int,
        default=3,
        help="Materialize terminal examples N times to counter the ~13%% deduplicated rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_repeat < 1:
        raise ValueError("--stop-repeat must be >= 1")
    rows = _read_jsonl(args.input)
    rejected = [row for row in rows if row.get("status") != "accepted"]
    if rejected:
        raise RuntimeError(
            f"refusing export: {len(rejected)} rows are quarantined or rejected"
        )

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        record = row["sft_record"]
        is_eval = bool(
            args.eval_output and re.search(r"-trial07-", str(row.get("case_id", "")))
        )
        repeat = (
            1
            if is_eval
            else args.stop_repeat if record["metadata"]["is_over"] else 1
        )
        for replica in range(repeat):
            value = json.loads(json.dumps(record, ensure_ascii=False))
            value["metadata"]["replica"] = replica
            value["metadata"]["sample_weight_reason"] = (
                "terminal_boundary_oversample" if repeat > 1 else "natural"
            )
            (eval_rows if is_eval else train_rows).append(value)

    if args.eval_output and not eval_rows:
        raise RuntimeError("--eval-output requested but no trial07 cases were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.eval_output:
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        with args.eval_output.open("w", encoding="utf-8") as handle:
            for row in eval_rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

    train_counts = Counter(
        "stop" if row["metadata"]["is_over"] else "continue" for row in train_rows
    )
    eval_counts = Counter(
        "stop" if row["metadata"]["is_over"] else "continue" for row in eval_rows
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_decision_counts_after_oversampling": dict(train_counts),
                "eval_decision_counts_natural": dict(eval_counts),
                "output": str(args.output.resolve()),
                "eval_output": str(args.eval_output.resolve()) if args.eval_output else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
