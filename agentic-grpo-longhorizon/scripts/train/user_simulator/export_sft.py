#!/usr/bin/env python3
"""Export accepted teacher generations as drop-in user-simulator chat SFT."""

from __future__ import annotations

import argparse
import json
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

    exported: list[dict[str, Any]] = []
    for row in rows:
        record = row["sft_record"]
        repeat = args.stop_repeat if record["metadata"]["is_over"] else 1
        for replica in range(repeat):
            value = json.loads(json.dumps(record, ensure_ascii=False))
            value["metadata"]["replica"] = replica
            value["metadata"]["sample_weight_reason"] = (
                "terminal_boundary_oversample" if repeat > 1 else "natural"
            )
            exported.append(value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter(
        "stop" if row["metadata"]["is_over"] else "continue" for row in exported
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "exported_rows": len(exported),
                "decision_counts_after_oversampling": dict(counts),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
