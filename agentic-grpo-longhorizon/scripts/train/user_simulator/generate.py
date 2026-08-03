#!/usr/bin/env python3
"""Generate pilot or full user-simulator labels with DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.deepseek_client import DeepSeekClient  # noqa: E402
from src.user_simulator_data.generation import (  # noqa: E402
    generate_one,
    replicate_cases,
    summarize_generations,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("allowed_for_generation") is False:
                raise RuntimeError(
                    f"refusing audit-only/holdout case at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--samples-per-case",
        type=int,
        default=1,
        help="Use 3 for the pilot stability gate; keep 1 for full generation.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _read_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]
    cases = replicate_cases(cases, samples_per_case=args.samples_per_case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_suffix(".report.json")

    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists() and args.no_resume:
        raise RuntimeError(
            f"refusing to append duplicate rows to existing output: {args.output}; "
            "choose a new output path or omit --no-resume"
        )
    if args.output.exists() and not args.no_resume:
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[row["case_id"]] = row
    pending = [case for case in cases if case["case_id"] not in completed]

    client = DeepSeekClient.from_environment(model=args.model)
    lock = Lock()

    def persist(row: dict[str, Any]) -> None:
        with lock:
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            completed[row["case_id"]] = row

    if args.workers <= 1:
        for index, case in enumerate(pending, 1):
            row = generate_one(client, case)
            persist(row)
            print(f"[{index}/{len(pending)}] {case['case_id']}: {row['status']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_one, client, case): case for case in pending}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                row = future.result()
                persist(row)
                print(f"[{index}/{len(pending)}] {case['case_id']}: {row['status']}")

    ordered_rows = [completed[case["case_id"]] for case in cases]
    report = summarize_generations(ordered_rows)
    report.update(
        {
            "model": args.model,
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["quality_gate_passed"] and any(
        case.get("quality_gate") for case in cases
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
