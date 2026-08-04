#!/usr/bin/env python3
"""Generate pilot or full user-simulator labels with DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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
from src.user_simulator_data.prompts import PROMPT_VERSION  # noqa: E402


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
    parser.add_argument(
        "--max-case-attempts",
        type=int,
        default=3,
        help="Retry rejected/quarantined cases up to this many total attempts.",
    )
    return parser.parse_args()


def _write_canonical_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically compact retries to exactly one final row per input case."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_case_attempts < 1:
        raise ValueError("--max-case-attempts must be >= 1")
    cases = _read_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]
    cases = replicate_cases(cases, samples_per_case=args.samples_per_case)
    case_ids = [str(case["case_id"]) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("input contains duplicate case_id values")
    input_case_ids = set(case_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_suffix(".report.json")

    completed: dict[str, dict[str, Any]] = {}
    attempt_counts: Counter[str] = Counter()
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
                    case_id = str(row["case_id"])
                    if case_id not in input_case_ids:
                        raise RuntimeError(
                            f"resume output contains case_id absent from input: {case_id}"
                        )
                    if row.get("prompt_version") != PROMPT_VERSION:
                        raise RuntimeError(
                            f"resume output prompt version mismatch for {case_id}: "
                            f"{row.get('prompt_version')} != {PROMPT_VERSION}"
                        )
                    generated_model = row.get("api", {}).get("model")
                    if generated_model and generated_model != args.model:
                        raise RuntimeError(
                            f"resume output model mismatch for {case_id}: "
                            f"{generated_model} != {args.model}"
                        )
                    attempt_counts[case_id] = max(
                        attempt_counts[case_id], int(row.get("generation_attempt", 1))
                    )
                    completed[case_id] = row

    def remaining_cases() -> list[dict[str, Any]]:
        return [
            case
            for case in cases
            if completed.get(case["case_id"], {}).get("status") != "accepted"
            and attempt_counts[case["case_id"]] < args.max_case_attempts
        ]

    pending = remaining_cases()
    client = DeepSeekClient.from_environment(model=args.model) if pending else None
    lock = Lock()

    def persist(row: dict[str, Any], attempt: int) -> None:
        with lock:
            row["generation_attempt"] = attempt
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            completed[row["case_id"]] = row
            attempt_counts[row["case_id"]] = attempt

    generation_round = 0
    while pending:
        generation_round += 1
        print(
            f"[round {generation_round}] generating {len(pending)} cases; "
            f"max_case_attempts={args.max_case_attempts}"
        )
        attempts = {
            case["case_id"]: attempt_counts[case["case_id"]] + 1 for case in pending
        }
        if args.workers <= 1:
            for index, case in enumerate(pending, 1):
                row = generate_one(client, case)
                persist(row, attempts[case["case_id"]])
                print(
                    f"[{index}/{len(pending)}] {case['case_id']}: {row['status']} "
                    f"(attempt {attempts[case['case_id']]})"
                )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(generate_one, client, case): case for case in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    case = futures[future]
                    row = future.result()
                    persist(row, attempts[case["case_id"]])
                    print(
                        f"[{index}/{len(pending)}] {case['case_id']}: {row['status']} "
                        f"(attempt {attempts[case['case_id']]})"
                    )
        pending = remaining_cases()

    ordered_rows = [completed[case["case_id"]] for case in cases]
    _write_canonical_jsonl(args.output, ordered_rows)
    report = summarize_generations(ordered_rows)
    report.update(
        {
            "model": args.model,
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "generation_attempts": sum(attempt_counts.values()),
            "retried_cases": sum(count > 1 for count in attempt_counts.values()),
            "max_generation_attempts": max(attempt_counts.values(), default=0),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_gate = (
        "quality_gate_passed"
        if any(case.get("quality_gate") for case in cases)
        else "export_gate_passed"
    )
    if not report[required_gate]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
