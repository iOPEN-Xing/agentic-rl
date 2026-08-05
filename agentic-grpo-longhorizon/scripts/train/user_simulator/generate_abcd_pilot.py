#!/usr/bin/env python3
"""Generate a small, resume-safe ABCD User-Simulator pilot with DeepSeek."""

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

from src.user_simulator_data.abcd_adapter import (  # noqa: E402
    ABCD_PROMPT_VERSION,
    generate_abcd_one,
    summarize_abcd_generations,
)
from src.user_simulator_data.deepseek_client import DeepSeekClient  # noqa: E402


DEFAULT_DIR = PROJECT_ROOT / "outputs/user_simulator_data/abcd_adapter/pilot"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("external_train_only") is not True:
                raise RuntimeError(f"non-train-only ABCD row at {path}:{line_number}")
            if row.get("expected_decision") != "continue":
                raise RuntimeError(f"non-CONTINUE ABCD row at {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "source_cases.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "v0.1.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_DIR / "v0.1.report.json")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-case-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_case_attempts < 1:
        raise ValueError("--workers and --max-case-attempts must be >= 1")
    cases = _read_jsonl(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("ABCD pilot input is empty")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("ABCD pilot input contains duplicate case IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = set(case_ids)
    completed: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    if args.output.exists():
        for row in _read_jsonl(args.output):
            case_id = str(row["case_id"])
            if case_id not in expected_ids:
                raise RuntimeError(f"resume output has unexpected case: {case_id}")
            if row.get("prompt_version") != ABCD_PROMPT_VERSION:
                raise RuntimeError(f"resume prompt version mismatch: {case_id}")
            attempts[case_id] = max(attempts[case_id], int(row.get("generation_attempt", 1)))
            completed[case_id] = row

    def pending_cases() -> list[dict[str, Any]]:
        return [
            case
            for case in cases
            if completed.get(case["case_id"], {}).get("status") != "accepted"
            and attempts[case["case_id"]] < args.max_case_attempts
        ]

    pending = pending_cases()
    client = DeepSeekClient.from_environment(model=args.model) if pending else None
    lock = Lock()

    def persist(row: dict[str, Any], attempt: int) -> None:
        with lock:
            row["generation_attempt"] = attempt
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            completed[str(row["case_id"])] = row
            attempts[str(row["case_id"])] = attempt

    round_index = 0
    while pending:
        round_index += 1
        print(f"[round {round_index}] generating {len(pending)} ABCD pilot cases")
        round_attempts = {
            str(case["case_id"]): attempts[str(case["case_id"])] + 1 for case in pending
        }
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_abcd_one, client, case): case for case in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                case_id = str(case["case_id"])
                row = future.result()
                persist(row, round_attempts[case_id])
                print(
                    f"[{index}/{len(pending)}] {case_id}: {row['status']} "
                    f"(attempt {round_attempts[case_id]})"
                )
        pending = pending_cases()

    ordered = [completed[case_id] for case_id in case_ids]
    _atomic_jsonl(args.output, ordered)
    report = summarize_abcd_generations(ordered)
    usage: Counter[str] = Counter()
    for row in ordered:
        for key, value in row.get("api", {}).get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[str(key)] += int(value)
    report.update(
        {
            "model": args.model,
            "prompt_version": ABCD_PROMPT_VERSION,
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "generation_attempts": sum(attempts.values()),
            "retried_cases": sum(value > 1 for value in attempts.values()),
            "api_usage_final_rows": dict(usage),
        }
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["deterministic_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
