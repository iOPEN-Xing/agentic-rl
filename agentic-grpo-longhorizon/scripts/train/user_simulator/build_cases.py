#!/usr/bin/env python3
"""Build curated pilot and leakage-safe τ-bench airline generation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.case_builder import (  # noqa: E402
    build_cases_from_historical_rows,
    load_historical_rows,
    prepare_curated_case,
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = json.dumps(
            {
                "scenario": case["scenario"],
                "observable_history": case["observable_history"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        unique.setdefault(hashlib.sha256(key.encode("utf-8")).hexdigest(), case)
    return list(unique.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical",
        type=Path,
        default=REPO_ROOT / "tau-bench/historical_trajectories/sonnet-35-new-airline.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "experiments/sft_collect_airline/split.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=PROJECT_ROOT / "configs/train/user_simulator/pilot_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text())
    seen_ids = {int(value) for value in split["seen_task_ids"]}
    unseen_ids = {int(value) for value in split["unseen_task_ids"]}

    pilot_values = json.loads(args.pilot_config.read_text())
    pilot_cases = [prepare_curated_case(value) for value in pilot_values]
    historical = build_cases_from_historical_rows(load_historical_rows(args.historical))
    historical = _deduplicate(historical)
    seen_cases = [case for case in historical if case["task_id"] in seen_ids]
    unseen_cases = [case for case in historical if case["task_id"] in unseen_ids]
    for case in seen_cases:
        case["quality_gate"] = False
        case["allowed_for_generation"] = True
        case["split"] = "seen_train"
    for case in unseen_cases:
        case["quality_gate"] = False
        case["allowed_for_generation"] = False
        case["split"] = "benchmark_holdout"
    for case in pilot_cases:
        if case["task_id"] not in seen_ids:
            raise ValueError(
                f"pilot case {case['case_id']} uses benchmark holdout task {case['task_id']}"
            )
        case["allowed_for_generation"] = True
        case["split"] = "curated_pilot"

    output_paths = {
        "pilot": args.output_dir / "pilot.jsonl",
        "seen_train": args.output_dir / "seen_train.jsonl",
        "benchmark_holdout": args.output_dir / "benchmark_holdout_DO_NOT_GENERATE.jsonl",
    }
    counts = {
        "pilot": _write_jsonl(output_paths["pilot"], pilot_cases),
        "seen_train": _write_jsonl(output_paths["seen_train"], seen_cases),
        "benchmark_holdout": _write_jsonl(
            output_paths["benchmark_holdout"], unseen_cases
        ),
    }
    decision_counts = {
        label: dict(Counter(case["expected_decision"] for case in cases))
        for label, cases in {
            "pilot": pilot_cases,
            "seen_train_historical_reference": seen_cases,
            "benchmark_holdout_historical_reference": unseen_cases,
        }.items()
    }
    manifest = {
        "version": "tau-airline-usim-cases-v2",
        "inputs": {
            "historical": str(args.historical.resolve()),
            "historical_sha256": _sha256(args.historical),
            "split": str(args.split.resolve()),
            "split_sha256": _sha256(args.split),
            "pilot_config": str(args.pilot_config.resolve()),
            "pilot_config_sha256": _sha256(args.pilot_config),
        },
        "counts": counts,
        "reference_decision_counts": decision_counts,
        "historical_reference_context_cuts": {
            label: [
                {
                    "case_id": case["case_id"],
                    "issues": case.get("reference_grounding_issues", []),
                }
                for case in cases
                if case.get("reference_grounding_issues")
            ]
            for label, cases in {
                "seen_train": seen_cases,
                "benchmark_holdout": unseen_cases,
            }.items()
        },
        "seen_task_ids": sorted(seen_ids),
        "benchmark_holdout_task_ids": sorted(unseen_ids),
        "leakage_rule": (
            "benchmark_holdout_DO_NOT_GENERATE.jsonl is audit-only and must never be sent "
            "to the teacher or used for user-simulator fine-tuning"
        ),
        "context_cut_rule": (
            "retain the current repairable state, then exclude every downstream state "
            "whose historical Customer prefix would contain an unsupported hard fact"
        ),
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items()},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
