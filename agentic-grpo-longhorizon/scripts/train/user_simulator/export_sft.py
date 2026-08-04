#!/usr/bin/env python3
"""Export audited generations as runtime-aligned chat SFT and clean TRL JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.case_builder import build_runtime_system_prompt  # noqa: E402
from src.user_simulator_data.contracts import STOP_TOKEN  # noqa: E402
from src.user_simulator_data.prompts import PROMPT_VERSION  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


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


def _default_trl_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_trl{path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="Original seen-case JSONL; when supplied, prefix/prompt alignment is exact-checked.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Optional eval JSONL; historical trial 07 is held out from train.",
    )
    parser.add_argument(
        "--trl-output",
        type=Path,
        default=None,
        help="Clean TRL conversational JSONL. Defaults to <output_stem>_trl.jsonl.",
    )
    parser.add_argument(
        "--trl-eval-output",
        type=Path,
        default=None,
        help="Clean TRL eval JSONL. Defaults beside --eval-output.",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument(
        "--stop-repeat",
        type=int,
        default=3,
        help="Materialize terminal examples N times in train; eval stays natural.",
    )
    return parser.parse_args()


def _validate_record(row: dict[str, Any]) -> None:
    case_id = str(row.get("case_id", "<missing>"))
    if row.get("status") != "accepted":
        raise RuntimeError(f"case is not accepted: {case_id}")
    if row.get("quality_issues"):
        raise RuntimeError(f"accepted case still has quality issues: {case_id}")
    if row.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError(f"top-level prompt version mismatch: {case_id}")
    record = row.get("sft_record")
    if not isinstance(record, dict):
        raise RuntimeError(f"case has no sft_record: {case_id}")
    if str(record.get("case_id", "")) != case_id:
        raise RuntimeError(f"generation/SFT case_id mismatch: {case_id}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise RuntimeError(f"case has invalid messages: {case_id}")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not str(message.get("content", ""))
        for message in messages
    ):
        raise RuntimeError(f"user-simulator record contains an unsupported role: {case_id}")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise RuntimeError(f"case must start system,user: {case_id}")
    if messages[-1].get("role") != "assistant":
        raise RuntimeError(f"case target must be the last assistant turn: {case_id}")
    if any(
        messages[index].get("role") == messages[index - 1].get("role")
        for index in range(2, len(messages))
    ):
        raise RuntimeError(f"user-simulator record contains consecutive roles: {case_id}")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(f"case metadata must be an object: {case_id}")
    target = str(messages[-1].get("content", ""))
    if not target:
        raise RuntimeError(f"case has an empty assistant target: {case_id}")
    if bool(metadata.get("is_over")) != (target == STOP_TOKEN):
        raise RuntimeError(f"STOP/is_over mismatch: {case_id}")
    if metadata.get("prompt_version") != PROMPT_VERSION:
        raise RuntimeError(
            f"prompt version mismatch for {case_id}: "
            f"{metadata.get('prompt_version')} != {PROMPT_VERSION}"
        )
    teacher_response = row.get("teacher_decision", {}).get("response")
    if target != teacher_response:
        raise RuntimeError(f"student target differs from teacher response: {case_id}")


def _validate_case_alignment(
    rows: list[dict[str, Any]], source_cases: list[dict[str, Any]]
) -> dict[str, int]:
    source_by_id = {str(case["case_id"]): case for case in source_cases}
    if len(source_by_id) != len(source_cases):
        raise RuntimeError("source cases contain duplicate case_id values")
    generated_ids = {str(row["case_id"]) for row in rows}
    source_ids = set(source_by_id)
    if generated_ids != source_ids:
        missing = sorted(source_ids - generated_ids)[:5]
        unexpected = sorted(generated_ids - source_ids)[:5]
        raise RuntimeError(
            f"generated/source case sets differ: missing={missing}, unexpected={unexpected}"
        )

    prompt_checks = 0
    prefix_checks = 0
    for row in rows:
        case_id = str(row["case_id"])
        source = source_by_id[case_id]
        record = row["sft_record"]
        messages = record["messages"]
        expected_system = build_runtime_system_prompt(str(source["scenario"]))
        if messages[0].get("content") != expected_system:
            raise RuntimeError(f"runtime system prompt mismatch: {case_id}")
        prompt_checks += 1
        if messages[:-1] != source["student_messages"]:
            raise RuntimeError(f"student context prefix differs from source case: {case_id}")
        prefix_checks += 1
        if int(record.get("task_id")) != int(source["task_id"]):
            raise RuntimeError(f"task_id mismatch: {case_id}")
    return {
        "runtime_system_prompt_checks": prompt_checks,
        "student_context_prefix_checks": prefix_checks,
    }


def main() -> None:
    args = parse_args()
    if args.stop_repeat < 1:
        raise ValueError("--stop-repeat must be >= 1")
    if args.trl_eval_output and not args.eval_output:
        raise ValueError("--trl-eval-output requires --eval-output")

    rows = _read_jsonl(args.input)
    if not rows:
        raise RuntimeError("input generation JSONL is empty")
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if any(not case_id for case_id in case_ids):
        raise RuntimeError("input contains a missing case_id")
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("input contains duplicate case_id values")
    for row in rows:
        _validate_record(row)

    alignment = {
        "runtime_system_prompt_checks": 0,
        "student_context_prefix_checks": 0,
    }
    if args.cases:
        alignment = _validate_case_alignment(rows, _read_jsonl(args.cases))

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    input_decisions: Counter[str] = Counter()
    target_provenance_counts: Counter[str] = Counter()
    semantic_audit_action_counts: Counter[str] = Counter()
    for row in rows:
        record = row["sft_record"]
        decision = "stop" if record["metadata"]["is_over"] else "continue"
        input_decisions[decision] += 1
        target_provenance_counts[
            str(record["metadata"].get("target_provenance", "missing"))
        ] += 1
        audit_action = record["metadata"].get("semantic_audit_action")
        if audit_action:
            semantic_audit_action_counts[str(audit_action)] += 1
        is_eval = bool(
            args.eval_output and re.search(r"-trial07-", str(row.get("case_id", "")))
        )
        repeat = 1 if is_eval else args.stop_repeat if decision == "stop" else 1
        for replica in range(repeat):
            value = json.loads(json.dumps(record, ensure_ascii=False))
            value["metadata"]["replica"] = replica
            value["metadata"]["sample_weight_reason"] = (
                "terminal_boundary_oversample" if repeat > 1 else "natural"
            )
            (eval_rows if is_eval else train_rows).append(value)

    if args.eval_output and not eval_rows:
        raise RuntimeError("--eval-output requested but no trial07 cases were found")
    train_ids = {row["case_id"] for row in train_rows}
    eval_ids = {row["case_id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise RuntimeError("train/eval case_id leakage detected")

    trl_output = args.trl_output or _default_trl_path(args.output)
    trl_eval_output = (
        args.trl_eval_output
        or (_default_trl_path(args.eval_output) if args.eval_output else None)
    )
    output_paths = [args.output, trl_output]
    if args.eval_output:
        output_paths.extend([args.eval_output, trl_eval_output])
    if len({path.resolve() for path in output_paths if path is not None}) != len(
        [path for path in output_paths if path is not None]
    ):
        raise RuntimeError("rich and TRL output paths must be distinct")
    protected_inputs = {args.input.resolve()}
    if args.cases:
        protected_inputs.add(args.cases.resolve())
    if protected_inputs & {path.resolve() for path in output_paths if path is not None}:
        raise RuntimeError("an export path would overwrite a source input")

    _write_jsonl(args.output, train_rows)
    if args.eval_output:
        _write_jsonl(args.eval_output, eval_rows)
    _write_jsonl(trl_output, ({"messages": row["messages"]} for row in train_rows))
    if trl_eval_output:
        _write_jsonl(
            trl_eval_output,
            ({"messages": row["messages"]} for row in eval_rows),
        )

    train_counts = Counter(
        "stop" if row["metadata"]["is_over"] else "continue" for row in train_rows
    )
    eval_counts = Counter(
        "stop" if row["metadata"]["is_over"] else "continue" for row in eval_rows
    )
    manifest_path = args.manifest_output or args.output.with_suffix(".manifest.json")
    if manifest_path.resolve() in protected_inputs or manifest_path.resolve() in {
        path.resolve() for path in output_paths if path is not None
    }:
        raise RuntimeError("manifest path must be distinct from all inputs and JSONL outputs")
    manifest = {
        "format_version": "tau-airline-user-simulator-sft-v1",
        "source": {
            "generations": str(args.input.resolve()),
            "generations_sha256": _sha256(args.input),
            "cases": str(args.cases.resolve()) if args.cases else None,
            "cases_sha256": _sha256(args.cases) if args.cases else None,
            "prompt_version": PROMPT_VERSION,
        },
        "alignment": {
            **alignment,
            "generated_case_ids_unique": len(case_ids),
            "train_eval_case_overlap": 0,
            "runtime_roles": {
                "system": "runtime LLMUserSimulationEnv system prompt",
                "user": "Agent-visible text passed to user simulator",
                "assistant": "simulated customer reply / exact STOP target",
            },
            "training_loss_mask_mode": "last_assistant",
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "counts": {
            "accepted_input_rows": len(rows),
            "input_decisions_natural": dict(input_decisions),
            "target_provenance": dict(target_provenance_counts),
            "semantic_audit_actions": dict(semantic_audit_action_counts),
            "train_rows_after_stop_oversampling": len(train_rows),
            "train_decisions_after_oversampling": dict(train_counts),
            "eval_rows_natural": len(eval_rows),
            "eval_decisions_natural": dict(eval_counts),
            "stop_repeat_train_only": args.stop_repeat,
        },
        "outputs": {
            "rich_train_jsonl": str(args.output.resolve()),
            "rich_train_sha256": _sha256(args.output),
            "rich_eval_jsonl": str(args.eval_output.resolve()) if args.eval_output else None,
            "rich_eval_sha256": _sha256(args.eval_output) if args.eval_output else None,
            "trl_train_jsonl": str(trl_output.resolve()),
            "trl_train_sha256": _sha256(trl_output),
            "trl_eval_jsonl": str(trl_eval_output.resolve()) if trl_eval_output else None,
            "trl_eval_sha256": _sha256(trl_eval_output) if trl_eval_output else None,
        },
        "trl_contract": {
            "dataset_loader": "load_dataset('json', data_files=...)",
            "schema": {"messages": "list[{role: str, content: str}]"},
            "conversational_format": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {**manifest, "manifest": str(manifest_path.resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
