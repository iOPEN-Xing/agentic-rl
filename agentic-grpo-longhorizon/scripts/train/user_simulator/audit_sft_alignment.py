#!/usr/bin/env python3
"""Audit User-Simulator SFT against the real multi-turn runtime contract.

This is intentionally tokenizer/model-free: it verifies every invariant that can be
proved from source data and code, and reports teacher-forcing distribution shift as an
explicit residual risk instead of pretending an offline JSONL check proves online
behavior.  It never calls a Teacher API, model server, trainer, or benchmark holdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.contracts import (  # noqa: E402
    STOP_TOKEN,
    TeacherDecision,
    validate_teacher_decision,
)


CONTROL_TOKEN_PATTERN = re.compile(
    r"<\|(?:im_start|im_end|endoftext)\|>|\[/?INST\]|```",
    re.IGNORECASE,
)
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}|Bearer\s+sk-", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(int((len(ordered) - 1) * fraction), len(ordered) - 1)]


def _runtime_prompt_builder(user_source: Path):
    """Extract the real method without importing tau-bench/litellm."""

    tree = ast.parse(user_source.read_text(encoding="utf-8"), filename=str(user_source))
    method: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMUserSimulationEnv":
            method = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef)
                    and child.name == "build_system_prompt"
                ),
                None,
            )
            break
    if method is None:
        raise RuntimeError("cannot find LLMUserSimulationEnv.build_system_prompt")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {"Optional": Optional}
    exec(compile(module, str(user_source), "exec"), namespace)
    return lambda instruction: namespace["build_system_prompt"](None, instruction)


def _conservative_token_upper_bound(messages: list[dict[str, Any]]) -> int:
    """Upper-bound content byte fallback plus generous Qwen chat-template overhead."""

    content_bytes = sum(len(str(item["content"]).encode("utf-8")) for item in messages)
    return content_bytes + 64 * len(messages) + 256


def _task_trial_turn(case_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"historical-airline-t(\d+)-trial(\d+)-u(\d+)", case_id)
    if not match:
        raise RuntimeError(f"unexpected historical case id: {case_id}")
    return tuple(int(value) for value in match.groups())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/cases/seen_train.jsonl",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/user_simulator_data/cases/benchmark_holdout_DO_NOT_GENERATE.jsonl"
        ),
    )
    parser.add_argument(
        "--audited",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs/user_simulator_data/full/v1.2_audited_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/user_simulator_data/full/v1.2_audited_v2.report.json"
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/train.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/eval.jsonl",
    )
    parser.add_argument(
        "--trl-train",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/train_trl.jsonl",
    )
    parser.add_argument(
        "--trl-eval",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/eval_trl.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs/train/sft/sft_user_simulator_qwen3_14b_lora.yaml"
        ),
    )
    parser.add_argument(
        "--runtime-user-source",
        type=Path,
        default=REPO_ROOT / "tau-bench/tau_bench/envs/user.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/user_simulator_data/sft/alignment_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _read_jsonl(args.cases)
    holdout = _read_jsonl(args.holdout)
    audited = _read_jsonl(args.audited)
    train = _read_jsonl(args.train)
    eval_rows = _read_jsonl(args.eval)
    trl_train = _read_jsonl(args.trl_train)
    trl_eval = _read_jsonl(args.trl_eval)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit_report = json.loads(args.audit_report.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    case_by_id = {str(row["case_id"]): row for row in cases}
    audited_by_id = {str(row["case_id"]): row for row in audited}
    if len(case_by_id) != len(cases) or len(audited_by_id) != len(audited):
        raise RuntimeError("duplicate natural case id")
    if set(case_by_id) != set(audited_by_id):
        raise RuntimeError("audited/source case populations differ")

    holdout_ids = {str(row["case_id"]) for row in holdout}
    if any(row.get("allowed_for_generation") is not False for row in holdout):
        raise RuntimeError("holdout contains a generation-enabled case")
    if holdout_ids & set(case_by_id):
        raise RuntimeError("seen/holdout case overlap")

    runtime_prompt = _runtime_prompt_builder(args.runtime_user_source)
    target_chars: list[int] = []
    message_counts: list[int] = []
    token_upper_bounds: list[int] = []
    task_counts: Counter[int] = Counter()
    trial_counts: Counter[int] = Counter()
    decisions: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    exact_reference_matches = 0
    secret_matches = 0

    for case_id, row in audited_by_id.items():
        case = case_by_id[case_id]
        if case.get("allowed_for_generation") is not True:
            raise RuntimeError(f"seen case is generation-disabled: {case_id}")
        if row.get("status") != "accepted" or row.get("quality_issues"):
            raise RuntimeError(f"audited row is not release-ready: {case_id}")
        record = row["sft_record"]
        messages = record["messages"]
        roles = [message.get("role") for message in messages]
        if roles[:2] != ["system", "user"] or roles[-1] != "assistant":
            raise RuntimeError(f"role boundary mismatch: {case_id}")
        if any(left == right for left, right in zip(roles, roles[1:])):
            raise RuntimeError(f"consecutive roles: {case_id}")
        if any(role not in {"system", "user", "assistant"} for role in roles):
            raise RuntimeError(f"unsupported role: {case_id}")
        if any(not str(message.get("content", "")) for message in messages):
            raise RuntimeError(f"empty message: {case_id}")
        if any(STOP_TOKEN in message["content"] for message in messages[1:-1]):
            raise RuntimeError(f"terminal marker appears in a live prefix: {case_id}")
        if messages[0]["content"] != runtime_prompt(case["scenario"]):
            raise RuntimeError(f"real runtime system prompt mismatch: {case_id}")
        if messages[:-1] != case["student_messages"]:
            raise RuntimeError(f"runtime prefix mismatch: {case_id}")

        decision = TeacherDecision.from_mapping(row["teacher_decision"])
        issues = validate_teacher_decision(
            decision,
            scenario=case["scenario"],
            observable_history=case["observable_history"],
            privileged_entities=case["privileged_reference"]["privileged_entities"],
        )
        if issues:
            raise RuntimeError(f"final target fails contract {issues}: {case_id}")
        target = messages[-1]["content"]
        if target != decision.response:
            raise RuntimeError(f"teacher/SFT target mismatch: {case_id}")
        if bool(record["metadata"]["is_over"]) != (target == STOP_TOKEN):
            raise RuntimeError(f"STOP metadata mismatch: {case_id}")
        if CONTROL_TOKEN_PATTERN.search(target):
            raise RuntimeError(f"template control token in target: {case_id}")
        if SECRET_PATTERN.search(json.dumps(row, ensure_ascii=False)):
            secret_matches += 1

        task_id, trial, _ = _task_trial_turn(case_id)
        task_counts[task_id] += 1
        trial_counts[trial] += 1
        decisions[decision.decision] += 1
        terminations[decision.termination_reason] += 1
        provenance[str(row["target_provenance"])] += 1
        target_chars.append(len(target))
        message_counts.append(len(messages))
        token_upper_bounds.append(_conservative_token_upper_bound(messages))
        exact_reference_matches += target == case["reference_response"]

    if secret_matches:
        raise RuntimeError("secret-shaped value found in audited data")

    train_ids = {str(row["case_id"]) for row in train}
    eval_ids = {str(row["case_id"]) for row in eval_rows}
    if train_ids & eval_ids:
        raise RuntimeError("train/eval case overlap")
    if (train_ids | eval_ids) != set(case_by_id):
        raise RuntimeError("train/eval do not cover the natural source exactly")
    if any("-trial07-" not in case_id for case_id in eval_ids):
        raise RuntimeError("eval contains a non-trial07 case")
    if any("-trial07-" in case_id for case_id in train_ids):
        raise RuntimeError("train contains a trial07 case")
    if (train_ids | eval_ids) & holdout_ids:
        raise RuntimeError("benchmark holdout leaked into SFT")

    for rows, trl_rows, name in (
        (train, trl_train, "train"),
        (eval_rows, trl_eval, "eval"),
    ):
        if len(rows) != len(trl_rows):
            raise RuntimeError(f"rich/TRL {name} row count mismatch")
        for rich, clean in zip(rows, trl_rows):
            if set(clean) != {"messages"} or clean["messages"] != rich["messages"]:
                raise RuntimeError(f"TRL {name} schema/content mismatch")

    natural_train_stop = sum(
        audited_by_id[case_id]["teacher_decision"]["decision"] == "stop"
        for case_id in train_ids
    )
    expected_train_rows = len(train_ids) + 2 * natural_train_stop
    if len(train) != expected_train_rows:
        raise RuntimeError("train STOP oversampling count mismatch")
    if len(eval_rows) != len(eval_ids):
        raise RuntimeError("eval must retain the natural distribution")

    if config["data"]["train_jsonl"] != "outputs/user_simulator_data/sft/train.jsonl":
        raise RuntimeError("training config points at the wrong train JSONL")
    if config["data"]["eval_jsonl"] != "outputs/user_simulator_data/sft/eval.jsonl":
        raise RuntimeError("training config points at the wrong eval JSONL")
    if config["data"].get("loss_mask_mode") != "last_assistant":
        raise RuntimeError("User-Simulator SFT must use last_assistant loss masking")
    if config["data"].get("chat_template_kwargs") != {"enable_thinking": False}:
        raise RuntimeError("training/runtime thinking contract mismatch")
    max_length = int(config["data"]["max_length"])
    if max(token_upper_bounds) >= max_length:
        raise RuntimeError("conservative serialized token upper bound reaches max_length")

    expected_hashes = {
        args.audited: manifest["source"]["generations_sha256"],
        args.train: manifest["outputs"]["rich_train_sha256"],
        args.eval: manifest["outputs"]["rich_eval_sha256"],
        args.trl_train: manifest["outputs"]["trl_train_sha256"],
        args.trl_eval: manifest["outputs"]["trl_eval_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"manifest hash mismatch: {path}")

    trajectory_groups: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for case_id in case_by_id:
        task_id, trial, turn = _task_trial_turn(case_id)
        trajectory_groups[(task_id, trial)].append((turn, case_id))
    linked_states = 0
    linked_exact_reconstruction = 0
    for values in trajectory_groups.values():
        values.sort()
        for (_, current_id), _ in zip(values, values[1:]):
            linked_states += 1
            linked_exact_reconstruction += (
                audited_by_id[current_id]["teacher_decision"]["response"]
                == case_by_id[current_id]["reference_response"]
            )

    report = {
        "status": "passed",
        "scope": (
            "offline runtime/data contract; online rollout quality is explicitly not proven"
        ),
        "runtime_alignment": {
            "real_runtime_prompt_exact": len(audited),
            "source_prefix_exact": len(audited),
            "role_alternation_exact": len(audited),
            "no_tool_roles": len(audited),
            "no_live_prefix_stop": len(audited),
            "last_assistant_targets": len(audited),
            "thinking_disabled": True,
            "loss_mask_mode": "last_assistant",
        },
        "population": {
            "natural_rows": len(audited),
            "seen_tasks": len(task_counts),
            "holdout_rows_audit_only": len(holdout),
            "holdout_in_sft": 0,
            "train_rows_stop_oversampled": len(train),
            "eval_rows_natural": len(eval_rows),
            "decisions": dict(decisions),
            "termination_reasons": dict(terminations),
            "target_provenance": dict(provenance),
            "task_count_range": [min(task_counts.values()), max(task_counts.values())],
            "trial_counts": dict(sorted(trial_counts.items())),
        },
        "quality_gates": {
            "final_contract_issues": 0,
            "observable_grounding_issues": 0,
            "privileged_entity_leaks": 0,
            "template_control_tokens": 0,
            "secret_matches": 0,
            "context_excluded_rows": audit_report["review_scope"]["context_excluded_rows"],
            "semantic_actions": audit_report["semantic_actions"],
        },
        "lengths": {
            "target_chars": {
                "min": min(target_chars),
                "median": median(target_chars),
                "p95": _percentile(target_chars, 0.95),
                "max": max(target_chars),
            },
            "messages_per_example": {
                "min": min(message_counts),
                "median": median(message_counts),
                "p95": _percentile(message_counts, 0.95),
                "max": max(message_counts),
            },
            "conservative_serialized_token_upper_bound": {
                "max": max(token_upper_bounds),
                "configured_max_length": max_length,
                "method": "UTF-8 content bytes + 64 tokens/message + 256 safety margin",
            },
        },
        "teacher_forcing_residual_risk": {
            "natural_targets_exactly_equal_historical_reference": exact_reference_matches,
            "linked_historical_states": linked_states,
            "teacher_target_exactly_reconstructs_next_historical_prefix": (
                linked_exact_reconstruction
            ),
            "interpretation": (
                "last-assistant one-step supervision is structurally correct, but most "
                "later prefixes remain historical/off-policy; frozen-policy online A/B "
                "and student-prefix rollout evaluation are still required"
            ),
        },
        "hashes": {
            "audited": _sha256(args.audited),
            "rich_train": _sha256(args.train),
            "rich_eval": _sha256(args.eval),
            "trl_train": _sha256(args.trl_train),
            "trl_eval": _sha256(args.trl_eval),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
