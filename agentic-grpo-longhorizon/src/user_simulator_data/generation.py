"""Teacher generation and deterministic quality-gate aggregation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .contracts import (
    ContractError,
    build_sft_record,
    parse_teacher_decision,
    validate_teacher_decision,
)
from .prompts import PROMPT_VERSION, build_teacher_messages


def replicate_cases(
    cases: Iterable[Mapping[str, Any]], *, samples_per_case: int
) -> list[dict[str, Any]]:
    """Create unique, resume-safe case IDs for stochastic pilot repeats."""

    if samples_per_case < 1:
        raise ValueError("samples_per_case must be >= 1")
    if samples_per_case == 1:
        return [dict(case) for case in cases]
    replicas: list[dict[str, Any]] = []
    for case in cases:
        base_case_id = str(case["case_id"])
        for replica in range(samples_per_case):
            value = dict(case)
            value["base_case_id"] = base_case_id
            value["replica"] = replica
            value["case_id"] = f"{base_case_id}-rep{replica:02d}"
            replicas.append(value)
    return replicas


def generate_one(client: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    """Generate, validate, and package one teacher-labeled case."""

    base = {
        "case_id": case["case_id"],
        "base_case_id": case.get("base_case_id", case["case_id"]),
        "replica": int(case.get("replica", 0)),
        "task_id": int(case["task_id"]),
        "source": case.get("source", "curated-pilot"),
        "quality_gate": bool(case.get("quality_gate", False)),
        "expected_decision": case.get("expected_decision"),
        "prompt_version": PROMPT_VERSION,
    }
    try:
        api_result = client.complete_json(build_teacher_messages(case))
        decision = parse_teacher_decision(api_result["content"])
        privileged = case.get("privileged_reference", {})
        issues = validate_teacher_decision(
            decision,
            scenario=str(case.get("scenario", "")),
            observable_history=case.get("observable_history", []),
            privileged_entities=privileged.get("privileged_entities", []),
        )
        expected = case.get("expected_decision")
        if case.get("quality_gate") and expected and decision.decision != expected:
            issues.append("expected_decision_mismatch")
        status = "accepted" if not issues else "quarantined"
        return {
            **base,
            "status": status,
            "quality_issues": sorted(set(issues)),
            "teacher_decision": decision.to_mapping(),
            "sft_record": build_sft_record(
                case,
                decision,
                prompt_version=PROMPT_VERSION,
                quality_issues=issues,
            ),
            "api": {
                "model": api_result.get("model"),
                "response_id": api_result.get("response_id"),
                "usage": api_result.get("usage", {}),
            },
        }
    except (ContractError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        return {
            **base,
            "status": "rejected",
            "quality_issues": [f"generation_error:{type(exc).__name__}"],
            "error": str(exc),
        }


def summarize_generations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pilot/full generation metrics with a conservative release gate."""

    records = list(rows)
    statuses = Counter(str(row.get("status", "missing")) for row in records)
    issues = Counter(
        issue for row in records for issue in row.get("quality_issues", [])
    )
    curated = [
        row
        for row in records
        if row.get("quality_gate") and row.get("expected_decision") is not None
    ]
    curated_matches = sum(
        row.get("teacher_decision", {}).get("decision")
        == row.get("expected_decision")
        for row in curated
    )
    curated_accuracy = curated_matches / len(curated) if curated else 0.0
    total = len(records)
    accepted = statuses["accepted"]
    leak_count = sum(
        count for issue, count in issues.items() if issue.startswith("privileged_entity_leak:")
    )
    quality_gate_passed = bool(
        total
        and curated
        and curated_accuracy == 1.0
        and statuses["rejected"] == 0
        and leak_count == 0
        and accepted == total
    )
    contract_gate_passed = bool(
        total
        and statuses["rejected"] == 0
        and leak_count == 0
        and accepted / total >= 0.98
    )
    return {
        "total": total,
        "status_counts": dict(statuses),
        "accepted_rate": accepted / total if total else 0.0,
        "quality_issue_counts": dict(issues),
        "curated_cases": len(curated),
        "curated_decision_matches": curated_matches,
        "curated_decision_accuracy": curated_accuracy,
        "privileged_entity_leaks": leak_count,
        "contract_gate_passed": contract_gate_passed,
        "quality_gate_passed": quality_gate_passed,
    }
