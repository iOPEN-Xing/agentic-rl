"""Teacher generation and deterministic quality-gate aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import re
from typing import Any, Iterable, Mapping

from .contracts import (
    ContractError,
    STOP_TOKEN,
    TeacherDecision,
    build_sft_record,
    parse_teacher_decision,
    validate_response_constraints,
    validate_teacher_decision,
)
from .prompts import PROMPT_VERSION, build_teacher_messages


_TERMINAL_COURTESY_PATTERN = re.compile(
    r"\b(?:that(?:'s| is) all|all (?:i|we) need(?:ed)?|"
    r"that(?:'s| is) what (?:i|we) need(?:ed)?|nothing else|goodbye)\b",
    flags=re.IGNORECASE,
)


def _normalize_terminal_courtesy(
    decision: TeacherDecision,
) -> tuple[TeacherDecision, list[str]]:
    """Map post-completion courtesy text to the runtime's exact STOP token.

    The teacher occasionally returns ``decision=continue`` with no unresolved goal and
    a closing utterance such as "No, that's all."  In the tau-bench runtime that extra
    turn is not a stylistic alternative: it is a delayed terminal transition.  We only
    canonicalize when both the structured state and the natural text independently say
    the dialogue is over.  A substantive response with an accidentally empty
    ``unresolved_goals`` list remains quarantined.
    """

    if (
        decision.decision == "continue"
        and not decision.unresolved_goals
        and _TERMINAL_COURTESY_PATTERN.search(decision.response)
        and "?" not in decision.response
        and not re.search(
            r"\b(?:but|however|except|before)\b", decision.response, re.IGNORECASE
        )
    ):
        normalized = replace(
            decision,
            decision="stop",
            is_over=True,
            termination_reason="goal_satisfied",
            goal_status="satisfied",
            communication_status="complete",
            response=STOP_TOKEN,
            resolved_goals=(
                decision.resolved_goals
                if decision.resolved_goals
                else ("all observable scenario goals",)
            ),
            unresolved_goals=(),
        )
        return normalized, ["terminal_courtesy_to_stop"]
    return decision, []


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
        decision, normalizations = _normalize_terminal_courtesy(decision)
        privileged = case.get("privileged_reference", {})
        issues = validate_teacher_decision(
            decision,
            scenario=str(case.get("scenario", "")),
            observable_history=case.get("observable_history", []),
            privileged_entities=privileged.get("privileged_entities", []),
        )
        issues.extend(
            validate_response_constraints(
                decision,
                case.get("response_constraints"),
            )
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
            "normalizations": normalizations,
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
                "request_config": api_result.get("request_config", {}),
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
    export_gate_passed = bool(
        total
        and statuses["rejected"] == 0
        and statuses["quarantined"] == 0
        and leak_count == 0
        and accepted == total
    )
    accepted_rows = [row for row in records if row.get("status") == "accepted"]
    decisions = Counter(
        str(row.get("teacher_decision", {}).get("decision", "missing"))
        for row in accepted_rows
    )
    termination_reasons = Counter(
        str(row.get("teacher_decision", {}).get("termination_reason", "missing"))
        for row in accepted_rows
    )
    goal_statuses = Counter(
        str(row.get("teacher_decision", {}).get("goal_status", "missing"))
        for row in accepted_rows
    )
    communication_statuses = Counter(
        str(row.get("teacher_decision", {}).get("communication_status", "missing"))
        for row in accepted_rows
    )
    reference_matrix = Counter(
        f"{row.get('expected_decision', 'missing')}->{row.get('teacher_decision', {}).get('decision', 'missing')}"
        for row in accepted_rows
    )
    response_lengths = sorted(
        len(str(row.get("teacher_decision", {}).get("response", "")))
        for row in accepted_rows
    )

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        index = min(int((len(values) - 1) * fraction), len(values) - 1)
        return values[index]

    usage_totals: Counter[str] = Counter()
    all_attempt_prompt_tokens_estimate = 0
    for row in records:
        usage = row.get("api", {}).get("usage", {})
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                usage_totals[str(key)] += int(value)
        prompt_tokens = usage.get("prompt_tokens")
        generation_attempt = int(row.get("generation_attempt", 1))
        if isinstance(prompt_tokens, (int, float)) and generation_attempt > 0:
            # The request messages do not change between retries, so prompt token
            # count is invariant. Prior completion/cache usage is unavailable after
            # canonical compaction and must not be presented as an exact bill total.
            all_attempt_prompt_tokens_estimate += int(prompt_tokens) * generation_attempt
    request_configs = Counter(
        json.dumps(
            row.get("api", {}).get("request_config", {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in records
        if row.get("api")
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
        "accepted_decision_counts": dict(decisions),
        "accepted_termination_reason_counts": dict(termination_reasons),
        "accepted_goal_status_counts": dict(goal_statuses),
        "accepted_communication_status_counts": dict(communication_statuses),
        "historical_reference_to_teacher_decision": dict(reference_matrix),
        "accepted_task_ids": sorted(
            {int(row["task_id"]) for row in accepted_rows if "task_id" in row}
        ),
        "accepted_response_chars": {
            "min": response_lengths[0] if response_lengths else 0,
            "median": percentile(response_lengths, 0.5),
            "p95": percentile(response_lengths, 0.95),
            "max": response_lengths[-1] if response_lengths else 0,
        },
        "api_usage_totals": dict(usage_totals),
        "api_usage_scope": (
            "canonical final response per case; excludes completion/cache usage from "
            "discarded retries"
        ),
        "all_attempt_prompt_tokens_estimate": all_attempt_prompt_tokens_estimate,
        "all_attempt_total_tokens_lower_bound": (
            all_attempt_prompt_tokens_estimate + usage_totals["completion_tokens"]
        ),
        "api_request_config_counts": dict(request_configs),
        "contract_gate_passed": contract_gate_passed,
        "export_gate_passed": export_gate_passed,
        "quality_gate_passed": quality_gate_passed,
    }
