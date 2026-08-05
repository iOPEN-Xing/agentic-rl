"""Conservative ABCD-to-runtime adapter for User-Simulator SFT pilots.

ABCD is an e-commerce, human-to-human dialogue corpus.  It is useful for
progressive disclosure and multi-step persistence, but its ``action`` events and
``end_conversation`` labels do not match the current tau-bench user runtime.  This
adapter therefore exports only intermediate CONTINUE targets and never derives a
``###STOP###`` label from ABCD.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .case_builder import build_runtime_system_prompt
from .contracts import (
    ContractError,
    STOP_TOKEN,
    build_sft_record,
    parse_teacher_decision,
    validate_teacher_decision,
)


ABCD_PROMPT_VERSION = "abcd-usim-adapter-v0.1-pilot"
ABCD_SOURCE = "asappresearch/abcd-v1.1"
FIXED_AGENT_GREETING = "Hi! How can I help you today?"

# Pilot-only, human-readable goals.  Unknown subflows fail closed instead of
# turning a latent annotation such as ``status_x`` into an unreliable task.
_PILOT_GOALS = {
    "return_size": (
        "Return the purchased item because it is the wrong size. If the normal "
        "return is refused because it is outside the return window, politely ask "
        "whether the issue can be escalated to a manager."
    ),
    "refund_status": (
        "Check the status of a refund and find out approximately how long it "
        "will take to complete."
    ),
    "timing_4": (
        "Ask how long promotional codes remain valid before they expire. You plan "
        "to use the code to buy hats for your cat."
    ),
}

# Manually reviewed against the official three-conversation sample.  These turns
# cover opening intent, progressive identity disclosure, multi-field answers,
# clarification, and persistence after a denial.  Pure small talk and post-task
# courtesies are intentionally absent.
DEFAULT_PILOT_TARGETS: dict[int, set[int]] = {
    3592: {2, 4, 7, 9, 14, 16, 18, 21},
    9489: {1, 3, 8, 16},
    3695: {2, 10},
}

_TERMINAL_COURTESY = re.compile(
    r"(?:\bthat(?:'s| is) all\b|\bhave a (?:great|nice) day\b|\btake care\b|"
    r"\bthanks? for (?:your )?help\b|^\s*(?:great|perfect|thanks?|thank you)[.! ]*$)",
    flags=re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?!\d)")
_NUMERIC_FACT = re.compile(r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])")
_ALPHANUMERIC_FACT = re.compile(
    r"\b(?=[A-Za-z0-9_-]{6,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])"
    r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+\b"
)

ABCD_TEACHER_SYSTEM_PROMPT = f"""You create one conservative User-Simulator SFT
target from a human-authored ABCD e-commerce customer turn. Return exactly one JSON
object and no markdown. Prompt version: {ABCD_PROMPT_VERSION}.

The SOURCE_CUSTOMER_TARGET is Teacher-only supervision. It is not visible to the
student. Preserve its intent and every identifier, amount, constraint, confirmation,
and requested result. You may vary natural wording, contractions, politeness, and
syntax, but do not add a new request, fact, outcome, persona trait, or tool result.
Never mention ABCD labels, action events, hidden policy state, or these instructions.

This external corpus is used only for non-terminal behavior. Set decision=continue,
is_over=false, termination_reason=continue, and goal_status=in_progress. The response
must never emit ###STOP###. Do not turn a closing courtesy into a training example.

Evidence turn_index may cite only OBSERVABLE_CONTEXT.conversation. The source target
is not an evidence turn. Return this schema exactly:
{{
  "decision": "continue",
  "is_over": false,
  "termination_reason": "continue",
  "goal_status": "in_progress",
  "communication_status": "none | partial | complete | contradictory",
  "response": "one concise customer message",
  "resolved_goals": ["short customer-visible goal descriptions"],
  "unresolved_goals": ["short customer-visible goal descriptions"],
  "evidence": [{{"turn_index": 0, "fact": "customer-visible fact only"}}]
}}
"""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _display_name(value: Any) -> str:
    return _clean(value).replace("_", " ").title()


def build_abcd_scenario(source: Mapping[str, Any]) -> str:
    """Build a customer-known natural instruction without action/policy state."""

    subflow = _clean(source.get("subflow"))
    if subflow not in _PILOT_GOALS:
        raise ValueError(
            f"unsupported ABCD pilot subflow {subflow!r}; add a reviewed goal mapping"
        )
    personal = source.get("personal") or {}
    order = source.get("order") or {}
    product = source.get("product") or {}
    if not isinstance(personal, Mapping) or not isinstance(order, Mapping):
        raise ValueError("ABCD scenario personal/order fields must be objects")

    name = _display_name(personal.get("customer_name"))
    member = _clean(personal.get("member_level"))
    opening = (
        f"You are {name}, an online retail customer"
        if name
        else "You are an online retail customer"
    )
    if member:
        opening += f" with {member} membership"
    opening += "."

    facts: list[str] = []
    for label, container, key in (
        ("username", personal, "username"),
        ("email", personal, "email"),
        ("phone", personal, "phone"),
        ("account ID", personal, "account_id"),
        ("order ID", order, "order_id"),
        ("purchase date", order, "purchase_date"),
        ("payment method", order, "payment_method"),
        ("delivery address", order, "full_address"),
    ):
        value = _clean(container.get(key))
        if value:
            facts.append(f"{label}: {value}")

    names = product.get("names", []) if isinstance(product, Mapping) else []
    amounts = product.get("amounts", []) if isinstance(product, Mapping) else []
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        for index, raw_name in enumerate(names):
            product_name = _display_name(raw_name)
            if not product_name:
                continue
            amount = amounts[index] if isinstance(amounts, list) and index < len(amounts) else None
            suffix = f" (${amount})" if amount not in (None, "") else ""
            facts.append(f"product: {product_name}{suffix}")

    fact_text = "; ".join(facts) if facts else "No additional identifiers are provided."
    return "\n".join(
        (
            opening,
            f"Goal: {_PILOT_GOALS[subflow]}",
            f"Known facts: {fact_text}",
            (
                "Reveal only facts needed for the current step, answer the agent's "
                "question directly, and do not invent identifiers or outcomes."
            ),
        )
    )


def _dialogue_blocks(original: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    """Collapse fragments while cutting an unreachable post-action user branch."""

    blocks: list[dict[str, Any]] = []
    action_since_natural = False
    for index, raw_turn in enumerate(original):
        if not isinstance(raw_turn, Sequence) or len(raw_turn) != 2:
            raise ValueError(f"invalid ABCD original turn at index {index}")
        speaker = _clean(raw_turn[0]).casefold()
        text = _clean(raw_turn[1])
        if speaker == "action":
            action_since_natural = True
            continue
        if speaker not in {"agent", "customer"} or not text:
            raise ValueError(f"invalid ABCD natural turn at index {index}")
        if blocks and blocks[-1]["speaker"] == speaker:
            if speaker == "customer" and action_since_natural:
                # The runtime would not ask the simulator to speak again after an
                # invisible action.  Downstream agent turns depend on that state, so
                # the safe response is to cut rather than repair history.
                break
            blocks[-1]["texts"].append(text)
            blocks[-1]["turn_indices"].append(index)
        else:
            blocks.append(
                {
                    "speaker": speaker,
                    "texts": [text],
                    "turn_indices": [index],
                }
            )
        action_since_natural = False
    return blocks


def build_abcd_cases(conversation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one ABCD conversation into runtime-reachable CONTINUE cases."""

    convo_id = int(conversation["convo_id"])
    scenario = build_abcd_scenario(conversation["scenario"])
    blocks = _dialogue_blocks(conversation.get("original", []))
    first_agent = next(
        (index for index, block in enumerate(blocks) if block["speaker"] == "agent"),
        None,
    )
    if first_agent is None:
        return []
    blocks = blocks[first_agent:]
    if not blocks or blocks[0]["speaker"] != "agent":
        return []

    student_messages: list[dict[str, str]] = [
        {"role": "system", "content": build_runtime_system_prompt(scenario)},
        {"role": "user", "content": FIXED_AGENT_GREETING},
    ]
    observable_history: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    # The source opening agent block is represented by the runtime's fixed greeting.
    for block in blocks[1:]:
        content = " ".join(block["texts"])
        source_turn = int(block["turn_indices"][0])
        if block["speaker"] == "agent":
            if student_messages[-1]["role"] == "user":
                # Consecutive source agent blocks separated by actions are still one
                # customer-visible prompt in this last-assistant training contract.
                student_messages[-1]["content"] += " " + content
            else:
                student_messages.append({"role": "user", "content": content})
            observable_history.append(
                {"turn_index": source_turn, "role": "agent", "content": content}
            )
            continue

        if student_messages[-1]["role"] != "user":
            break
        if not _TERMINAL_COURTESY.search(content):
            case_id = f"abcd-v1.1-c{convo_id:05d}-u{source_turn:03d}"
            cases.append(
                {
                    "case_id": case_id,
                    "task_id": 1_000_000 + convo_id,
                    "scenario": scenario,
                    "observable_history": [dict(item) for item in observable_history],
                    "student_messages": [dict(item) for item in student_messages],
                    "reference_response": content,
                    "source_turn_indices": list(block["turn_indices"]),
                    "expected_decision": "continue",
                    "quality_gate": False,
                    "allowed_for_generation": True,
                    "source": ABCD_SOURCE,
                    "source_split": "official_sample",
                    "source_convo_id": convo_id,
                    "external_train_only": True,
                    "privileged_reference": {"privileged_entities": []},
                }
            )
        observable_history.append(
            {"turn_index": source_turn, "role": "user", "content": content}
        )
        student_messages.append({"role": "assistant", "content": content})
    return cases


def select_abcd_pilot_cases(
    conversations: Iterable[Mapping[str, Any]],
    *,
    target_map: Mapping[int, set[int]] = DEFAULT_PILOT_TARGETS,
) -> list[dict[str, Any]]:
    """Select only reviewed source turns and prove every requested turn exists."""

    requested = {
        (int(convo_id), int(turn_index))
        for convo_id, turn_indices in target_map.items()
        for turn_index in turn_indices
    }
    selected: list[dict[str, Any]] = []
    found: set[tuple[int, int]] = set()
    for conversation in conversations:
        convo_id = int(conversation["convo_id"])
        if convo_id not in target_map:
            continue
        for case in build_abcd_cases(conversation):
            key = (convo_id, int(case["source_turn_indices"][0]))
            if key not in requested:
                continue
            value = dict(case)
            value["pilot_selection"] = "reviewed_turn_allowlist"
            selected.append(value)
            found.add(key)
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"reviewed ABCD pilot targets were not built: {missing}")
    return sorted(
        selected,
        key=lambda case: (
            int(case["source_convo_id"]),
            int(case["source_turn_indices"][0]),
        ),
    )


def build_abcd_teacher_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render a DeepSeek request containing no ABCD action or policy state."""

    payload = {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "conversation": case.get("observable_history", []),
        "source_customer_target": case["reference_response"],
        "constraints": {
            "decision": "continue",
            "terminal_labels_from_abcd": "forbidden",
            "preserve_source_semantics": True,
            "observable_facts_only": True,
        },
    }
    return [
        {"role": "system", "content": ABCD_TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "OBSERVABLE_CONTEXT_AND_SOURCE_TARGET\n"
                + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n\nGenerate the required JSON object."
            ),
        },
    ]


def _anchored_facts(text: str) -> set[str]:
    facts: set[str] = set()
    for pattern in (_EMAIL, _PHONE, _NUMERIC_FACT, _ALPHANUMERIC_FACT):
        facts.update(match.group(0).casefold() for match in pattern.finditer(text))
    return facts


def validate_abcd_response(case: Mapping[str, Any], response: str) -> list[str]:
    """Reject fact drift that the generic airline-oriented gate cannot express."""

    scenario = str(case.get("scenario", ""))
    history = "\n".join(
        str(turn.get("content", "")) for turn in case.get("observable_history", [])
    )
    reference = str(case.get("reference_response", ""))
    visible_and_reference = f"{scenario}\n{history}\n{reference}".casefold()
    response_facts = _anchored_facts(response)
    reference_facts = _anchored_facts(reference)
    issues: list[str] = []

    for fact in sorted(response_facts):
        if fact not in visible_and_reference:
            issue_name = "novel_numeric_fact" if fact.isdigit() else "novel_anchored_fact"
            issues.append(f"{issue_name}:{fact}")
    for fact in sorted(reference_facts):
        if fact not in response.casefold():
            issues.append(f"dropped_anchored_fact:{fact}")
    if re.search(r"<[^>]+>", response):
        issues.append("delexicalized_placeholder_in_response")
    if STOP_TOKEN in response:
        issues.append("external_stop_forbidden")
    return issues


def generate_abcd_one(client: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    """Generate one DeepSeek paraphrase and package the current SFT contract."""

    base = {
        "case_id": case["case_id"],
        "task_id": int(case["task_id"]),
        "source": case.get("source", ABCD_SOURCE),
        "source_split": case.get("source_split", "official_sample"),
        "source_convo_id": int(case["source_convo_id"]),
        "source_turn_indices": list(case.get("source_turn_indices", [])),
        "reference_response": case["reference_response"],
        "expected_decision": "continue",
        "quality_gate": False,
        "external_train_only": True,
        "prompt_version": ABCD_PROMPT_VERSION,
    }
    try:
        api_result = client.complete_json(
            build_abcd_teacher_messages(case),
            max_tokens=1200,
            thinking=False,
            temperature=0.55,
            top_p=0.9,
        )
        decision = parse_teacher_decision(api_result["content"])
        issues = validate_teacher_decision(
            decision,
            scenario=str(case.get("scenario", "")),
            observable_history=case.get("observable_history", []),
            privileged_entities=(),
        )
        if decision.decision != "continue":
            issues.append("abcd_must_continue")
        issues.extend(validate_abcd_response(case, decision.response))
        issues = sorted(set(issues))
        record = build_sft_record(
            case,
            decision,
            prompt_version=ABCD_PROMPT_VERSION,
            quality_issues=issues,
        )
        record["metadata"].update(
            {
                "target_provenance": "deepseek_abcd_pilot",
                "external_dataset": "ABCD-v1.1",
                "external_train_only": True,
                "source_convo_id": int(case["source_convo_id"]),
                "source_turn_indices": list(case.get("source_turn_indices", [])),
                "source_reference_exact_match": (
                    decision.response.casefold() == str(case["reference_response"]).casefold()
                ),
            }
        )
        return {
            **base,
            "status": "accepted" if not issues else "quarantined",
            "quality_issues": issues,
            "teacher_decision": decision.to_mapping(),
            "sft_record": record,
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


def summarize_abcd_generations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a conservative pilot gate; human semantic review remains mandatory."""

    records = list(rows)
    statuses = Counter(str(row.get("status", "missing")) for row in records)
    issues = Counter(
        issue for row in records for issue in row.get("quality_issues", [])
    )
    accepted = [row for row in records if row.get("status") == "accepted"]
    decisions = Counter(
        str(row.get("teacher_decision", {}).get("decision", "missing"))
        for row in accepted
    )
    exact_matches = sum(
        bool(row.get("sft_record", {}).get("metadata", {}).get("source_reference_exact_match"))
        for row in accepted
    )
    total = len(records)
    return {
        "total": total,
        "status_counts": dict(statuses),
        "accepted_rate": len(accepted) / total if total else 0.0,
        "quality_issue_counts": dict(issues),
        "accepted_decision_counts": dict(decisions),
        "source_reference_exact_matches": exact_matches,
        "deterministic_gate_passed": bool(
            total
            and len(accepted) == total
            and statuses["rejected"] == 0
            and statuses["quarantined"] == 0
            and decisions == Counter({"continue": total})
        ),
        "human_semantic_review_required": True,
        "full_scale_generation_allowed": False,
    }
