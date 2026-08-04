"""Strict contracts between a teacher model, QA, and user-simulator SFT.

The teacher emits rich JSON so we can audit the termination boundary.  The
student target remains plain text: a natural customer message or the exact
``###STOP###`` marker expected by the current τ-bench runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


STOP_TOKEN = "###STOP###"
DECISIONS = {"continue", "stop"}
TERMINATION_REASONS = {
    "continue",
    "goal_satisfied",
    "goal_failed",
    "cannot_continue",
}
GOAL_STATUSES = {"in_progress", "satisfied", "failed", "blocked", "unknown"}
COMMUNICATION_STATUSES = {"none", "partial", "complete", "contradictory"}

_LABELED_IDENTIFIER_CLAIM = re.compile(
    r"\b(?P<label>reservation|booking|confirmation|user|payment|certificate|"
    r"gift\s+card|credit\s+card)\s*(?:id|code)\s*(?:is|:)\s*"
    r"(?:it(?:'s| is)\s*)?[\"']?(?P<value>[*A-Za-z0-9][*A-Za-z0-9_-]{2,}|\.\.\.)",
    flags=re.IGNORECASE,
)
_FLIGHT_CODE = re.compile(r"\bHAT\d{2,4}\b", flags=re.IGNORECASE)
_USER_ID_IN_SCENARIO = re.compile(
    r"\byou are\s+([a-z][a-z0-9_]+)|"
    r"\buser id is\s+([a-z][a-z0-9_]+)|"
    r"\bwith id:\s*([a-z][a-z0-9_]+)",
    flags=re.IGNORECASE,
)
_NON_VALUE_IDENTIFIER_WORDS = {
    "in",
    "missing",
    "none",
    "not",
    "on",
    "the",
    "unavailable",
    "unknown",
}
_AIRPORTS = {
    "SFO": "San Francisco",
    "JFK": "New York",
    "LAX": "Los Angeles",
    "ORD": "Chicago",
    "DFW": "Dallas",
    "DEN": "Denver",
    "SEA": "Seattle",
    "ATL": "Atlanta",
    "MIA": "Miami",
    "BOS": "Boston",
    "PHX": "Phoenix",
    "IAH": "Houston",
    "LAS": "Las Vegas",
    "MCO": "Orlando",
    "EWR": "Newark",
    "CLT": "Charlotte",
    "MSP": "Minneapolis",
    "DTW": "Detroit",
    "PHL": "Philadelphia",
    "LGA": "LaGuardia",
}


def _scenario_user_ids(scenario: str) -> set[str]:
    values: set[str] = set()
    for match in _USER_ID_IN_SCENARIO.finditer(scenario):
        values.add(next(group for group in match.groups() if group).casefold())
    return values


def _agent_presented_as_example(token: str, agent_text: str) -> bool:
    """Return whether an Agent mentioned ``token`` only as an identifier example.

    A generic phrase such as ``would you like HAT123`` is not an example.  We require
    explicit exemplar language near the token so a real offered flight remains usable.
    """

    escaped = re.escape(token)
    return bool(
        re.search(
            rf"(?:for\s+example|e\.g\.|such\s+as|"
            rf"(?:code|identifier|\bid\b)[^\n]{{0,20}}\blike\b)"
            rf"[^\n]{{0,45}}[\"']?{escaped}\b",
            agent_text,
            flags=re.IGNORECASE,
        )
    )


def validate_observable_grounding_text(
    response: str,
    *,
    scenario: str = "",
    observable_history: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """Flag hard facts that a runtime User Simulator cannot legitimately know.

    The simulator observes the scenario plus natural-language conversation, not tool
    state.  Merely checking known privileged IDs is insufficient: a Teacher can invent
    a *new* placeholder such as ``ABC123`` that is absent from both observable and gold
    data.  This gate therefore validates typed identifier claims, flight codes, and
    explicit dates of birth against the actual observable prefix.  It deliberately does
    not reject derived arithmetic (for example ``$192 - $152 = $40``).
    """

    visible_source = f"{scenario}\n{_history_text(observable_history)}"
    visible_text = visible_source.casefold()
    agent_text = "\n".join(
        str(item.get("content", ""))
        for item in observable_history
        if str(item.get("role", "")).lower() == "agent"
    )
    scenario_user_ids = _scenario_user_ids(scenario)
    issues: list[str] = []

    for match in _LABELED_IDENTIFIER_CLAIM.finditer(response):
        label = re.sub(r"\s+", "_", match.group("label").casefold())
        value = match.group("value").strip()
        folded = value.casefold()
        if value == "...":
            issues.append(f"placeholder_identifier:{label}")
            continue
        if folded in _NON_VALUE_IDENTIFIER_WORDS:
            continue
        if label in {"reservation", "booking", "confirmation"} and (
            folded in scenario_user_ids
            or any(
                folded == user_id.rsplit("_", 1)[-1]
                for user_id in scenario_user_ids
            )
        ):
            issues.append(f"user_id_mislabeled_as_{label}_id:{value}")
            continue
        if folded not in visible_text:
            issues.append(f"unsupported_{label}_id:{value}")
            continue
        if label in {"reservation", "booking", "confirmation"} and _agent_presented_as_example(
            value, agent_text
        ):
            issues.append(f"agent_example_copied_as_{label}_id:{value}")

    if re.search(
        r"reservation id is the one i mentioned earlier", response, re.IGNORECASE
    ) and not any(
        re.search(
            r"(?:reservation|booking|confirmation)\s*(?:id|code)\s*(?:is|:)",
            str(item.get("content", "")),
            re.IGNORECASE,
        )
        for item in observable_history
        if str(item.get("role", "")).lower() == "user"
    ):
        issues.append("unsupported_prior_reservation_id_mention")

    for match in _FLIGHT_CODE.finditer(response):
        value = match.group(0)
        if value.casefold() not in visible_text:
            issues.append(f"unsupported_flight_code:{value}")
        elif _agent_presented_as_example(value, agent_text):
            issues.append(f"agent_example_copied_as_flight_code:{value}")

    # Airport code/city aliases are public equivalents (for example EWR/Newark), so
    # either side in the observable prefix is sufficient.  If neither side is present,
    # the response has supplied a route endpoint that the runtime simulator was never
    # told.  This catches route drift that typed booking-ID checks cannot see.
    for airport, city in _AIRPORTS.items():
        response_mentions_location = bool(
            re.search(rf"\b{re.escape(airport)}\b", response)
            or re.search(rf"\b{re.escape(city)}\b", response, re.IGNORECASE)
        )
        visible_mentions_location = bool(
            re.search(rf"\b{re.escape(airport)}\b", visible_source)
            or re.search(rf"\b{re.escape(city)}\b", visible_source, re.IGNORECASE)
        )
        if response_mentions_location and not visible_mentions_location:
            issues.append(f"unsupported_airport_or_city:{airport}")

    if re.search(r"date of birth|\bdob\b|born on", response, re.IGNORECASE):
        for year in re.findall(r"\b(?:19|20)\d{2}\b", response):
            if year not in visible_text:
                issues.append(f"unsupported_date_of_birth_year:{year}")

    if re.search(r"\[(?:date|id|number|value)\]", response, re.IGNORECASE):
        issues.append("template_placeholder_in_response")
    if re.search(
        r"\b(?:reservation|booking|confirmation|payment|user)\b[^\n]{0,60}"
        r"\b(?:id|code)\b[^\n]{0,60}\bis\s+\.\.\.",
        response,
        re.IGNORECASE,
    ):
        issues.append("placeholder_identifier:untyped")
    return sorted(set(issues))


class ContractError(ValueError):
    """The teacher response cannot be made safe by deterministic cleanup."""


@dataclass(frozen=True)
class Evidence:
    turn_index: int
    fact: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Evidence":
        if "turn_index" not in value or "fact" not in value:
            raise ContractError("evidence requires turn_index and fact")
        return cls(turn_index=int(value["turn_index"]), fact=str(value["fact"]).strip())


@dataclass(frozen=True)
class TeacherDecision:
    decision: str
    is_over: bool
    termination_reason: str
    goal_status: str
    communication_status: str
    response: str
    resolved_goals: tuple[str, ...]
    unresolved_goals: tuple[str, ...]
    evidence: tuple[Evidence, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherDecision":
        required = {
            "decision",
            "is_over",
            "termination_reason",
            "goal_status",
            "communication_status",
            "response",
            "resolved_goals",
            "unresolved_goals",
            "evidence",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ContractError(f"missing teacher fields: {', '.join(missing)}")
        if not isinstance(value["is_over"], bool):
            raise ContractError("is_over must be a JSON boolean")
        for field_name in ("resolved_goals", "unresolved_goals", "evidence"):
            if not isinstance(value[field_name], list):
                raise ContractError(f"{field_name} must be a JSON array")

        response = re.sub(r"[ \t]+", " ", str(value["response"]).strip())
        return cls(
            decision=str(value["decision"]).strip(),
            is_over=value["is_over"],
            termination_reason=str(value["termination_reason"]).strip(),
            goal_status=str(value["goal_status"]).strip(),
            communication_status=str(value["communication_status"]).strip(),
            response=response,
            resolved_goals=tuple(str(item).strip() for item in value["resolved_goals"]),
            unresolved_goals=tuple(
                str(item).strip() for item in value["unresolved_goals"]
            ),
            evidence=tuple(Evidence.from_mapping(item) for item in value["evidence"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["resolved_goals"] = list(self.resolved_goals)
        value["unresolved_goals"] = list(self.unresolved_goals)
        value["evidence"] = [asdict(item) for item in self.evidence]
        return value


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def parse_teacher_decision(raw: str) -> TeacherDecision:
    """Parse and structurally validate one DeepSeek JSON response."""

    try:
        payload = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid teacher JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ContractError("teacher output must be one JSON object")
    decision = TeacherDecision.from_mapping(payload)
    validate_teacher_decision(decision)
    return decision


def _history_text(observable_history: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(item.get("content", "")) for item in observable_history)


def validate_teacher_decision(
    decision: TeacherDecision,
    *,
    scenario: str = "",
    observable_history: Sequence[Mapping[str, Any]] = (),
    privileged_entities: Iterable[str] = (),
    max_response_chars: int = 320,
) -> list[str]:
    """Validate hard invariants and return non-fatal quality issues.

    Hard contradictions raise :class:`ContractError`; semantic heuristics return
    issue strings so generation can quarantine rather than silently discard data.
    """

    if decision.decision not in DECISIONS:
        raise ContractError(f"invalid decision: {decision.decision}")
    if decision.termination_reason not in TERMINATION_REASONS:
        raise ContractError(
            f"invalid termination_reason: {decision.termination_reason}"
        )
    if decision.goal_status not in GOAL_STATUSES:
        raise ContractError(f"invalid goal_status: {decision.goal_status}")
    if decision.communication_status not in COMMUNICATION_STATUSES:
        raise ContractError(
            f"invalid communication_status: {decision.communication_status}"
        )

    contains_stop = STOP_TOKEN in decision.response
    if contains_stop and decision.decision != "stop":
        raise ContractError("STOP response requires decision=stop")
    if decision.decision == "stop" and decision.response != STOP_TOKEN:
        raise ContractError("terminal response must be exactly ###STOP###")
    if decision.decision == "continue" and decision.is_over:
        raise ContractError("continue decision requires is_over=false")
    if decision.decision == "stop" and not decision.is_over:
        raise ContractError("stop decision requires is_over=true")
    if decision.decision == "continue" and decision.termination_reason != "continue":
        raise ContractError("continue decision requires termination_reason=continue")
    if decision.decision == "stop" and decision.termination_reason == "continue":
        raise ContractError("stop decision requires a terminal reason")
    if decision.termination_reason == "goal_satisfied":
        if decision.goal_status != "satisfied":
            raise ContractError("goal_satisfied requires goal_status=satisfied")
        if decision.unresolved_goals:
            raise ContractError("goal_satisfied requires no unresolved_goals")
    if decision.termination_reason == "goal_failed" and decision.goal_status != "failed":
        raise ContractError("goal_failed requires goal_status=failed")
    if (
        decision.termination_reason == "cannot_continue"
        and decision.goal_status not in {"blocked", "failed"}
    ):
        raise ContractError("cannot_continue requires blocked or failed goal_status")

    issues: list[str] = []
    if not decision.response:
        issues.append("empty_response")
    if "\n" in decision.response:
        issues.append("multiline_response")
    if len(decision.response) > max_response_chars:
        issues.append(f"response_too_long:{len(decision.response)}")

    issues.extend(
        validate_observable_grounding_text(
            decision.response,
            scenario=scenario,
            observable_history=observable_history,
        )
    )

    visible_text = f"{scenario}\n{_history_text(observable_history)}".casefold()
    response_text = decision.response.casefold()
    for entity in sorted({str(item).strip() for item in privileged_entities if item}):
        if entity.casefold() in response_text and entity.casefold() not in visible_text:
            issues.append(f"privileged_entity_leak:{entity}")

    customer_messages = [
        str(item.get("content", "")).strip().casefold()
        for item in observable_history
        if str(item.get("role", "")).lower() == "user"
    ]
    latest_agent_message = next(
        (
            str(item.get("content", "")).casefold()
            for item in reversed(observable_history)
            if str(item.get("role", "")).lower() == "agent"
        ),
        "",
    )
    confirmation_requested = any(
        cue in latest_agent_message
        for cue in (
            "please confirm",
            "confirm if",
            "do you confirm",
            "shall i proceed",
            "would you like me to proceed",
            "please say \"yes\"",
            "please say 'yes'",
            "say \"yes\" to confirm",
            "say 'yes' to confirm",
        )
    )
    is_short_confirmation = bool(
        re.match(
            r"^(?:yes|yeah|yep|sure|please proceed)\b",
            decision.response.casefold(),
        )
    )
    if (
        decision.response.casefold() in customer_messages
        and not (confirmation_requested and is_short_confirmation)
    ):
        issues.append("exact_user_repetition")
    valid_turn_indices = {
        int(item["turn_index"])
        for item in observable_history
        if "turn_index" in item
    }
    for item in decision.evidence:
        if item.turn_index not in valid_turn_indices:
            issues.append(f"invalid_evidence_turn:{item.turn_index}")
    if decision.decision == "continue" and not decision.unresolved_goals:
        issues.append("continue_without_unresolved_goal")
    if (
        decision.termination_reason == "goal_satisfied"
        and decision.communication_status != "complete"
    ):
        issues.append("satisfied_without_complete_communication")
    return issues


def validate_response_constraints(
    decision: TeacherDecision,
    constraints: Optional[Mapping[str, Any]],
) -> list[str]:
    """Apply case-specific semantic gates to curated pilot responses.

    These constraints are deliberately absent from historical full-batch cases. They
    turn a small set of high-risk behaviors—such as eager slot disclosure—into
    executable release tests without teaching the model a single canonical wording.
    Each inner ``must_include_any_of_each`` group is an OR-set; all groups must match.
    """

    if not constraints:
        return []
    text = decision.response.casefold()
    issues: list[str] = []
    for group_index, raw_group in enumerate(
        constraints.get("must_include_any_of_each", [])
    ):
        group = [str(item).strip().casefold() for item in raw_group if str(item).strip()]
        if group and not any(item in text for item in group):
            issues.append(f"missing_required_response_group:{group_index}")
    for item_index, raw_item in enumerate(constraints.get("must_not_include", [])):
        item = str(raw_item).strip().casefold()
        if item and item in text:
            issues.append(f"forbidden_response_content:{item_index}")
    max_chars = constraints.get("max_chars")
    if max_chars is not None and len(decision.response) > int(max_chars):
        issues.append(f"case_response_too_long:{len(decision.response)}")
    return issues


def build_sft_record(
    source_case: Mapping[str, Any],
    decision: TeacherDecision,
    *,
    prompt_version: str,
    quality_issues: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Create a drop-in chat SFT record with audit labels outside the target."""

    messages = [dict(message) for message in source_case["student_messages"]]
    if not messages or messages[-1].get("role") != "user":
        raise ContractError("student_messages must end with role=user before target")
    messages.append({"role": "assistant", "content": decision.response})
    return {
        "case_id": source_case["case_id"],
        "task_id": int(source_case["task_id"]),
        "messages": messages,
        "metadata": {
            "decision": decision.decision,
            "is_over": decision.is_over,
            "termination_reason": decision.termination_reason,
            "goal_status": decision.goal_status,
            "communication_status": decision.communication_status,
            "resolved_goals": list(decision.resolved_goals),
            "unresolved_goals": list(decision.unresolved_goals),
            "evidence": [asdict(item) for item in decision.evidence],
            "quality_issues": list(quality_issues or ()),
            "prompt_version": prompt_version,
        },
    }
