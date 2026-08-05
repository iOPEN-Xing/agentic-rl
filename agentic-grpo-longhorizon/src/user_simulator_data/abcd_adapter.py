"""Conservative ABCD-to-runtime adapter for User-Simulator SFT pilots.

ABCD is an e-commerce, human-to-human dialogue corpus.  It is useful for
progressive disclosure and multi-step persistence, but its ``action`` events and
``end_conversation`` labels do not match the current tau-bench user runtime.  This
adapter therefore exports only intermediate CONTINUE targets and never derives a
``###STOP###`` label from ABCD.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .case_builder import build_runtime_system_prompt
from .contracts import (
    ContractError,
    STOP_TOKEN,
    build_sft_record,
    parse_teacher_decision,
    validate_teacher_decision,
)


ABCD_PROMPT_VERSION = "abcd-usim-adapter-v0.3-pilot"
ABCD_SOURCE = "asappresearch/abcd-v1.1"
FIXED_AGENT_GREETING = "Hi! How can I help you today?"

# These goals were reviewed at the *conversation* level.  The fallback in 3592
# and the ETA request in 9489 are not universal properties of their subflows.
# Binding them to convo_id prevents an apparently known subflow from silently
# applying a pilot-specific goal to unreviewed full-dataset conversations.
_PILOT_SCENARIO_SPECS = {
    3592: {
        "subflow": "return_size",
        "goal": (
            "Return the purchased item because it is the wrong size. If the normal "
            "return is refused because it is outside the return window, politely ask "
            "whether the issue can be escalated to a manager."
        ),
        "goal_labels": (
            "return the wrong-size item",
            "request manager escalation if the normal return is denied",
        ),
    },
    9489: {
        "subflow": "refund_status",
        "goal": (
            "Check the status of a refund and find out approximately how long it "
            "will take to complete."
        ),
        "goal_labels": (
            "learn the refund status",
            "learn the approximate refund completion time",
        ),
    },
    3695: {
        "subflow": "timing_4",
        "goal": "Ask how long promotional codes remain valid before they expire.",
        "context": "You plan to use the promo code to buy hats for your cat.",
        "goal_labels": ("learn the promo-code validity period",),
    },
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
    r"(?:\b(?:that'?s|that is) all\b|\bhave a (?:great|nice) day\b|\btake care\b|"
    r"\bthanks? for (?:(?:your|the) )?help\b|\bnothing else[.! ]*$|"
    r"\bthat(?:'ll| will) be (?:all|it|everything)\b|"
    r"\bi (?:can )?see\b.*\bnow\b.*\bthank|"
    r"\bgreat[, ]+thanks?[.! ]*$|"
    r"^\s*(?:great[, ]*)?no[, ]+thanks?[.! ]*$|"
    r"^\s*(?:great|perfect|thanks?|thank you)[.! ]*$)",
    flags=re.IGNORECASE,
)
_OPENING_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey(?: ho)?|good (?:morning|afternoon|evening))"
    r"(?: there)?[!. ]*$",
    flags=re.IGNORECASE,
)
_AGENT_OPENING_SIGNAL = re.compile(
    r"\b(?:hi|hello|hey|welcome|good (?:morning|afternoon|evening)|"
    r"thanks? for (?:contacting|calling|reaching)|"
    r"how (?:can|may) i (?:help|assist))\b",
    flags=re.IGNORECASE,
)
_AGENT_SERVICE_CUE = re.compile(
    r"\b(?:help|assist|support|welcome)\b", flags=re.IGNORECASE
)
_ABCD_GOAL_EXPRESSION_CUE = re.compile(
    r"\b(?:want|need|would like|looking|trying|help|issue|problem|question|"
    r"check|change|cancel|return|refund|buy|purchase|order|account|password|"
    r"username|shipping|delivery|price|cost|promo|membership|subscription|"
    r"bill|card|stock|website|product|wrong|missing|charged|status|when|where|"
    r"why|how|what|can you|could you)\b",
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

communication_status audits how much of the customer's OVERALL BUSINESS GOAL the
Agent has communicated before this new customer turn; it does not grade whether the
source customer reply is a complete answer to the latest question. Use none when the
Agent has only greeted, requested a slot, or provided no requested result. Use partial
when the Agent has resolved at least one but not all overall goals. ``complete`` is
forbidden in this CONTINUE-only pilot. resolved_goals must list actual business goals
already answered by the Agent, while unresolved_goals must retain every remaining
business goal. Providing identity fields is progress, not a resolved business goal.
Copy every string from BUSINESS_GOAL_LABELS exactly once into either resolved_goals
or unresolved_goals. Do not paraphrase, omit, duplicate, or invent goal labels.

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


def canonical_abcd_intent(conversation: Mapping[str, Any]) -> str:
    """Return the 55-way intent label, not the 96-way raw scenario leaf."""

    labels: list[str] = []
    for turn_index, turn in enumerate(conversation.get("delexed", [])):
        targets = turn.get("targets") if isinstance(turn, Mapping) else None
        if not isinstance(targets, list) or not targets or not _clean(targets[0]):
            raise ValueError(f"missing ABCD canonical intent at turn {turn_index}")
        labels.append(_clean(targets[0]))
    if not labels:
        raise ValueError("ABCD conversation has no delexed intent labels")
    unique = set(labels)
    if len(unique) != 1:
        raise ValueError(f"ABCD canonical intent drift: {sorted(unique)}")
    return labels[0]


def build_abcd_scenario(source: Mapping[str, Any], *, convo_id: int) -> str:
    """Build a customer-known natural instruction without action/policy state."""

    spec = _PILOT_SCENARIO_SPECS.get(int(convo_id))
    if spec is None:
        raise ValueError(f"unreviewed ABCD pilot conversation: {convo_id}")
    subflow = _clean(source.get("subflow"))
    if subflow != spec["subflow"]:
        raise ValueError(
            f"ABCD pilot subflow drift for {convo_id}: "
            f"{subflow!r} != {spec['subflow']!r}"
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
    lines = [opening, f"Goal: {spec['goal']}"]
    if spec.get("context"):
        lines.append(f"Context (not a separate goal): {spec['context']}")
    lines.extend(
        (
            f"Known facts: {fact_text}",
            (
                "Reveal only facts needed for the current step, answer the agent's "
                "question directly, and do not invent identifiers or outcomes."
            ),
        )
    )
    return "\n".join(lines)


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


def _runtime_blocks_after_fixed_greeting(
    blocks: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
    str,
]:
    """Map either source opening convention onto the runtime's fixed greeting."""

    if not blocks:
        return [], None, None, "empty"
    if blocks[0]["speaker"] == "agent":
        return list(blocks[1:]), blocks[0], None, "agent_first"
    if blocks[0]["speaker"] == "customer":
        opening_text = " ".join(blocks[0]["texts"])
        if _OPENING_GREETING.fullmatch(opening_text):
            if len(blocks) > 1 and blocks[1]["speaker"] == "agent":
                agent_opening = " ".join(blocks[1]["texts"])
                replaceable = bool(_OPENING_GREETING.fullmatch(agent_opening)) or (
                    bool(_AGENT_OPENING_SIGNAL.search(agent_opening))
                    and bool(_AGENT_SERVICE_CUE.search(agent_opening))
                )
                if replaceable:
                    return (
                        list(blocks[2:]),
                        blocks[1],
                        blocks[0],
                        "customer_greeting_then_agent",
                    )
                return (
                    [],
                    blocks[1],
                    blocks[0],
                    "customer_greeting_unmappable",
                )
            return [], None, blocks[0], "customer_greeting_unmappable"
        return list(blocks), None, None, "customer_goal_first"
    raise ValueError(f"invalid ABCD opening speaker: {blocks[0]['speaker']!r}")


def count_abcd_continue_candidates(conversation: Mapping[str, Any]) -> dict[str, Any]:
    """Count conservative full-data candidates without constructing a scenario."""

    original = list(conversation.get("original", []))
    blocks = _dialogue_blocks(original)
    natural_indices = [
        index
        for index, turn in enumerate(original)
        if isinstance(turn, Sequence)
        and not isinstance(turn, (str, bytes))
        and len(turn) == 2
        and _clean(turn[0]).casefold() in {"agent", "customer"}
    ]
    consumed_indices = [index for block in blocks for index in block["turn_indices"]]
    runtime_blocks, _, _, source_opening_mode = _runtime_blocks_after_fixed_greeting(
        blocks
    )
    customer_blocks = [
        block for block in runtime_blocks if block["speaker"] == "customer"
    ]
    courtesy_blocks = [
        block
        for block in customer_blocks
        if _TERMINAL_COURTESY.search(" ".join(block["texts"]))
    ]
    candidates = [block for block in customer_blocks if block not in courtesy_blocks]
    return {
        "candidate_continue_blocks": len(candidates),
        "candidate_response_chars": sum(
            len(" ".join(block["texts"])) for block in candidates
        ),
        "customer_blocks_after_runtime_greeting": len(customer_blocks),
        "customer_fragments_merged": sum(
            max(len(block["turn_indices"]) - 1, 0) for block in customer_blocks
        ),
        "terminal_courtesy_blocks": len(courtesy_blocks),
        "action_turns": sum(
            isinstance(turn, Sequence)
            and not isinstance(turn, (str, bytes))
            and len(turn) == 2
            and _clean(turn[0]).casefold() == "action"
            for turn in original
        ),
        "causal_cut_applied": bool(
            natural_indices
            and consumed_indices
            and max(consumed_indices) < max(natural_indices)
        ),
        "opening_mapping_unmappable": (
            source_opening_mode == "customer_greeting_unmappable"
        ),
    }


def _is_low_information_abcd_response(candidate: Mapping[str, Any]) -> bool:
    response = _clean(candidate.get("reference_response"))
    words = response.split()
    return len(words) <= 2 and not any(
        character.isdigit() for character in response
    )


def _abcd_agent_echo_ratio(response: str, preceding_agent: str) -> float:
    response_tokens = set(re.findall(r"[a-z0-9]+", response.casefold()))
    if not response_tokens:
        return 0.0
    agent_tokens = set(re.findall(r"[a-z0-9]+", preceding_agent.casefold()))
    return len(response_tokens & agent_tokens) / len(response_tokens)


def _abcd_review_target_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer a grounded progress turn after preserving the opening intent turn."""

    response = _clean(candidate.get("reference_response"))
    preceding_agent = _clean(candidate.get("preceding_agent_text"))
    return (
        bool(candidate.get("hidden_action_turn_indices_before_target")),
        float(candidate.get("agent_echo_ratio", 0.0)) >= 0.8,
        _is_low_information_abcd_response(candidate),
        "?" not in preceding_agent,
        "?" not in response,
        -int(candidate.get("source_fragment_count", 1)),
        -min(len(response), 240),
        int(candidate["source_turn_indices"][0]),
    )


def _looks_like_abcd_goal_expression(candidate: Mapping[str, Any]) -> bool:
    response = _clean(candidate.get("reference_response"))
    return len(response.split()) >= 3 and bool(
        _ABCD_GOAL_EXPRESSION_CUE.search(response)
    )


def build_abcd_review_item(
    conversation: Mapping[str, Any],
    *,
    targets_per_conversation: int = 2,
) -> dict[str, Any]:
    """Build a non-generatable, action-redacted conversation review item.

    This is deliberately separate from :func:`build_abcd_cases`: a full-data
    conversation has no approved natural-language goal yet.  The artifact gives a
    reviewer source evidence and candidate customer targets, but no runtime system
    prompt and no path into Teacher generation until the review fields are completed
    by a later, separately validated promotion step.
    """

    if targets_per_conversation <= 0:
        raise ValueError("targets_per_conversation must be positive")
    convo_id = int(conversation["convo_id"])
    scenario = conversation.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"ABCD scenario must be an object: {convo_id}")
    flow = _clean(scenario.get("flow"))
    raw_subflow = _clean(scenario.get("subflow"))
    if not flow or not raw_subflow:
        raise ValueError(f"ABCD flow/subflow missing: {convo_id}")

    original = list(conversation.get("original", []))
    blocks = _dialogue_blocks(original)
    runtime_blocks, source_opening, source_customer_greeting, source_opening_mode = (
        _runtime_blocks_after_fixed_greeting(blocks)
    )
    action_turn_indices = [
        index
        for index, turn in enumerate(original)
        if isinstance(turn, Sequence)
        and not isinstance(turn, (str, bytes))
        and len(turn) == 2
        and _clean(turn[0]).casefold() == "action"
    ]
    last_visible_agent_turn = (
        max(source_opening["turn_indices"]) if source_opening is not None else -1
    )
    student_prefix: list[dict[str, str]] = [
        {"role": "user", "content": FIXED_AGENT_GREETING}
    ]
    candidates: list[dict[str, Any]] = []
    customer_turns_seen = 0

    for block in runtime_blocks:
        content = " ".join(block["texts"])
        if block["speaker"] == "agent":
            if student_prefix[-1]["role"] == "user":
                student_prefix[-1]["content"] += " " + content
            else:
                student_prefix.append({"role": "user", "content": content})
            last_visible_agent_turn = max(block["turn_indices"])
            continue

        if student_prefix[-1]["role"] != "user":
            break
        if not _TERMINAL_COURTESY.search(content):
            preceding_agent = student_prefix[-1]["content"]
            candidates.append(
                {
                    "source_turn_indices": list(block["turn_indices"]),
                    "reference_response": content,
                    "preceding_agent_text": preceding_agent,
                    "student_prefix_without_system": [
                        dict(message) for message in student_prefix
                    ],
                    "source_fragment_count": len(block["turn_indices"]),
                    "response_chars": len(content),
                    "is_opening_target": customer_turns_seen == 0,
                    "agent_echo_ratio": round(
                        _abcd_agent_echo_ratio(content, preceding_agent), 4
                    ),
                    "hidden_action_turn_indices_before_target": [
                        index
                        for index in action_turn_indices
                        if last_visible_agent_turn < index < block["turn_indices"][0]
                    ],
                }
            )
        student_prefix.append({"role": "assistant", "content": content})
        customer_turns_seen += 1

    proposed: list[dict[str, Any]] = []
    eligible_candidates = [
        candidate
        for candidate in candidates
        if not candidate["hidden_action_turn_indices_before_target"]
    ]
    early_goal = next(
        (
            candidate
            for candidate in eligible_candidates
            if _looks_like_abcd_goal_expression(candidate)
        ),
        None,
    )
    if early_goal is None:
        early_goal = next(
            (
                candidate
                for candidate in eligible_candidates
                if not _is_low_information_abcd_response(candidate)
            ),
            eligible_candidates[0] if eligible_candidates else None,
        )
    if early_goal is not None:
        proposed.append(
            {
                **early_goal,
                "selection_reason": (
                    "early_goal_expression"
                    if _looks_like_abcd_goal_expression(early_goal)
                    else "fallback_early_customer_turn"
                ),
            }
        )
    early_goal_end = (
        max(early_goal["source_turn_indices"]) if early_goal is not None else -1
    )
    remaining = [
        candidate
        for candidate in eligible_candidates
        if candidate is not early_goal
        and min(candidate["source_turn_indices"]) > early_goal_end
    ]
    for candidate in sorted(remaining, key=_abcd_review_target_rank):
        if len(proposed) >= targets_per_conversation:
            break
        proposed.append({**candidate, "selection_reason": "later_goal_progress"})

    counts = count_abcd_continue_candidates(conversation)
    target_reviews = [
        {
            "source_turn_indices": list(target["source_turn_indices"]),
            "eligibility": "pending",
            "issue_tags": [],
            "notes": "",
        }
        for target in proposed
    ]
    return {
        "format_version": "abcd-conversation-review-v1",
        "source": ABCD_SOURCE,
        "source_split": "train",
        "source_convo_id": convo_id,
        "canonical_intent": canonical_abcd_intent(conversation),
        "flow": flow,
        "raw_subflow": raw_subflow,
        "raw_leaf": f"{flow}/{raw_subflow}",
        "source_opening_mode": source_opening_mode,
        "source_scenario": json.loads(json.dumps(scenario, ensure_ascii=False)),
        "source_opening_agent": (
            " ".join(source_opening["texts"]) if source_opening is not None else ""
        ),
        "source_opening_turn_indices": (
            list(source_opening["turn_indices"])
            if source_opening is not None
            else []
        ),
        "source_opening_customer_greeting_turn_indices": (
            list(source_customer_greeting["turn_indices"])
            if source_customer_greeting is not None
            else []
        ),
        "redacted_action_turn_indices": action_turn_indices,
        "candidate_counts": counts,
        "proposed_targets": proposed,
        "selection_eligible": len(proposed) == targets_per_conversation,
        "review": {
            "status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "goal": "",
            "context": "",
            "goal_labels": [],
            "target_reviews": target_reviews,
            "notes": "",
        },
        "allowed_for_generation": False,
    }


def _abcd_review_item_rank(item: Mapping[str, Any], *, seed: str) -> tuple[Any, ...]:
    stable = hashlib.sha256(
        (
            f"{seed}|{item['canonical_intent']}|{item['raw_leaf']}|"
            f"{item['source_convo_id']}"
        ).encode("utf-8")
    ).hexdigest()
    counts = item.get("candidate_counts", {})
    return (
        bool(counts.get("causal_cut_applied")),
        -int(counts.get("candidate_continue_blocks", 0)),
        stable,
    )


def select_abcd_review_queue(
    conversations: Iterable[Mapping[str, Any]],
    *,
    conversations_per_intent: int = 2,
    targets_per_conversation: int = 2,
    seed: str = "abcd-55-intent-review-v1",
) -> list[dict[str, Any]]:
    """Select a deterministic, raw-leaf-diverse queue for human review."""

    if conversations_per_intent <= 0:
        raise ValueError("conversations_per_intent must be positive")
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for conversation in conversations:
        item = build_abcd_review_item(
            conversation,
            targets_per_conversation=targets_per_conversation,
        )
        if not item["selection_eligible"]:
            continue
        by_intent.setdefault(item["canonical_intent"], []).append(item)

    selected: list[dict[str, Any]] = []
    insufficient: dict[str, int] = {}
    for intent in sorted(by_intent):
        ranked = sorted(
            by_intent[intent],
            key=lambda item: _abcd_review_item_rank(item, seed=seed),
        )
        if len(ranked) < conversations_per_intent:
            insufficient[intent] = len(ranked)
            continue
        intent_selected: list[dict[str, Any]] = []
        used_raw_leaves: set[str] = set()
        for item in ranked:
            if item["raw_leaf"] in used_raw_leaves:
                continue
            intent_selected.append(item)
            used_raw_leaves.add(item["raw_leaf"])
            if len(intent_selected) == conversations_per_intent:
                break
        if len(intent_selected) < conversations_per_intent:
            chosen_ids = {int(item["source_convo_id"]) for item in intent_selected}
            for item in ranked:
                if int(item["source_convo_id"]) in chosen_ids:
                    continue
                intent_selected.append(item)
                chosen_ids.add(int(item["source_convo_id"]))
                if len(intent_selected) == conversations_per_intent:
                    break
        for selection_rank, item in enumerate(intent_selected, start=1):
            value = dict(item)
            value["queue_selection"] = {
                "seed": seed,
                "intent_rank": selection_rank,
                "conversations_per_intent": conversations_per_intent,
                "targets_per_conversation": targets_per_conversation,
                "raw_leaf_diversity_preferred": True,
            }
            selected.append(value)

    if insufficient:
        raise ValueError(f"insufficient ABCD review coverage: {insufficient}")
    return selected


def build_abcd_cases(conversation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one ABCD conversation into runtime-reachable CONTINUE cases."""

    convo_id = int(conversation["convo_id"])
    source_scenario = conversation["scenario"]
    scenario = build_abcd_scenario(source_scenario, convo_id=convo_id)
    scenario_spec = _PILOT_SCENARIO_SPECS[convo_id]
    blocks = _dialogue_blocks(conversation.get("original", []))
    runtime_blocks, _, _, _ = _runtime_blocks_after_fixed_greeting(blocks)
    if not runtime_blocks:
        return []

    student_messages: list[dict[str, str]] = [
        {"role": "system", "content": build_runtime_system_prompt(scenario)},
        {"role": "user", "content": FIXED_AGENT_GREETING},
    ]
    observable_history: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    # A source agent opening is replaced by the runtime greeting.  A source customer
    # opening becomes the first target conditioned on that same greeting.
    for block in runtime_blocks:
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
                    "business_goal_labels": list(scenario_spec["goal_labels"]),
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
        "business_goal_labels": case["business_goal_labels"],
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
        if decision.communication_status == "complete":
            issues.append("continue_with_complete_communication")
        expected_goals = {str(goal) for goal in case.get("business_goal_labels", [])}
        resolved_goals = set(decision.resolved_goals)
        unresolved_goals = set(decision.unresolved_goals)
        if (
            not expected_goals
            or resolved_goals & unresolved_goals
            or resolved_goals | unresolved_goals != expected_goals
            or len(decision.resolved_goals) + len(decision.unresolved_goals)
            != len(expected_goals)
        ):
            issues.append("business_goal_partition_mismatch")
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


def validate_abcd_resume_alignment(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    expected_model: str,
) -> None:
    """Refuse stale generations when a source case changed under the same ID."""

    case_id = str(case["case_id"])
    if str(row.get("case_id")) != case_id:
        raise ValueError(f"resume case ID drift: {case_id}")
    if row.get("prompt_version") != ABCD_PROMPT_VERSION:
        raise ValueError(f"resume prompt version drift: {case_id}")
    generated_model = row.get("api", {}).get("model")
    if generated_model and generated_model != expected_model:
        raise ValueError(f"resume model drift: {case_id}")
    if row.get("reference_response") != case.get("reference_response"):
        raise ValueError(f"resume source target drift: {case_id}")
    record = row.get("sft_record") or {}
    messages = record.get("messages") or []
    if messages[:-1] != case.get("student_messages"):
        raise ValueError(f"resume source prefix drift: {case_id}")
    if list(row.get("source_turn_indices", [])) != list(
        case.get("source_turn_indices", [])
    ):
        raise ValueError(f"resume source turn drift: {case_id}")
    decision = row.get("teacher_decision") or {}
    generated_goals = {
        str(goal)
        for field in ("resolved_goals", "unresolved_goals")
        for goal in decision.get(field, [])
    }
    if generated_goals != {str(goal) for goal in case.get("business_goal_labels", [])}:
        raise ValueError(f"resume business goal drift: {case_id}")


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
