"""Conservative ABCD-to-runtime adapter for User-Simulator SFT pilots.

ABCD is an e-commerce, human-to-human dialogue corpus.  It is useful for
progressive disclosure and multi-step persistence, but its ``action`` events and
``end_conversation`` labels do not match the current tau-bench user runtime.  This
adapter therefore exports only intermediate CONTINUE targets and never derives a
``###STOP###`` label from ABCD.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .case_builder import build_runtime_system_prompt


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

_TERMINAL_COURTESY = re.compile(
    r"(?:\bthat(?:'s| is) all\b|\bhave a (?:great|nice) day\b|\btake care\b|"
    r"\bthanks? for (?:your )?help\b|^\s*(?:great|perfect|thanks?|thank you)[.! ]*$)",
    flags=re.IGNORECASE,
)


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


def build_abcd_teacher_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    raise NotImplementedError


def generate_abcd_one(client: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
