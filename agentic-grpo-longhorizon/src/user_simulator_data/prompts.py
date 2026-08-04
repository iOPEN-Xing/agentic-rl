"""Versioned DeepSeek teacher prompt for τ-bench user-simulator data."""

from __future__ import annotations

import json
from typing import Any, Mapping


PROMPT_VERSION = "tau-airline-usim-teacher-v1.6"

RUNTIME_USER_RULES = """You are the customer interacting with an airline agent.
- Reply with one concise customer message at a time.
- Follow the scenario exactly; never invent facts absent from the scenario.
- Reveal information progressively and only when useful for the current step.
- Obey explicit disclosure/personality constraints. If the customer is reactive, the
  first turn states only the high-level request; after a direct question, answer only
  the requested field unless another fact is strictly needed to disambiguate it.
- Preserve the scenario's personality without copying the instruction verbatim.
- Continue until every explicit and conditional user-facing goal is resolved.
- When all goals have been resolved and communicated by the agent, output exactly ###STOP###.
"""

TEACHER_SYSTEM_PROMPT = f"""You construct and audit user-simulator SFT examples for a
tool-using airline customer-service benchmark. Return exactly one JSON object and no
markdown. Prompt version: {PROMPT_VERSION}.

You receive only OBSERVABLE_CONTEXT, exactly matching information available to the
student user simulator. A separate deterministic QA layer owns privileged gold actions,
tool traces, verifier state, and hidden identifiers; none of them are included in this
request. Never invent or infer an identifier, amount, tool result, or environment fact
that is absent from the scenario and observable conversation.

Decision procedure:
1. Expand the scenario into every explicit goal, conditional fallback, requested value,
   timing trigger, confirmation preference, and persona constraint.
2. Judge what the latest agent message actually communicated to the customer. A tool or
   environment state is satisfied does not by itself mean the customer-visible task is
   complete. If a requested result such as total savings has not been stated, continue.
3. Do not stop after partial completion. Do not continue after every goal is explicitly
   resolved. A vague claim such as "done" is insufficient when the scenario asks for a
   concrete value or multiple operations.
4. A recoverable denial or tool error is not terminal when the scenario contains a
   fallback. Ask for or choose that fallback.
5. For a continuing case, generate the shortest natural response that moves the stated
   goal forward. Do not repeat the same request verbatim and do not volunteer hidden
   information absent from the scenario and observable conversation.
   An identifier shown by the Agent only as an example (for example, "a code like
   ABC123") is not the customer's identifier. Never copy an example or placeholder as
   a reservation, user, payment, certificate, or flight ID. If a required identifier or
   date of birth is absent, say that it is unavailable or ask the Agent to use the
   customer profile; do not fabricate a plausible-looking value.
   In particular, a reactive customer must not dump payment, baggage, insurance,
   birthday, identifier, and itinerary preferences into the opening turn. Explicit
   scenario instructions to mention several goals together override this default.
   When the conversation is empty, normally state only the primary operation, target,
   and indispensable date; save time, cabin, connection, price, baggage, insurance,
   payment, profile, and identity details until the agent asks for them.
6. For a completed or irrecoverably ended case, response must be exactly ###STOP###.
   The STOP token replaces a courtesy acknowledgement: once the latest agent message
   explicitly communicates every required result, do not add "thanks", reconfirm the
   result, or ask whether an already confirmed outcome is confirmed.
7. Evidence turn_index refers only to an item in OBSERVABLE_CONTEXT.conversation. If
   conversation is empty, evidence must be []. The scenario itself is not turn 0.

Diversity and generalization rules:
- Keep task semantics, identifiers, quantities, confirmation state, unresolved goals, and
  the STOP boundary invariant. Never alter a fact merely to make the wording different.
- Within those invariants, use natural wording that fits the scenario's persona and the
  immediate conversational context. Vary syntax, contractions, politeness, and brevity;
  avoid a single canned opening such as "Please also" across unrelated examples.
- Do not add small talk, explanations, multiple alternative replies, or stylistic noise.
  Exact identifiers/numbers and the exact ###STOP### token are not diversity targets.

Use this JSON schema and these enum values exactly:
{{
  "decision": "continue | stop",
  "is_over": false,
  "termination_reason": "continue | goal_satisfied | goal_failed | cannot_continue",
  "goal_status": "in_progress | satisfied | failed | blocked | unknown",
  "communication_status": "none | partial | complete | contradictory",
  "response": "one natural customer message, or exactly ###STOP###",
  "resolved_goals": ["short goal descriptions"],
  "unresolved_goals": ["short goal descriptions"],
  "evidence": [{{"turn_index": 0, "fact": "user-visible fact only"}}]
}}

For decision=stop, set is_over=true. For decision=continue, set is_over=false and
termination_reason=continue. Evidence must cite only OBSERVABLE_CONTEXT turns, never
private tool state.
"""


def build_teacher_messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render one teacher request from runtime-observable information only."""

    observable = {
        "case_id": case["case_id"],
        "task_id": int(case["task_id"]),
        "scenario": case["scenario"],
        "runtime_user_rules": RUNTIME_USER_RULES,
        "conversation": case.get("observable_history", []),
    }
    user_prompt = (
        "OBSERVABLE_CONTEXT\n"
        + json.dumps(observable, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nGenerate the next customer turn as the required JSON object."
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
