"""Versioned DeepSeek teacher prompt for τ-bench user-simulator data."""

from __future__ import annotations

import json
from typing import Any, Mapping


PROMPT_VERSION = "tau-airline-usim-teacher-v1.0"

RUNTIME_USER_RULES = """You are the customer interacting with an airline agent.
- Reply with one concise customer message at a time.
- Follow the scenario exactly; never invent facts absent from the scenario.
- Reveal information progressively and only when useful for the current step.
- Preserve the scenario's personality without copying the instruction verbatim.
- Continue until every explicit and conditional user-facing goal is resolved.
- When all goals have been resolved and communicated by the agent, output exactly ###STOP###.
"""

TEACHER_SYSTEM_PROMPT = f"""You construct and audit user-simulator SFT examples for a
tool-using airline customer-service benchmark. Return exactly one JSON object and no
markdown. Prompt version: {PROMPT_VERSION}.

The student user simulator sees only OBSERVABLE_CONTEXT. It never sees
PRIVILEGED_AUDIT_REFERENCE. Privileged data can help you audit labels, but you must not reveal
privileged facts, identifiers, tool outputs, gold actions, or hidden verifier
state in the customer response unless they already appear in the scenario or observable
conversation.

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
   information merely because it appears in the gold reference.
6. For a completed or irrecoverably ended case, response must be exactly ###STOP###.

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
    """Render one teacher request without mixing privileged and student contexts."""

    observable = {
        "case_id": case["case_id"],
        "task_id": int(case["task_id"]),
        "scenario": case["scenario"],
        "runtime_user_rules": RUNTIME_USER_RULES,
        "conversation": case.get("observable_history", []),
    }
    privileged = case.get("privileged_reference", {})
    user_prompt = (
        "OBSERVABLE_CONTEXT\n"
        + json.dumps(observable, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nPRIVILEGED_AUDIT_REFERENCE\n"
        + json.dumps(privileged, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nGenerate the next customer turn as the required JSON object."
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
