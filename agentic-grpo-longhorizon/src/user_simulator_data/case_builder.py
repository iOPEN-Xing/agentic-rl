"""Convert public τ-bench trajectories into regeneration cases.

Historical customer turns are references for audit and coverage, not trusted labels.
Tool messages stay in the privileged teacher reference and are deliberately absent from
the student input, matching the current ``LLMUserSimulationEnv`` observation contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import STOP_TOKEN


def build_runtime_system_prompt(scenario: str) -> str:
    return f"""You are a user interacting with an agent.

Instruction: {scenario}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide information necessary for the current step.
- Do not hallucinate information not provided in the instruction.
- If every instruction goal is satisfied and communicated, generate '###STOP###' as a standalone message.
- Paraphrase rather than repeating the exact instruction.
- Keep the conversation natural and follow the instruction's personality.
""".strip()


def prepare_curated_case(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add the role-flipped student prompt to a human-curated pilot case."""

    case = dict(value)
    scenario = str(case["scenario"]).strip()
    student_messages: list[dict[str, str]] = [
        {"role": "system", "content": build_runtime_system_prompt(scenario)},
        {"role": "user", "content": "Hi! How can I help you today?"},
    ]
    normalized_history: list[dict[str, Any]] = []
    for index, message in enumerate(case.get("observable_history", [])):
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "") or "").strip()
        if role not in {"user", "agent"} or not content:
            raise ValueError(
                f"curated case {case.get('case_id')} has invalid observable turn {index}"
            )
        normalized_history.append(
            {
                "turn_index": int(message.get("turn_index", index)),
                "role": role,
                "content": content,
            }
        )
        student_messages.append(
            {"role": "assistant" if role == "user" else "user", "content": content}
        )
    case["scenario"] = scenario
    case["observable_history"] = normalized_history
    case["student_messages"] = student_messages
    case.setdefault("source", "curated-pilot")
    case.setdefault("quality_gate", True)
    case.setdefault("privileged_reference", {})
    return case


def _tool_trace(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call.get("function", {})
                trace.append(
                    {
                        "turn_index": index,
                        "kind": "tool_call",
                        "name": function.get("name"),
                        "arguments": function.get("arguments", {}),
                    }
                )
        elif role == "tool":
            trace.append(
                {
                    "turn_index": index,
                    "kind": "tool_result",
                    "name": message.get("name"),
                    "content": message.get("content", ""),
                }
            )
    return trace


def _privileged_entities(task: Mapping[str, Any]) -> list[str]:
    text = json.dumps(
        {"actions": task.get("actions", []), "outputs": task.get("outputs", [])},
        ensure_ascii=False,
    )
    candidates = set(re.findall(r"\b[A-Z0-9][A-Z0-9_-]{4,}\b", text))
    return sorted(candidates)


def build_cases_from_historical_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    only_rewards: set[float] | None = None,
) -> list[dict[str, Any]]:
    """Build one next-customer-turn case for every historical customer message."""

    cases: list[dict[str, Any]] = []
    for row in rows:
        reward = float(row.get("reward", 0.0))
        if only_rewards is not None and reward not in only_rewards:
            continue
        task = row.get("info", {}).get("task", {})
        scenario = str(task.get("instruction", "")).strip()
        task_id = int(row["task_id"])
        trial = int(row.get("trial", 0))
        trajectory = list(row.get("traj", []))

        observable_history: list[dict[str, Any]] = []
        student_messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_runtime_system_prompt(scenario)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        for message_index, message in enumerate(trajectory):
            role = str(message.get("role", ""))
            content = str(message.get("content", "") or "").strip()
            if role == "system" or role == "tool":
                continue
            if role == "assistant":
                # The current simulator is called only for text responses, never after
                # an agent tool-call step. Preserve exactly that runtime observation.
                if message.get("tool_calls") or not content:
                    continue
                observable_history.append(
                    {"turn_index": message_index, "role": "agent", "content": content}
                )
                student_messages.append({"role": "user", "content": content})
                continue
            if role != "user" or not content:
                continue

            expected_decision = "stop" if STOP_TOKEN in content else "continue"
            case_id = (
                f"historical-airline-t{task_id:04d}-trial{trial:02d}-"
                f"u{message_index:03d}"
            )
            cases.append(
                {
                    "case_id": case_id,
                    "task_id": task_id,
                    "scenario": scenario,
                    "observable_history": [dict(item) for item in observable_history],
                    "student_messages": [dict(item) for item in student_messages],
                    "reference_response": content,
                    "expected_decision": expected_decision,
                    "source": "tau-bench/sonnet-3.5-historical-airline",
                    "privileged_reference": {
                        "trajectory_reward": reward,
                        "required_actions": task.get("actions", []),
                        "required_outputs": task.get("outputs", []),
                        "tool_trace": _tool_trace(trajectory[:message_index]),
                        "privileged_entities": _privileged_entities(task),
                        "warning": (
                            "Historical STOP and trajectory reward are independent labels; "
                            "regenerate the response instead of copying it."
                        ),
                    },
                }
            )
            observable_history.append(
                {"turn_index": message_index, "role": "user", "content": content}
            )
            student_messages.append({"role": "assistant", "content": content})
    return cases


def load_historical_rows(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list):
        raise ValueError("historical trajectory file must contain a JSON array")
    return value
