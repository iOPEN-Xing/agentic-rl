from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.envs.reward_fusion import JudgeSignalPolicy, reward_consistency
from src.envs.user_simulator_judge import JudgeFeedback, UserSimulatorTurnJudge


class _MockCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _MockClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_MockCompletions(content))


def test_feedback_parses_fenced_json_and_clamps_scores():
    feedback = JudgeFeedback.from_response(
        'result: ```json\n{"task_progress": 1.2, "tool_correctness": 0.5, '
        '"communication": -0.2, "rationale": "ok"}\n```'
    )
    assert feedback.valid
    assert feedback.task_progress == 1.0
    assert feedback.communication == 0.0
    assert feedback.score == pytest.approx(0.65)


def test_judge_uses_structured_prompt_and_returns_feedback():
    client = _MockClient(
        '{"task_progress": 0.8, "tool_correctness": 0.9, '
        '"communication": 0.7, "improvement_hint": "verify the date"}'
    )
    judge = UserSimulatorTurnJudge(
        model="judge-model",
        base_url="http://localhost:8001/v1",
        client=client,
    )
    feedback = asyncio.run(judge.evaluate(
        agent_message="I updated the booking.",
        task_goal="Change the return flight.",
        conversation_history=[{"role": "user", "content": "Please change it."}],
        action_history=[{
            "tool": "update_reservation_flights",
            "is_error": True,
            "observation": "Error: cabin cannot be changed.",
        }],
    ))

    assert feedback.valid
    assert feedback.score == pytest.approx(0.825)
    request = client.chat.completions.kwargs
    assert request["temperature"] == 0.0
    assert "task_goal" in request["messages"][1]["content"]
    assert "Error: cabin cannot be changed." in request["messages"][1]["content"]


def test_judge_failure_is_explicitly_unavailable():
    judge = UserSimulatorTurnJudge(
        model="judge-model",
        base_url="http://localhost:8001/v1",
        client=_MockClient('{"task_progress": "high"}'),
    )
    feedback = asyncio.run(judge.evaluate(
        agent_message="response",
        task_goal="goal",
        conversation_history=[],
        action_history=[],
    ))
    assert not feedback.valid
    assert feedback.score == 0.0
    assert "ValueError" in feedback.error


def test_judge_signal_is_centered_and_bounded():
    policy = JudgeSignalPolicy(neutral_score=0.5)
    positive = JudgeFeedback(
        task_progress=0.8,
        tool_correctness=0.9,
        communication=0.7,
        valid=True,
    )
    negative = JudgeFeedback(valid=True)

    assert policy.compute(positive) == pytest.approx(0.65)
    assert policy.compute(negative) == pytest.approx(-1.0)


def test_unavailable_judge_has_no_trainable_signal():
    policy = JudgeSignalPolicy()
    assert policy.compute(JudgeFeedback.unavailable("timeout")) is None
    with pytest.raises(ValueError):
        policy.compute(JudgeFeedback(task_progress=float("nan"), valid=True))


class _FixedJudge:
    async def evaluate(self, **kwargs):
        return JudgeFeedback(
            task_progress=0.8,
            tool_correctness=0.9,
            communication=0.7,
            improvement_hint="verify the date",
            valid=True,
        )


class _FakeEnv:
    task = SimpleNamespace(instruction="Change the return flight.")

    def step(self, action):
        return SimpleNamespace(observation="What date works?", reward=0.0, done=False)


def test_tau_bench_interaction_keeps_outcome_pure_and_emits_auxiliary_signal():
    from src.envs.tau_bench_context import make_initial_state
    from src.envs.tau_bench_interaction import TauBenchInteraction

    interaction = TauBenchInteraction({
        "judge_enabled": True,
        "judge_neutral_score": 0.5,
    })
    interaction._turn_judge = _FixedJudge()
    state = make_initial_state(0)
    state["action_history"].append({
        "tool": "update_reservation_flights",
        "parameters": {"reservation_id": "ABC123"},
        "is_error": False,
        "observation": "Reservation updated.",
    })
    interaction._instance_dict["request"] = {
        "env": _FakeEnv(),
        "state": state,
    }

    terminated, response, reward, metadata = asyncio.run(interaction.generate_response(
        "request",
        [{"role": "assistant", "content": "I can update the return flight."}],
    ))

    assert not terminated
    assert response == "What date works?"
    assert reward == pytest.approx(0.65)
    assert metadata["judge_valid"]
    assert metadata["judge_score"] == pytest.approx(0.825)
    assert metadata["judge_training_signal"] == pytest.approx(0.65)
    assert metadata["judge_improvement_hint"] == "verify the date"
    session_score = asyncio.run(interaction.calculate_score("request"))
    assert session_score["score"] == 0.0
    assert session_score["outcome_score"] == 0.0
    assert session_score["judge_mean_score"] == pytest.approx(0.825)
    assert session_score["judge_valid_turns"] == 1
    assert session_score["judge_valid_rate"] == 1.0
    assert state["last_judged_action_index"] == 1


class _UnavailableJudge:
    async def evaluate(self, **kwargs):
        return JudgeFeedback.unavailable("timeout")


def test_tau_bench_interaction_does_not_materialize_failed_judge_call():
    from src.envs.tau_bench_context import make_initial_state
    from src.envs.tau_bench_interaction import TauBenchInteraction

    interaction = TauBenchInteraction({"judge_enabled": True})
    interaction._turn_judge = _UnavailableJudge()
    interaction._instance_dict["request"] = {
        "env": _FakeEnv(),
        "state": make_initial_state(0),
    }

    _, _, reward, metadata = asyncio.run(interaction.generate_response(
        "request",
        [{"role": "assistant", "content": "I can help."}],
    ))

    assert reward is None
    assert not metadata["judge_valid"]
    assert metadata["judge_training_signal"] is None



def test_reward_consistency_flags_negative_correlation():
    result = reward_consistency([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
    assert result["correlation"] == pytest.approx(-1.0)
    assert result["conflict"]

    constant = reward_consistency([0.0, 0.0, 0.0], [0.2, 0.3, 0.4])
    assert constant["correlation"] is None
    assert not constant["conflict"]
