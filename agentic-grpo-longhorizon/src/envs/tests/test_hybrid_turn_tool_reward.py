import asyncio
from types import SimpleNamespace

import pytest

from src.envs.tau_bench_context import CURRENT_TAU_ENV, CURRENT_TAU_STATE, make_initial_state
from src.envs.tau_bench_interaction import TauBenchInteraction, _compute_reasoning_quality_score
from src.envs.tau_bench_tools import TauBench_think_Tool
from verl.tools.schemas import OpenAIFunctionToolSchema


class _FakeEnv:
    def step(self, action):
        assert action.name == "think"
        assert action.kwargs == {"note": "check"}
        return SimpleNamespace(observation="ok", reward=0.25, done=False)


def test_tau_bench_tool_exposes_incremental_reward_as_turn_event():
    schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "think",
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    )
    tool = TauBench_think_Tool(config={}, tool_schema=schema)
    state = make_initial_state(task_id=3)
    env_token = CURRENT_TAU_ENV.set(_FakeEnv())
    state_token = CURRENT_TAU_STATE.set(state)
    try:
        response, turn_reward, metadata = asyncio.run(
            tool.execute("instance", {"note": "check"})
        )
    finally:
        CURRENT_TAU_STATE.reset(state_token)
        CURRENT_TAU_ENV.reset(env_token)

    assert response.text == "ok"
    assert turn_reward == pytest.approx(0.25)
    assert metadata["inc_reward"] == pytest.approx(0.25)
    assert state["total_reward"] == pytest.approx(0.25)
    assert state["action_history"][-1]["inc_reward"] == pytest.approx(0.25)


class _FakeUserEnv:
    def __init__(self, reward, done):
        self.reward = reward
        self.done = done

    def step(self, action):
        return SimpleNamespace(
            observation="next user message",
            reward=self.reward,
            done=self.done,
        )


def _generate_user_event(reward, done):
    interaction = object.__new__(TauBenchInteraction)
    state = make_initial_state(task_id=5)
    interaction._instance_dict = {
        "instance": {
            "env": _FakeUserEnv(reward=reward, done=done),
            "state": state,
        }
    }
    interaction.max_turns = 30
    interaction.reward_mode = "binary"
    interaction._compute_reward = lambda current_state: 1.0
    try:
        return asyncio.run(
            interaction.generate_response(
                "instance",
                [{"role": "assistant", "content": "final answer"}],
            )
        )
    finally:
        CURRENT_TAU_STATE.set(None)
        CURRENT_TAU_ENV.set(None)


def test_user_interaction_exposes_nonterminal_incremental_reward():
    should_terminate, response, turn_reward, metadata = _generate_user_event(0.25, done=False)
    assert not should_terminate
    assert response == "next user message"
    assert turn_reward == pytest.approx(0.25)
    assert "session_score" not in metadata


def test_user_interaction_keeps_session_score_separate_at_termination():
    should_terminate, response, turn_reward, metadata = _generate_user_event(0.5, done=True)
    assert should_terminate
    assert response == ""
    assert turn_reward == pytest.approx(0.5)
    assert metadata["session_score"] == pytest.approx(1.0)


def test_prm_lite_penalizes_the_invalid_action_itself():
    action = {
        "tool": "unknown_tool",
        "parameters": {},
        "param_str": "{}",
        "inc_reward": 0.0,
        "done": False,
        "is_error": True,
        "extracted_entities": {},
        "content": "reasoned invalid call with sufficient context",
    }
    assert _compute_reasoning_quality_score([action]) == pytest.approx(-0.10)


class _FailingEnv:
    def step(self, _action):
        raise RuntimeError("backend unavailable")


def test_tau_bench_tool_marks_runtime_failure_as_untrainable():
    schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "think",
            "description": "test",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    tool = TauBench_think_Tool(config={}, tool_schema=schema)
    env_token = CURRENT_TAU_ENV.set(_FailingEnv())
    state_token = CURRENT_TAU_STATE.set(make_initial_state(task_id=3))
    try:
        _response, reward, metadata = asyncio.run(tool.execute("instance", {}))
    finally:
        CURRENT_TAU_STATE.reset(state_token)
        CURRENT_TAU_ENV.reset(env_token)

    assert reward == 0.0
    assert metadata["valid_for_training"] is False
    assert metadata["error"] == "env_step_exception"
