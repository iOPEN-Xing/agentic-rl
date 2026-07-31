import asyncio
from functools import wraps
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopWorkerBase,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
    merge_initial_user_message,
)
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.interactions.base import BaseInteraction
from verl.tools.schemas import ToolResponse

from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    make_initial_state,
)
from src.envs.tau_bench_interaction import TauBenchInteraction


def sync_async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _FinalizeTrackingInteraction(BaseInteraction):
    def __init__(self, fail_start=False):
        super().__init__({})
        self.fail_start = fail_start
        self.finalized = []

    async def start_interaction(self, instance_id=None, **kwargs):
        if self.fail_start:
            raise RuntimeError("reset failed")
        return instance_id

    async def get_initial_observation(self, instance_id, **kwargs):
        return "initial observation"

    async def calculate_score(self, instance_id, **kwargs):
        return 1.0

    async def finalize_interaction(self, instance_id, **kwargs):
        self.finalized.append(instance_id)


def _minimal_loop(interaction, pending_handler):
    loop = object.__new__(ToolAgentLoop)
    loop.interaction_config_file = "interaction.yaml"
    loop.interaction_map = {"tau": interaction}
    loop.response_length = 8
    loop._handle_pending_state = pending_handler
    return loop


def test_initial_reset_observation_is_injected_exactly_once():
    messages = [{"role": "system", "content": "system"}]
    assert merge_initial_user_message(messages, "initial observation")
    assert messages[-1] == {"role": "user", "content": "initial observation"}
    assert not merge_initial_user_message(messages, "initial observation")


def test_mismatched_parquet_user_message_is_rejected():
    messages = [{"role": "user", "content": "stale observation"}]
    with pytest.raises(ValueError, match="does not match"):
        merge_initial_user_message(messages, "reset observation")


@sync_async_test
async def test_run_finalizes_success_and_converts_runtime_failure_to_invalid_rollout():
    for should_raise in (False, True):
        interaction = _FinalizeTrackingInteraction()

        async def _pending(agent_data, _sampling_params):
            agent_data.prompt_ids = [1, 2]
            if should_raise:
                raise RuntimeError("rollout failed")
            return AgentState.TERMINATED

        loop = _minimal_loop(interaction, _pending)
        output = await loop.run(
            {},
            raw_prompt=[{"role": "system", "content": "system"}],
            extra_info={"interaction_kwargs": {"name": "tau"}},
        )
        assert interaction.finalized
        assert output.extra_fields["valid_for_training"] is (not should_raise)
        assert output.reward_score == (0.0 if should_raise else 1.0)


@sync_async_test
async def test_start_failure_still_finalizes_and_is_invalid():
    interaction = _FinalizeTrackingInteraction(fail_start=True)

    async def _unused(_agent_data, _sampling_params):
        raise AssertionError("pending state must not run")

    output = await _minimal_loop(interaction, _unused).run(
        {},
        raw_prompt=[{"role": "system", "content": "system"}],
        extra_info={"interaction_kwargs": {"name": "tau"}},
    )
    assert len(interaction.finalized) == 1
    assert output.extra_fields["valid_for_training"] is False
    assert output.reward_score == 0.0


class _Tokenizer:
    def apply_chat_template(self, _messages, **_kwargs):
        return [99, 10]


@sync_async_test
@pytest.mark.parametrize(
    ("metadata", "valid_for_training"),
    [
        ({"done": True, "valid_for_training": True}, True),
        ({"valid_for_training": False, "detail": "service unavailable"}, False),
    ],
)
async def test_tool_metadata_terminates_loop(metadata, valid_for_training):
    loop = object.__new__(ToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.processor = None
    loop.tool_parser_name = "hermes"
    loop.tokenizer = _Tokenizer()
    loop.loop = asyncio.get_running_loop()
    loop.system_prompt = [99]
    loop.response_length = 16

    executed = []

    async def _call_tool(tool_call, _tools_kwargs):
        executed.append(tool_call.name)
        return ToolResponse(text="terminal"), 1.0, metadata

    loop._call_tool = _call_tool
    agent_data = AgentData([], None, {}, "request", {})
    agent_data.prompt_ids = [7]
    agent_data.response_mask = [1]
    agent_data.assistant_turn_rewards = [0.0]
    agent_data.tool_calls = [SimpleNamespace(name="finish"), SimpleNamespace(name="not_executed")]

    state = await loop._handle_processing_tools_state(agent_data)

    assert state == AgentState.TERMINATED
    assert agent_data.valid_for_training is valid_for_training
    assert agent_data.total_tool_calls == 1
    assert executed == ["finish"]


@sync_async_test
async def test_tau_finalize_clears_state_and_contextvars():
    interaction = TauBenchInteraction({"reward_mode": "binary"})
    env = object()
    state = {}
    interaction._instance_dict["request"] = {"env": env, "state": state}
    CURRENT_TAU_ENV.set(env)
    CURRENT_TAU_STATE.set(state)
    CURRENT_ASSISTANT_CONTENT.set("assistant")

    await interaction.finalize_interaction("request")

    assert "request" not in interaction._instance_dict
    assert CURRENT_TAU_ENV.get() is None
    assert CURRENT_TAU_STATE.get() is None
    assert CURRENT_ASSISTANT_CONTENT.get() is None


@sync_async_test
@pytest.mark.parametrize(
    "call",
    [
        SimpleNamespace(name="unknown_tool", arguments="{}", parse_error=None, raw_call=None),
        SimpleNamespace(name="tool", arguments="{bad", parse_error=None, raw_call=None),
        SimpleNamespace(name="<malformed_tool_call>", arguments="{}", parse_error="bad JSON", raw_call="{bad"),
    ],
)
async def test_invalid_tool_call_remains_a_trainable_policy_error(call):
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {}
    state = make_initial_state(1)
    token = CURRENT_TAU_STATE.set(state)
    try:
        response, reward, metadata = await loop._call_tool(call, {})
    finally:
        CURRENT_TAU_STATE.reset(token)

    assert response.text.startswith("Error:")
    assert reward == 0.0
    assert metadata["valid_for_training"] is True
    assert metadata["policy_error"] is True
    assert state["action_history"][-1]["is_error"] is True


@sync_async_test
async def test_hermes_parser_preserves_malformed_call_as_policy_action():
    class _MalformedTokenizer:
        def decode(self, _token_ids):
            return '<tool_call>{"name" "broken", "arguments": {}}</tool_call>'

    content, calls = await HermesToolParser(_MalformedTokenizer()).extract_tool_calls([1, 2])

    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "<malformed_tool_call>"
    assert calls[0].parse_error
    assert calls[0].raw_call == '{"name" "broken", "arguments": {}}'


@sync_async_test
async def test_tool_runtime_exception_is_not_trained_as_policy_failure():
    class _FailingTool:
        released = False

        async def create(self, **kwargs):
            return "tool-instance", ToolResponse()

        async def execute(self, instance_id, parameters):
            raise RuntimeError("service unavailable")

        async def release(self, instance_id):
            self.released = True

    tool = _FailingTool()
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"known_tool": tool}

    _response, reward, metadata = await loop._call_tool(
        SimpleNamespace(name="known_tool", arguments="{}", parse_error=None, raw_call=None), {}
    )

    assert reward == 0.0
    assert metadata["valid_for_training"] is False
    assert metadata["error"] == "tool_runtime_exception"
    assert tool.released


@sync_async_test
async def test_tau_start_exposes_reset_observation(monkeypatch):
    class _FakeEnv:
        def __init__(self):
            self.reset_task_ids = []

        def reset(self, task_index=None):
            self.reset_task_ids.append(task_index)
            return SimpleNamespace(observation="initial user request")

    fake_env = _FakeEnv()
    monkeypatch.setattr("tau_bench.envs.get_env", lambda **_kwargs: fake_env)
    interaction = TauBenchInteraction({"reward_mode": "binary"})

    instance_id = await interaction.start_interaction("request", task_id=7)
    try:
        assert instance_id == "request"
        assert await interaction.get_initial_observation(instance_id) == "initial user request"
        assert fake_env.reset_task_ids == [7]
    finally:
        await interaction.finalize_interaction("request")


def _internal_output(valid_for_training):
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]]),
        response_ids=torch.tensor([[3, 4, 0, 0]]),
        input_ids=torch.tensor([[1, 2, 3, 4, 0, 0]]),
        position_ids=torch.tensor([[0, 1, 2, 3, 0, 0]]),
        response_mask=torch.tensor([[1, 1, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]),
        reward_score=1.0 if valid_for_training else 0.0,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"assistant_turn_spans": [(0, 2)], "valid_for_training": valid_for_training},
    )


def test_postprocess_zeroes_every_trainable_token_for_invalid_trajectory():
    worker = object.__new__(AgentLoopWorkerBase)
    batch = worker._postprocess([_internal_output(True), _internal_output(False)])

    assert torch.equal(batch.batch["valid_for_training"], torch.tensor([True, False]))
    assert torch.equal(batch.batch["response_mask"][0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(batch.batch["response_mask"][1], torch.zeros(4, dtype=torch.long))
