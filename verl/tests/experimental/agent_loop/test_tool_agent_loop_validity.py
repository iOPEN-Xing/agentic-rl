from __future__ import annotations

import asyncio
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
    _merge_initial_observation,
)
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.schemas import ToolResponse
from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_STATE,
    make_initial_state,
)


def test_initial_observation_is_appended_once_and_existing_content_is_strictly_validated():
    messages = [{"role": "system", "content": "system"}]
    _merge_initial_observation(messages, "initial task")
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
    ]

    _merge_initial_observation(messages, "initial task")
    assert len(messages) == 2

    with pytest.raises(ValueError, match="does not match"):
        _merge_initial_observation(messages, "different task")


def _make_loop(interaction):
    loop = object.__new__(ToolAgentLoop)
    loop.interaction_config_file = "interaction.yaml"
    loop.interaction_map = {"tau": interaction}
    loop.response_length = 16

    async def terminate_immediately(agent_data, sampling_params):
        interaction.observed_messages = list(agent_data.messages)
        return AgentState.TERMINATED

    loop._handle_pending_state = terminate_immediately
    return loop


def _run_loop(loop, raw_prompt):
    run_impl = getattr(ToolAgentLoop.run, "__wrapped__", ToolAgentLoop.run)
    return asyncio.run(
        run_impl(
            loop,
            {},
            raw_prompt=raw_prompt,
            extra_info={"interaction_kwargs": {"name": "tau"}},
        )
    )


class _FakeInteraction:
    def __init__(
        self,
        *,
        score_error: bool = False,
        start_error: bool = False,
        initial_observation: str | None = "initial task",
    ):
        self.started = False
        self.finalized = False
        self.score_error = score_error
        self.start_error = start_error
        self.initial_observation = initial_observation
        self.observed_messages = None

    async def start_interaction(self, instance_id, **kwargs):
        self.started = True
        if self.start_error:
            raise RuntimeError("environment reset failed")

    async def get_initial_observation(self, instance_id, **kwargs):
        return self.initial_observation

    async def calculate_score(self, instance_id):
        if self.score_error:
            raise RuntimeError("score service unavailable")
        return 1.0

    async def finalize_interaction(self, instance_id):
        self.finalized = True


def test_run_does_not_duplicate_an_existing_initial_user_message_and_always_finalizes():
    interaction = _FakeInteraction()
    output = _run_loop(
        _make_loop(interaction),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "initial task"},
        ],
    )

    assert interaction.started is True
    assert interaction.finalized is True
    assert interaction.observed_messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
    ]
    assert output.extra_fields["valid_for_training"] is True
    assert output.reward_score == pytest.approx(1.0)


def test_none_initial_observation_preserves_dataset_prompt_for_generic_interactions():
    interaction = _FakeInteraction(initial_observation=None)
    raw_prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "dataset task"},
    ]
    output = _run_loop(_make_loop(interaction), raw_prompt)

    assert interaction.observed_messages == raw_prompt
    assert interaction.finalized is True
    assert output.extra_fields["valid_for_training"] is True


def test_initial_message_mismatch_is_invalid_and_still_finalizes():
    interaction = _FakeInteraction()
    output = _run_loop(
        _make_loop(interaction),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "stale dataset task"},
        ],
    )

    assert interaction.finalized is True
    assert output.extra_fields["valid_for_training"] is False
    assert output.reward_score == pytest.approx(0.0)
    assert "does not match" in output.extra_fields["failure_reason"]


def test_score_failure_is_invalid_and_still_finalizes():
    interaction = _FakeInteraction(score_error=True)
    output = _run_loop(
        _make_loop(interaction),
        [{"role": "system", "content": "system"}],
    )

    assert interaction.finalized is True
    assert output.extra_fields["valid_for_training"] is False
    assert output.reward_score == pytest.approx(0.0)
    assert "score service unavailable" in output.extra_fields["failure_reason"]


def test_start_failure_is_invalid_and_still_finalizes():
    interaction = _FakeInteraction(start_error=True)
    output = _run_loop(
        _make_loop(interaction),
        [{"role": "system", "content": "system"}],
    )

    assert interaction.started is True
    assert interaction.finalized is True
    assert output.extra_fields["valid_for_training"] is False
    assert "environment reset failed" in output.extra_fields["failure_reason"]


def test_failure_before_first_generation_does_not_move_prompt_tokens_into_response():
    interaction = _FakeInteraction()
    loop = _make_loop(interaction)

    async def fail_after_tokenizing_prompt(agent_data, sampling_params):
        agent_data.prompt_ids = [11, 12, 13]
        raise RuntimeError("generation backend unavailable")

    loop._handle_pending_state = fail_after_tokenizing_prompt
    output = _run_loop(loop, [{"role": "system", "content": "system"}])

    assert output.prompt_ids == [11, 12, 13]
    assert output.response_ids == []
    assert output.response_mask == []
    assert output.extra_fields["valid_for_training"] is False


def test_malformed_json_and_unknown_tool_are_trainable_policy_errors():
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {}
    state = make_initial_state(task_id=3)
    state_token = CURRENT_TAU_STATE.set(state)
    content_token = CURRENT_ASSISTANT_CONTENT.set("I will use the requested tool.")
    try:
        malformed = asyncio.run(
            loop._call_tool(SimpleNamespace(name="known", arguments="{"), {})
        )
        unknown = asyncio.run(
            loop._call_tool(SimpleNamespace(name="missing", arguments='{"id": 7}'), {})
        )
    finally:
        CURRENT_ASSISTANT_CONTENT.reset(content_token)
        CURRENT_TAU_STATE.reset(state_token)

    assert malformed[2] == {
        "error": "invalid_tool_arguments",
        "policy_error": True,
        "valid_for_training": True,
    }
    assert unknown[2] == {
        "error": "unknown_tool",
        "policy_error": True,
        "valid_for_training": True,
    }
    assert state["num_tool_calls"] == 2
    assert [action["tool"] for action in state["action_history"]] == ["known", "missing"]
    assert all(action["is_error"] for action in state["action_history"])
    assert [action["error_type"] for action in state["action_history"]] == [
        "invalid_tool_arguments",
        "unknown_tool",
    ]
    assert state["action_history"][1]["parameters"] == {"id": 7}
    assert state["action_history"][0]["content"] == "I will use the requested tool."


def test_hermes_parser_preserves_malformed_tool_call_as_a_policy_action():
    class FakeTokenizer:
        def decode(self, token_ids):
            return '<tool_call>{"name" "broken", "arguments": {}}</tool_call>'

    parser = HermesToolParser(FakeTokenizer())
    content, calls = asyncio.run(parser.extract_tool_calls([1, 2]))

    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "<malformed_tool_call>"
    assert calls[0].parse_error
    assert calls[0].raw_call == '{"name" "broken", "arguments": {}}'

    loop = object.__new__(ToolAgentLoop)
    loop.tools = {}
    state = make_initial_state(task_id=4)
    state_token = CURRENT_TAU_STATE.set(state)
    try:
        response = asyncio.run(loop._call_tool(calls[0], {}))
    finally:
        CURRENT_TAU_STATE.reset(state_token)

    assert response[2]["error"] == "invalid_tool_arguments"
    assert response[2]["valid_for_training"] is True
    assert state["num_tool_calls"] == 1
    assert state["action_history"][0]["tool"] == "<malformed_tool_call>"
    assert state["action_history"][0]["is_error"] is True
    assert state["action_history"][0]["raw_arguments"] == calls[0].raw_call


def test_unknown_tool_without_tau_context_remains_a_recoverable_policy_error():
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {}
    state_token = CURRENT_TAU_STATE.set(None)
    try:
        response = asyncio.run(
            loop._call_tool(SimpleNamespace(name="missing", arguments="{}"), {})
        )
    finally:
        CURRENT_TAU_STATE.reset(state_token)

    assert response[0].text == "Error: unknown tool 'missing'"
    assert response[2] == {
        "error": "unknown_tool",
        "policy_error": True,
        "valid_for_training": True,
    }


def test_tool_runtime_failure_is_not_trainable_and_release_still_runs():
    class FailingTool:
        def __init__(self):
            self.released = False

        async def create(self, **kwargs):
            return "tool-instance", ToolResponse()

        async def execute(self, instance_id, arguments):
            raise RuntimeError("environment transport failed")

        async def release(self, instance_id):
            self.released = True

    tool = FailingTool()
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"known": tool}

    response = asyncio.run(
        loop._call_tool(SimpleNamespace(name="known", arguments="{}"), {})
    )

    assert tool.released is True
    assert response[2]["valid_for_training"] is False
    assert response[2]["error"] == "tool_runtime_exception"


def test_done_tool_response_terminates_without_counting_unexecuted_parallel_calls():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [99, 10]

    loop = object.__new__(ToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 1024
    loop.tool_response_truncate_side = "left"
    loop.processor = None
    loop.tool_parser_name = "hermes"
    loop.tokenizer = FakeTokenizer()
    loop.system_prompt = [99]
    loop.response_length = 32
    executed = []

    async def call_tool(tool_call, tools_kwargs):
        executed.append(tool_call.name)
        return ToolResponse(text="environment finished"), 0.0, {"done": True, "tool": tool_call.name}

    loop._call_tool = call_tool
    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="request",
        tools_kwargs={},
    )
    agent_data.tool_calls = [
        SimpleNamespace(name="first", arguments="{}"),
        SimpleNamespace(name="second", arguments="{}"),
    ]

    async def exercise_state():
        loop.loop = asyncio.get_running_loop()
        return await loop._handle_processing_tools_state(agent_data)

    state = asyncio.run(exercise_state())

    assert state is AgentState.TERMINATED
    assert executed == ["first"]
    assert agent_data.total_tool_calls == 1


def _internal_output(valid_for_training: bool):
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
        extra_fields={
            "assistant_turn_spans": [(0, 2)],
            "valid_for_training": valid_for_training,
        },
    )


def test_postprocess_zeroes_every_trainable_token_for_invalid_trajectory():
    worker = object.__new__(AgentLoopWorkerBase)
    batch = worker._postprocess([_internal_output(True), _internal_output(False)])

    assert torch.equal(batch.batch["valid_for_training"], torch.tensor([True, False]))
    assert torch.equal(batch.batch["response_mask"][0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(batch.batch["response_mask"][1], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch.batch["turn_ids"][0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(batch.batch["turn_ids"][1], torch.zeros(4, dtype=torch.long))
