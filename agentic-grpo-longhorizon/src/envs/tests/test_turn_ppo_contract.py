from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from src.envs.tau_bench_interaction import TauBenchInteraction
from src.envs.tau_bench_context import (
    CURRENT_ASSISTANT_CONTENT,
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TURN_PPO_CONFIG = PROJECT_ROOT / "configs/train/grpo/turn_level_reward.yaml"


def test_turn_ppo_config_selects_strict_turn_objective_at_equal_rollout_budget():
    config = yaml.safe_load(TURN_PPO_CONFIG.read_text())

    assert config["algorithm"]["adv_estimator"] == "turn_gae"
    assert config["algorithm"]["gamma"] == pytest.approx(0.99)
    assert config["algorithm"]["lam"] == pytest.approx(0.9)
    assert config["algorithm"]["use_kl_in_reward"] is False
    assert config["algorithm"]["turn_level_reward"]["enabled"] is False
    assert config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "turn_ppo"
    assert (
        config["actor_rollout_ref"]["rollout"]["multi_turn"]["interaction_config_path"]
        == "configs/interaction_config/tau_bench_airline.yaml"
    )

    task_batch = config["data"]["train_batch_size"]
    group_size = config["actor_rollout_ref"]["rollout"]["n"]
    assert task_batch * group_size == 32
    assert group_size == 1
    assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == pytest.approx(1.0e-6)
    assert config["critic"]["optim"]["lr"] == pytest.approx(1.0e-5)
    assert config["data"]["train_files"].startswith("${oc.env:AGENTIC_RL_VANILLA_DATA_ROOT,")
    assert config["data"]["train_files"].endswith("/train.parquet")
    assert config["data"]["val_files"].endswith("/val.parquet")

    # Hardware/model topology stays on the existing project contract.
    assert config["actor_rollout_ref"]["model"]["path"] == "/data/xjz/model/qwen3-8b"
    assert config["actor_rollout_ref"]["model"]["lora_rank"] == 16
    assert config["actor_rollout_ref"]["rollout"]["name"] == "vllm"
    assert config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 4
    assert config["trainer"]["n_gpus_per_node"] == 4


def test_tau_bench_reset_observation_is_exposed_as_the_initial_user_query(monkeypatch):
    class FakeEnvironment:
        def reset(self, task_index):
            assert task_index == 7
            return SimpleNamespace(observation="Please change my reservation.")

    def get_env(**kwargs):
        assert kwargs["task_index"] == 7
        return FakeEnvironment()

    fake_envs_module = ModuleType("tau_bench.envs")
    fake_envs_module.get_env = get_env
    fake_tau_bench_module = ModuleType("tau_bench")
    fake_tau_bench_module.envs = fake_envs_module
    monkeypatch.setitem(sys.modules, "tau_bench", fake_tau_bench_module)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", fake_envs_module)

    async def exercise_contract():
        interaction = TauBenchInteraction({"env_name": "airline"})
        instance_id = await interaction.start_interaction("trajectory-7", task_id=7)

        assert instance_id == "trajectory-7"
        assert (
            await interaction.get_initial_observation(instance_id)
            == "Please change my reservation."
        )
        assert CURRENT_TAU_ENV.get() is not None
        assert CURRENT_TAU_STATE.get() is not None

        CURRENT_ASSISTANT_CONTENT.set("temporary assistant content")
        await interaction.finalize_interaction(instance_id)
        assert CURRENT_TAU_ENV.get() is None
        assert CURRENT_TAU_STATE.get() is None
        assert CURRENT_ASSISTANT_CONTENT.get() is None

    asyncio.run(exercise_contract())


def test_calculate_score_requires_a_live_interaction_instance():
    interaction = TauBenchInteraction({"env_name": "airline"})

    with pytest.raises(RuntimeError, match="no initialized state"):
        asyncio.run(interaction.calculate_score("missing"))


STRICT_TURN_PPO_CONFIG = PROJECT_ROOT / "configs/train/grpo/turn_ppo_strict.yaml"


def test_turn_ppo_strict_config_selects_strict_turn_objective_at_equal_rollout_budget():
    config = yaml.safe_load(STRICT_TURN_PPO_CONFIG.read_text())

    assert config["algorithm"]["adv_estimator"] == "turn_gae"
    assert config["algorithm"]["gamma"] == pytest.approx(0.99)
    assert config["algorithm"]["lam"] == pytest.approx(0.9)
    assert config["algorithm"]["use_kl_in_reward"] is False
    assert config["algorithm"]["turn_level_reward"]["enabled"] is False
    assert config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "turn_ppo"

    task_batch = config["data"]["train_batch_size"]
    group_size = config["actor_rollout_ref"]["rollout"]["n"]
    assert task_batch * group_size == 32
    assert group_size == 1
    assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == pytest.approx(1.0e-6)
    assert config["critic"]["optim"]["lr"] == pytest.approx(1.0e-5)
    assert config["data"]["train_files"].startswith("${oc.env:AGENTIC_RL_VANILLA_DATA_ROOT,")
    assert config["data"]["train_files"].endswith("/train.parquet")
    assert config["data"]["val_files"].endswith("/val.parquet")

    assert config["trainer"]["default_local_dir"] == "experiments/turn_ppo_strict/checkpoints"
    assert config["trainer"]["experiment_name"] == "turn_ppo_strict"
    assert config["trainer"]["n_gpus_per_node"] == 4


def test_terminal_event_path_writes_assistant_turn_spans():
    """``AgentData.finalize_pending_turn`` must roll the cursor forward so a follow-on
    trajectory can never append a span that overlaps an earlier assistant turn even
    when the agent loop short-circuited on environment_done or should_terminate_sequence.
    """
    from verl.experimental.agent_loop.tool_agent_loop import AgentData

    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="req-terminal",
        tools_kwargs={},
    )

    # Simulate one assistant turn that produced 7 trainable tokens.
    agent_data.assistant_turn_spans.append((0, 7))
    agent_data.response_mask = [1] * 7
    agent_data.pending_turn_start = 0

    # Simulate environment_done early return after the assistant turn but before a
    # follow-on user message arrived.
    agent_data.finalize_pending_turn()

    assert agent_data.pending_turn_start == len(agent_data.response_mask)
    assert agent_data.assistant_turn_spans == [(0, 7)]


def test_materialize_turn_ids_writes_always():
    """``materialize_turn_ids`` must produce a valid Batch entry even when no
    samples recorded ``assistant_turn_spans``. This guards the strict turn-PPO
    fail-fast contract enforced by _validate_turn_ids.
    """
    import torch
    from verl.experimental.agent_loop.turn_ppo_utils import materialize_turn_ids

    response_mask = torch.zeros((2, 8), dtype=torch.long)
    response_mask[0, :5] = 1

    turn_ids = materialize_turn_ids([[], []], response_mask)

    assert turn_ids.shape == response_mask.shape
    assert turn_ids.dtype == torch.long
    # No trainable token was covered, so all turn ids stay zero and downstream
    # validation should raise rather than silently pass.
    assert turn_ids.sum().item() == 0


def test_megatron_turn_ids_propagation():
    """The Megatron actor must thread ``turn_ids`` through the policy loss whenever
    loss_mode=turn_ppo. We patch ``get_policy_loss_fn`` so this stays a unit test.
    """
    import types
    import sys
    import verl.trainer.ppo.core_algos as core_algos

    captured: dict[str, object] = {}

    def _policy_loss(**kwargs):
        captured["kwargs"] = kwargs
        return 0.0, {"actor/pg_clipfrac": 0.0}

    real_get = core_algos.get_policy_loss_fn
    core_algos.get_policy_loss_fn = lambda _name: _policy_loss  # type: ignore[assignment]
    try:
        config = types.SimpleNamespace(
            policy_loss={"loss_mode": "turn_ppo"},
            use_kl_loss=False,
            entropy_coeff=0.0,
            loss_agg_mode="token-mean",
        )

        _policy_loss(
            old_log_prob=None,
            log_prob=None,
            advantages=None,
            response_mask=None,
            loss_agg_mode="token-mean",
            config=config,
            rollout_is_weights=None,
            turn_ids=types.SimpleNamespace(shape=(1, 8)),
        )
    finally:
        core_algos.get_policy_loss_fn = real_get  # type: ignore[assignment]

    assert "turn_ids" in captured["kwargs"]
    assert captured["kwargs"]["turn_ids"].shape == (1, 8)


def test_turn_id_validate_handles_terminal_truncated_response():
    """``_validate_turn_ids`` should accept a multi-turn assistant layout where the
    final turn was truncated exactly at the response boundary (the behaviour our
    ``finalize_pending_turn`` helper is designed to preserve).
    """
    import torch
    from verl.trainer.ppo.core_algos import _validate_turn_ids

    response_mask = torch.ones((1, 6), dtype=torch.long)
    turn_ids = torch.zeros_like(response_mask, dtype=torch.long)
    turn_ids[0, 0:3] = 1
    turn_ids[0, 3:6] = 2

    turns = _validate_turn_ids(turn_ids, response_mask)
    assert turns == [[1, 2]]
