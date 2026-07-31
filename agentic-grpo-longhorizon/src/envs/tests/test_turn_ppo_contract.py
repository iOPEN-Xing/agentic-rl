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
