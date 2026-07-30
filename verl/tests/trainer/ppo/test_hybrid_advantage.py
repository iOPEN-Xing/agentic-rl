import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.hybrid_advantage import (
    compute_grpo_hybrid_advantage,
    compute_hybrid_alpha,
    compute_turn_level_mc_advantage,
    normalize_advantage_component,
)
from verl.trainer.ppo.ray_trainer import compute_advantage


@pytest.fixture(autouse=True)
def _ensure_hybrid_estimator_is_registered():
    core_algos.register_adv_est("grpo_hybrid")(compute_grpo_hybrid_advantage)


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _hybrid_config(schedule="linear"):
    return OmegaConf.create(
        {
            "norm_adv_by_std_in_grpo": True,
            "hybrid_advantage": {
                "turn_gamma": 0.5,
                "session_weight_start": 0.8,
                "session_weight_end": 0.2,
                "schedule": schedule,
            },
        }
    )


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alpha_schedule_has_exact_endpoints_and_midpoint(schedule):
    assert compute_hybrid_alpha(0, 100, schedule=schedule) == pytest.approx(0.8)
    assert compute_hybrid_alpha(50, 100, schedule=schedule) == pytest.approx(0.5)
    assert compute_hybrid_alpha(100, 100, schedule=schedule) == pytest.approx(0.2)
    assert compute_hybrid_alpha(150, 100, schedule=schedule) == pytest.approx(0.2)


def test_alpha_schedule_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="positive"):
        compute_hybrid_alpha(0, 0)
    with pytest.raises(ValueError, match="Unsupported"):
        compute_hybrid_alpha(0, 10, schedule="learned")


def test_turn_mc_uses_discounted_future_events_and_maps_only_assistant_spans():
    response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    spans = _object_array([
        [(0, 2), (3, 5)],
        [(0, 2), (3, 5)],
        [(0, 2), (3, 5)],
    ])
    rewards = _object_array([
        [0.0, 2.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])

    advantages, turn_mask = compute_turn_level_mc_advantage(
        response_mask=response_mask,
        index=np.array(["task", "task", "task"], dtype=object),
        assistant_turn_spans=spans,
        assistant_turn_rewards=rewards,
        gamma=0.5,
    )

    # First-turn returns are [1, 1, 0], proving that the future reward from
    # sample 0 is discounted back to the first assistant turn.
    assert advantages[0, 0] == pytest.approx(advantages[1, 0])
    assert advantages[0, 0] > advantages[2, 0]
    # Second-turn returns are [2, 0, 0].
    assert advantages[0, 3] > advantages[1, 3]
    assert advantages[1, 3] == pytest.approx(advantages[2, 3])
    assert torch.equal(advantages[:, 2], torch.zeros(3))
    assert torch.equal(turn_mask, response_mask)


def test_turn_mc_rejects_misaligned_events():
    with pytest.raises(ValueError, match="mismatch"):
        compute_turn_level_mc_advantage(
            response_mask=torch.ones(1, 2),
            index=np.array(["task"], dtype=object),
            assistant_turn_spans=_object_array([[(0, 2)]]),
            assistant_turn_rewards=_object_array([[]]),
        )


def test_component_normalization_equalizes_scale_on_its_mask():
    values = torch.tensor([[10.0, 20.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    normalized = normalize_advantage_component(values, mask)
    valid = normalized[mask.bool()]

    assert valid.mean() == pytest.approx(0.0)
    assert valid.std(unbiased=False) == pytest.approx(1.0, abs=1e-5)
    assert normalized[0, 2] == 0.0


def test_hybrid_schedule_moves_credit_from_session_to_conflicting_turn_signal():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    response_mask = torch.ones(2, 2)
    uid = np.array(["task", "task"], dtype=object)
    spans = _object_array([[(0, 2)], [(0, 2)]])
    turn_rewards = _object_array([[-1.0], [1.0]])

    early, _ = compute_grpo_hybrid_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=uid,
        assistant_turn_spans=spans,
        assistant_turn_rewards=turn_rewards,
        global_step=0,
        total_steps=100,
        config=_hybrid_config(),
    )
    late, _ = compute_grpo_hybrid_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=uid,
        assistant_turn_spans=spans,
        assistant_turn_rewards=turn_rewards,
        global_step=100,
        total_steps=100,
        config=_hybrid_config(),
    )

    assert early[0, 0] > 0
    assert late[0, 0] < 0
    assert early[1, 0] < 0
    assert late[1, 0] > 0


def test_hybrid_is_invariant_to_turn_reward_scale():
    kwargs = {
        "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "response_mask": torch.ones(2, 2),
        "index": np.array(["task", "task"], dtype=object),
        "assistant_turn_spans": _object_array([[(0, 2)], [(0, 2)]]),
        "global_step": 50,
        "total_steps": 100,
        "config": _hybrid_config(),
    }
    base, _ = compute_grpo_hybrid_advantage(
        **kwargs,
        assistant_turn_rewards=_object_array([[-1.0], [1.0]]),
    )
    scaled, _ = compute_grpo_hybrid_advantage(
        **kwargs,
        assistant_turn_rewards=_object_array([[-100.0], [100.0]]),
    )
    assert torch.allclose(base, scaled, atol=1e-5)


def test_trainer_wires_turn_events_and_global_step_to_hybrid_estimator():
    tensors = {
        "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "response_mask": torch.ones(2, 2),
    }
    non_tensors = {
        "uid": np.array(["task", "task"], dtype=object),
        "assistant_turn_spans": _object_array([[(0, 2)], [(0, 2)]]),
        "assistant_turn_rewards": _object_array([[-1.0], [1.0]]),
    }
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    result = compute_advantage(
        data,
        adv_estimator="grpo_hybrid",
        config=_hybrid_config(),
        global_step=100,
        total_steps=100,
    )
    assert result.batch["advantages"][0, 0] < 0
    assert torch.equal(result.batch["advantages"], result.batch["returns"])


def test_trainer_rejects_hybrid_rollouts_without_turn_events():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.ones(2, 2),
            "response_mask": torch.ones(2, 2),
        },
        non_tensors={"uid": np.array(["task", "task"], dtype=object)},
    )
    with pytest.raises(ValueError, match="assistant_turn_spans"):
        compute_advantage(data, adv_estimator="grpo_hybrid", config=_hybrid_config())
