import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.hybrid_advantage import (
    compute_grpo_hybrid_advantage,
    compute_trace_log_ratio_values,
    compute_trace_turn_rewards,
)
from verl.trainer.ppo.ray_trainer import compute_advantage


@pytest.fixture(autouse=True)
def _ensure_hybrid_estimator_is_registered():
    core_algos.register_adv_est("grpo_hybrid")(compute_grpo_hybrid_advantage)


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _hybrid_config():
    return OmegaConf.create(
        {
            "norm_adv_by_std_in_grpo": True,
            "hybrid_advantage": {
                "outcome_weight": 1.0,
                "turn_weight": 0.2,
                "gap_epsilon": 0.1,
                "td_horizon": 3,
                "td_gamma": 0.8,
                "terminal_scale": 2.0,
            },
        }
    )


def test_log_ratio_td_reproduces_paper_appendix_numbers():
    first = compute_trace_log_ratio_values(
        torch.tensor([-5.1187, -1.5712]),
        gap_epsilon=0.001,
    )
    second = compute_trace_log_ratio_values(
        torch.tensor([-10.6570, -7.1061]),
        gap_epsilon=0.001,
    )

    assert first[0] == pytest.approx(0.0)
    assert second[0] == pytest.approx(0.0)
    assert first[1] - first[0] == pytest.approx(1.1806, abs=1e-3)
    assert second[1] - second[0] == pytest.approx(0.4052, abs=1e-3)


def test_k_step_credit_and_terminal_fill_match_algorithm_one():
    values = torch.tensor([0.0, 0.2, 0.1, 0.5])
    outcome_advantage = 1.25

    rewards = compute_trace_turn_rewards(
        state_values=values,
        outcome_advantage=outcome_advantage,
        td_horizon=3,
        td_gamma=0.8,
        terminal_scale=2.0,
    )

    deltas = [0.2, -0.1, 0.4]
    expected_first = (deltas[0] + 0.8 * deltas[1] + 0.8**2 * deltas[2]) / (1 + 0.8 + 0.8**2)
    expected_first += 2.0 * 0.8**3 * outcome_advantage
    expected_second = (deltas[1] + 0.8 * deltas[2]) / (1 + 0.8)
    expected_second += 2.0 * 0.8**2 * outcome_advantage
    expected_third = deltas[2] + 2.0 * 0.8 * outcome_advantage

    assert rewards.tolist() == pytest.approx(
        [expected_first, expected_second, expected_third]
    )


def test_k_zero_disables_dense_td_but_keeps_terminal_anchor():
    rewards = compute_trace_turn_rewards(
        state_values=torch.tensor([0.0, 0.2, 0.5]),
        outcome_advantage=-1.0,
        td_horizon=0,
        td_gamma=0.8,
        terminal_scale=2.0,
    )

    # Algorithm 1 defines K=0 as disabling c^(K). Only the final transition
    # reaches the terminal outcome when no look-ahead window is present.
    assert rewards.tolist() == pytest.approx([0.0, -1.6])


def test_hybrid_maps_trace_credit_to_transition_turns_only():
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    spans = _object_array([
        [(0, 2), (3, 5)],
        [(0, 2), (3, 5)],
    ])
    prefix_scores = _object_array([
        [-3.0, -1.0, -0.5],
        [-3.0, -2.5, -2.0],
    ])

    advantages, returns = compute_grpo_hybrid_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["task", "task"], dtype=object),
        trace_turn_spans=spans,
        trace_prefix_avg_log_probs=prefix_scores,
        config=_hybrid_config(),
    )

    # The local signal is applied to the two decision turns. The last actor
    # token (index 5) is the final-answer tail and receives outcome only.
    assert advantages[0, 0] > advantages[0, 5]
    assert advantages[1, 0] < advantages[1, 5]
    assert advantages[0, 5] == pytest.approx(1.0)
    assert advantages[1, 5] == pytest.approx(-1.0)
    assert torch.equal(advantages[:, 2], torch.zeros(2))
    assert torch.equal(advantages, returns)


def test_turn_credit_is_trajectory_local_and_not_group_normalized():
    token_level_rewards = torch.zeros(3, 2)
    response_mask = torch.ones(3, 2)
    spans = _object_array([[(0, 2)], [(0, 2)], [(0, 2)]])
    scores = _object_array([
        [-4.0, -2.0],
        [-4.0, -2.0],
        [-4.0, -3.5],
    ])

    advantages, _ = compute_grpo_hybrid_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["task", "task", "task"], dtype=object),
        trace_turn_spans=spans,
        trace_prefix_avg_log_probs=scores,
        config=_hybrid_config(),
    )

    assert advantages[0, 0] == pytest.approx(advantages[1, 0])
    assert advantages[0, 0] > advantages[2, 0] > 0


def test_trace_rejects_misaligned_prefix_scores():
    with pytest.raises(ValueError, match="one more prefix score"):
        compute_grpo_hybrid_advantage(
            token_level_rewards=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2),
            index=np.array(["task"], dtype=object),
            trace_turn_spans=_object_array([[(0, 2)]]),
            trace_prefix_avg_log_probs=_object_array([[-2.0]]),
            config=_hybrid_config(),
        )


def test_invalid_trajectory_is_excluded_from_trace_outcome_statistics():
    common = {
        "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "response_mask": torch.ones(2, 2),
        "index": np.array(["task", "task"], dtype=object),
        "trace_turn_spans": _object_array([[(0, 2)], [(0, 2)]]),
        "trace_prefix_avg_log_probs": _object_array([
            [-3.0, -1.0],
            [-3.0, -2.5],
        ]),
        "config": _hybrid_config(),
    }
    expected, _ = compute_grpo_hybrid_advantage(**common)

    actual, _ = compute_grpo_hybrid_advantage(
        token_level_rewards=torch.tensor(
            [[0.0, 1.0], [0.0, 0.0], [100.0, 100.0]]
        ),
        response_mask=torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        ),
        index=np.array(["task", "task", "task"], dtype=object),
        trace_turn_spans=_object_array([[(0, 2)], [(0, 2)], [(0, 2)]]),
        trace_prefix_avg_log_probs=_object_array([
            [-3.0, -1.0],
            [-3.0, -2.5],
            [-100.0, -0.01],
        ]),
        config=_hybrid_config(),
    )

    assert torch.allclose(actual[:2], expected, atol=1e-6)
    assert torch.equal(actual[2], torch.zeros(2))


def test_all_invalid_group_returns_zero_without_consuming_bad_trace_metadata():
    advantages, returns = compute_grpo_hybrid_advantage(
        token_level_rewards=torch.full((2, 2), 100.0),
        response_mask=torch.zeros(2, 2),
        index=np.array(["task", "task"], dtype=object),
        trace_turn_spans=_object_array([[(9, 1)], None]),
        trace_prefix_avg_log_probs=_object_array([[], None]),
        config=_hybrid_config(),
    )

    assert torch.equal(advantages, torch.zeros(2, 2))
    assert torch.equal(returns, advantages)


def test_trainer_wires_trace_scores_to_hybrid_estimator():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "response_mask": torch.ones(2, 2),
        },
        non_tensors={
            "uid": np.array(["task", "task"], dtype=object),
            "trace_turn_spans": _object_array([[(0, 2)], [(0, 2)]]),
            "trace_prefix_avg_log_probs": _object_array([
                [-3.0, -1.0],
                [-3.0, -2.5],
            ]),
        },
    )
    result = compute_advantage(
        data,
        adv_estimator="grpo_hybrid",
        config=_hybrid_config(),
    )

    assert result.batch["advantages"][0, 0] > 0
    assert result.batch["advantages"][1, 0] < 0
    assert torch.equal(result.batch["advantages"], result.batch["returns"])


def test_trainer_rejects_hybrid_rollouts_without_trace_scores():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.ones(2, 2),
            "response_mask": torch.ones(2, 2),
        },
        non_tensors={"uid": np.array(["task", "task"], dtype=object)},
    )
    with pytest.raises(ValueError, match="trace_prefix_avg_log_probs"):
        compute_advantage(data, adv_estimator="grpo_hybrid", config=_hybrid_config())
