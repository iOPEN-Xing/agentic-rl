from __future__ import annotations

import pytest
import torch

from verl.trainer.ppo.ray_trainer import combine_judge_turn_rewards


def test_judge_turn_reward_is_bounded_mean_contribution():
    session = torch.tensor([[0.0, 0.0, 0.0, 0.6]])
    turns = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True, False]])
    config = {"judge_turn_reward": {"enabled": True, "turn_weight": 0.1}}

    combined, metrics = combine_judge_turn_rewards(session, turns, mask, config)

    assert combined.sum().item() == pytest.approx(0.7)
    assert metrics["reward/judge_turn_events_per_trajectory"] == pytest.approx(2.0)


def test_judge_turn_weight_cannot_break_outcome_margin():
    with pytest.raises(ValueError):
        combine_judge_turn_rewards(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.bool),
            {"judge_turn_reward": {"enabled": True, "turn_weight": 0.2}},
        )
