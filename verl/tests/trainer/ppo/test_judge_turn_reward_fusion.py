from __future__ import annotations

import pytest
import torch

from verl.trainer.ppo.ray_trainer import (
    combine_judge_turn_rewards,
    summarize_judge_diagnostics,
)


def test_judge_turn_reward_is_bounded_mean_contribution():
    session = torch.tensor([[0.0, 0.0, 0.0, 0.6]])
    turns = torch.tensor([[1.0, 0.0, -0.5, 0.0]])
    mask = torch.tensor([[True, False, True, False]])
    config = {"judge_turn_reward": {"enabled": True, "turn_weight": 0.1}}

    combined, metrics = combine_judge_turn_rewards(session, turns, mask, config)

    assert combined.sum().item() == pytest.approx(0.625)
    assert metrics["reward/judge_turn_events_per_trajectory"] == pytest.approx(2.0)


def test_judge_turn_weight_cannot_break_outcome_margin():
    with pytest.raises(ValueError):
        combine_judge_turn_rewards(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.bool),
            {"judge_turn_reward": {"enabled": True, "turn_weight": 0.2}},
        )


def test_judge_turn_score_must_be_centered_and_bounded():
    with pytest.raises(ValueError):
        combine_judge_turn_rewards(
            torch.zeros(1, 2),
            torch.tensor([[1.1, 0.0]]),
            torch.tensor([[True, False]]),
            {"judge_turn_reward": {"enabled": True, "turn_weight": 0.05}},
        )


def test_judge_diagnostics_are_computed_across_trajectories():
    metrics = summarize_judge_diagnostics({
        "outcome_reward": [0.0, 0.0, 1.0, 1.0],
        "judge_mean_score": [0.1, 0.2, 0.8, 0.9],
        "judge_valid_rate": [1.0, 0.5, 1.0, 1.0],
    })

    assert metrics["reward/judge_valid_rate"] == pytest.approx(0.875)
    assert metrics["reward/judge_score_mean"] == pytest.approx(0.5)
    assert metrics["reward/judge_outcome_correlation"] > 0.9
    assert metrics["reward/judge_outcome_conflict"] == 0.0
