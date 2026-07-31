from __future__ import annotations

import pytest
import torch

from verl.trainer.ppo.ray_trainer import combine_outcome_and_turn_rewards


def test_turn_reward_fusion_normalizes_by_event_count():
    outcome = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    turns = torch.tensor([[0.4, 0.0, -0.2, 0.0]])
    mask = torch.tensor([[True, False, True, False]])
    config = {
        "turn_level_reward": {
            "enabled": True,
            "outcome_weight": 1.0,
            "turn_weight": 0.3,
            "normalize_by_count": True,
        }
    }

    combined, metrics = combine_outcome_and_turn_rewards(outcome, turns, mask, config)

    assert combined.sum().item() == pytest.approx(1.03)
    assert metrics["reward/turn_events_per_trajectory"] == pytest.approx(2.0)


def test_turn_reward_fusion_is_noop_when_disabled():
    outcome = torch.tensor([[0.0, 1.0]])
    combined, metrics = combine_outcome_and_turn_rewards(
        outcome,
        torch.tensor([[0.5, 0.0]]),
        torch.tensor([[True, False]]),
        {},
    )
    assert torch.equal(combined, outcome)
    assert metrics == {}
