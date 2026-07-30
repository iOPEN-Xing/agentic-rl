from __future__ import annotations

import pytest
import torch

from verl.experimental.agent_loop.reward_utils import materialize_turn_rewards


def test_materialize_turn_rewards_targets_last_trainable_token():
    response_mask = torch.tensor([
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
    ])
    spans = [
        [
            {"start": 0, "end": 3, "reward": 0.4},
            {"start": 3, "end": 5, "reward": -0.2},
        ],
        [{"start": 0, "end": 4, "reward": 0.5}],
    ]

    rewards, event_mask = materialize_turn_rewards(spans, response_mask)

    assert rewards[0].tolist() == pytest.approx([0.0, 0.4, 0.0, -0.2, 0.0, 0.0])
    assert rewards[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    assert event_mask.sum().item() == 3


def test_materialize_turn_rewards_rejects_batch_mismatch():
    with pytest.raises(ValueError):
        materialize_turn_rewards([], torch.ones(1, 3))
