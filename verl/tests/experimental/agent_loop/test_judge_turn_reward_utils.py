from __future__ import annotations

import pytest
import torch

from verl.experimental.agent_loop.reward_utils import materialize_turn_rewards


def test_judge_rewards_are_aligned_to_last_assistant_token():
    response_mask = torch.tensor([[1, 1, 0, 1, 1, 0]])
    spans = [[
        {"start": 0, "end": 3, "reward": 0.25},
        {"start": 3, "end": 6, "reward": 0.75},
    ]]

    rewards, event_mask = materialize_turn_rewards(spans, response_mask)

    assert rewards[0].tolist() == pytest.approx([0.0, 0.25, 0.0, 0.0, 0.75, 0.0])
    assert event_mask.sum().item() == 2
