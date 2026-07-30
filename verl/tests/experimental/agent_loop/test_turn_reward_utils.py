import math

import pytest

from verl.experimental.agent_loop.turn_reward_utils import (
    accumulate_latest_turn_reward,
    append_assistant_turn,
    clip_turn_reward_events,
)


def test_turn_events_accumulate_and_clip_in_alignment():
    spans = []
    rewards = []

    append_assistant_turn(spans, rewards, 0, 3)
    assert accumulate_latest_turn_reward(rewards, 0.25)
    assert accumulate_latest_turn_reward(rewards, -0.05)
    append_assistant_turn(spans, rewards, 5, 9)
    assert accumulate_latest_turn_reward(rewards, 1.0)

    clipped_spans, clipped_rewards = clip_turn_reward_events(spans, rewards, response_length=7)
    assert clipped_spans == [(0, 3), (5, 7)]
    assert clipped_rewards == pytest.approx([0.2, 1.0])


def test_turn_event_validation_fails_loud():
    with pytest.raises(ValueError, match="before any assistant"):
        accumulate_latest_turn_reward([], 1.0)
    with pytest.raises(ValueError, match="finite"):
        accumulate_latest_turn_reward([0.0], math.inf)
    with pytest.raises(ValueError, match="overlap"):
        append_assistant_turn([(0, 3)], [0.0], 2, 4)
    with pytest.raises(ValueError, match="mismatch"):
        clip_turn_reward_events([(0, 2)], [], response_length=2)
