import pytest
import torch

from verl.experimental.agent_loop.turn_ppo_utils import (
    append_assistant_turn,
    clip_assistant_turn_spans,
    materialize_turn_ids,
)


def test_materialize_turn_ids_assigns_each_assistant_generation_its_own_turn():
    response_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0]], dtype=torch.long)
    spans = [[(0, 2), (4, 7)]]

    turn_ids = materialize_turn_ids(spans, response_mask)

    assert torch.equal(turn_ids, torch.tensor([[1, 1, 0, 0, 2, 2, 2, 0]]))


def test_materialize_turn_ids_rejects_unassigned_trainable_tokens():
    response_mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="not covered"):
        materialize_turn_ids([[(0, 2)]], response_mask)


def test_materialize_turn_ids_rejects_overlapping_spans():
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="overlap"):
        materialize_turn_ids([[(0, 3), (2, 4)]], response_mask)


def test_clip_assistant_turn_spans_clips_only_the_truncated_final_turn():
    clipped = clip_assistant_turn_spans([(0, 2), (4, 9)], response_length=7)

    assert clipped == [(0, 2), (4, 7)]


def test_append_assistant_turn_rejects_empty_or_non_monotonic_spans():
    spans = [(0, 2)]

    with pytest.raises(ValueError, match="positive length"):
        append_assistant_turn(spans, 2, 2)
    with pytest.raises(ValueError, match="overlap"):
        append_assistant_turn(spans, 1, 3)
