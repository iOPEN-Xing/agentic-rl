import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.trace_reference import (
    build_trace_reference_batch,
    scatter_trace_prefix_scores,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert text == '{"gold":true}'
        assert add_special_tokens is False
        return [91, 92]


class _VariableLengthTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "short":
            return [91]
        if text == "long":
            return [91, 92, 93, 94, 95]
        raise AssertionError(f"unexpected target: {text}")


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _rollout_batch():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 10, 11]]),
            "responses": torch.tensor([[21, 22, 30, 31, 32, 41, 42]]),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            ),
        },
        non_tensors={
            "trace_state_boundaries": _object_array([[0, 3, 5]]),
            "trace_turn_spans": _object_array([[(0, 2), (3, 5)]]),
            "reward_model": _object_array([
                {"ground_truth": '{"gold":true}'}
            ]),
        },
    )


def test_reference_batch_scores_each_state_with_the_same_gold_target():
    reference = build_trace_reference_batch(
        _rollout_batch(),
        _FakeTokenizer(),
        max_scoring_length=32,
        max_target_tokens=8,
        score_temperature=1.0,
    )

    assert reference is not None
    assert reference.sample_state_indices == [(0, 0), (0, 1), (0, 2)]
    assert reference.data.batch["responses"].tolist() == [
        [91, 92],
        [91, 92],
        [91, 92],
    ]
    assert reference.target_mask.sum(dim=1).tolist() == [2, 2, 2]
    assert reference.data.meta_info["ref_log_prob_temperature"] == 1.0


def test_reference_batch_refuses_context_truncation():
    with pytest.raises(ValueError, match="silently truncating"):
        build_trace_reference_batch(
            _rollout_batch(),
            _FakeTokenizer(),
            max_scoring_length=6,
            max_target_tokens=8,
            score_temperature=1.0,
        )


def test_reference_batch_refuses_cross_sample_padding_overflow():
    rollout_batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([
                [10, 11, 12, 13, 14],
                [0, 0, 0, 15, 16],
            ]),
            "responses": torch.tensor([
                [21, 22],
                [31, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0],
            ]),
        },
        non_tensors={
            "trace_state_boundaries": _object_array([[0, 2], [0, 1]]),
            "trace_turn_spans": _object_array([[(0, 2)], [(0, 1)]]),
            "reward_model": _object_array([
                {"ground_truth": "short"},
                {"ground_truth": "long"},
            ]),
        },
    )

    with pytest.raises(ValueError, match="batch padding"):
        build_trace_reference_batch(
            rollout_batch,
            _VariableLengthTokenizer(),
            max_scoring_length=8,
            max_target_tokens=8,
            score_temperature=1.0,
        )


def test_reference_scores_are_teacher_forced_means_scattered_by_state():
    reference = build_trace_reference_batch(
        _rollout_batch(),
        _FakeTokenizer(),
        max_scoring_length=32,
        max_target_tokens=8,
        score_temperature=1.0,
    )
    ref_log_probs = torch.tensor([
        [-3.0, -1.0],
        [-2.0, -1.0],
        [-1.0, -1.0],
    ])

    scattered = scatter_trace_prefix_scores(reference, ref_log_probs)

    assert scattered[0] == pytest.approx([-2.0, -1.5, -1.0])
