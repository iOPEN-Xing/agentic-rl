# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Build and reduce frozen-reference scoring batches for TRACE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class TraceReferenceBatch:
    data: DataProto
    sample_state_indices: list[tuple[int, int]]
    target_mask: torch.Tensor
    original_batch_size: int


def _pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: int,
    *,
    left: bool,
) -> torch.Tensor:
    max_length = max(int(sequence.numel()) for sequence in sequences)
    padded = torch.full(
        (len(sequences), max_length),
        fill_value=pad_value,
        dtype=torch.long,
    )
    for row_index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        if left:
            padded[row_index, max_length - length :] = sequence
        else:
            padded[row_index, :length] = sequence
    return padded


def _extract_gold_target(reward_model_entry: Any, sample_index: int) -> str:
    if not isinstance(reward_model_entry, dict):
        raise ValueError(
            f"TRACE sample {sample_index} requires reward_model to be a dict"
        )
    target = reward_model_entry.get("ground_truth")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            f"TRACE sample {sample_index} requires a non-empty training-only ground_truth"
        )
    return target


def build_trace_reference_batch(
    rollout_batch: DataProto,
    tokenizer,
    *,
    max_scoring_length: int,
    max_target_tokens: int,
    score_temperature: float,
) -> TraceReferenceBatch | None:
    """Create prefix+gold-target sequences for one batched reference forward.

    Prefix states are response-token offsets recorded immediately before
    assistant generation. The gold target is never inserted into the actor
    rollout; it exists only in this separate teacher-forced scoring batch.
    """
    required_tensor_keys = ("prompts", "responses", "attention_mask")
    missing_tensors = [
        key for key in required_tensor_keys if key not in rollout_batch.batch
    ]
    if missing_tensors:
        raise ValueError(f"TRACE scoring batch is missing tensors {missing_tensors}")
    required_non_tensor_keys = (
        "trace_state_boundaries",
        "trace_turn_spans",
        "reward_model",
    )
    missing_non_tensors = [
        key
        for key in required_non_tensor_keys
        if key not in rollout_batch.non_tensor_batch
    ]
    if missing_non_tensors:
        raise ValueError(
            f"TRACE scoring batch is missing metadata {missing_non_tensors}"
        )
    if max_scoring_length <= 0:
        raise ValueError(
            f"max_scoring_length must be positive, got {max_scoring_length}"
        )
    if max_target_tokens <= 0:
        raise ValueError(
            f"max_target_tokens must be positive, got {max_target_tokens}"
        )
    if score_temperature <= 0:
        raise ValueError(
            f"score_temperature must be positive, got {score_temperature}"
        )

    prompts = rollout_batch.batch["prompts"].detach().cpu()
    responses = rollout_batch.batch["responses"].detach().cpu()
    attention_mask = rollout_batch.batch["attention_mask"].detach().cpu()
    prompt_width = prompts.shape[1]
    batch_size = prompts.shape[0]

    prefix_sequences: list[torch.Tensor] = []
    target_sequences: list[torch.Tensor] = []
    sample_state_indices: list[tuple[int, int]] = []
    for sample_index in range(batch_size):
        turn_spans = list(
            rollout_batch.non_tensor_batch["trace_turn_spans"][sample_index]
            or []
        )
        state_boundaries = list(
            rollout_batch.non_tensor_batch["trace_state_boundaries"][sample_index]
            or []
        )
        if not turn_spans:
            continue
        if len(state_boundaries) != len(turn_spans) + 1:
            raise ValueError(
                "TRACE requires one more state boundary than transition span for "
                f"sample {sample_index}: {len(state_boundaries)} != "
                f"{len(turn_spans)} + 1"
            )

        target_text = _extract_gold_target(
            rollout_batch.non_tensor_batch["reward_model"][sample_index],
            sample_index,
        )
        target_token_ids = tokenizer.encode(
            target_text,
            add_special_tokens=False,
        )
        if not target_token_ids:
            raise ValueError(f"TRACE target for sample {sample_index} tokenized empty")
        if len(target_token_ids) > max_target_tokens:
            raise ValueError(
                f"TRACE target for sample {sample_index} has "
                f"{len(target_token_ids)} tokens, exceeding {max_target_tokens}"
            )
        target = torch.tensor(target_token_ids, dtype=torch.long)

        prompt_mask = attention_mask[sample_index, :prompt_width].bool()
        prompt = prompts[sample_index][prompt_mask]
        actual_response_length = int(
            attention_mask[sample_index, prompt_width:].sum().item()
        )
        previous_boundary = -1
        for state_index, boundary_value in enumerate(state_boundaries):
            boundary = int(boundary_value)
            if (
                boundary < 0
                or boundary <= previous_boundary
                or boundary > actual_response_length
            ):
                raise ValueError(
                    f"Invalid TRACE state boundary for sample {sample_index}: "
                    f"{state_boundaries!r}, actual response length "
                    f"{actual_response_length}"
                )
            previous_boundary = boundary
            prefix = torch.cat(
                (prompt, responses[sample_index, :boundary]),
                dim=0,
            )
            if prefix.numel() + target.numel() > max_scoring_length:
                raise ValueError(
                    f"TRACE prefix+target for sample {sample_index}, state "
                    f"{state_index} has {prefix.numel() + target.numel()} tokens, "
                    f"exceeding max_scoring_length={max_scoring_length}; "
                    "silently truncating would change the state-value definition"
                )
            prefix_sequences.append(prefix)
            target_sequences.append(target)
            sample_state_indices.append((sample_index, state_index))

    if not prefix_sequences:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("TRACE reference scoring requires a pad or EOS token id")

    padded_prompts = _pad_sequences(
        prefix_sequences,
        int(pad_token_id),
        left=True,
    )
    padded_targets = _pad_sequences(
        target_sequences,
        int(pad_token_id),
        left=False,
    )
    padded_scoring_width = padded_prompts.shape[1] + padded_targets.shape[1]
    if padded_scoring_width > max_scoring_length:
        raise ValueError(
            "TRACE batch padding would create a scoring tensor with "
            f"{padded_scoring_width} tokens, exceeding "
            f"max_scoring_length={max_scoring_length}. Each prefix-target pair "
            "fits individually, but combining the longest prefix and longest "
            "target from different samples does not; split the batch instead "
            "of silently exceeding the scoring limit."
        )
    prompt_mask = torch.zeros_like(padded_prompts, dtype=torch.bool)
    for row_index, prefix in enumerate(prefix_sequences):
        prompt_mask[row_index, -prefix.numel() :] = True
    target_mask = torch.zeros_like(padded_targets, dtype=torch.bool)
    for row_index, target in enumerate(target_sequences):
        target_mask[row_index, : target.numel()] = True

    input_ids = torch.cat((padded_prompts, padded_targets), dim=1)
    full_attention_mask = torch.cat((prompt_mask, target_mask), dim=1).long()
    position_ids = compute_position_id_with_mask(full_attention_mask)
    scoring_data = DataProto.from_dict(
        tensors={
            "prompts": padded_prompts,
            "responses": padded_targets,
            "response_mask": target_mask.long(),
            "input_ids": input_ids,
            "attention_mask": full_attention_mask,
            "position_ids": position_ids,
        },
        meta_info={"ref_log_prob_temperature": float(score_temperature)},
    )
    return TraceReferenceBatch(
        data=scoring_data,
        sample_state_indices=sample_state_indices,
        target_mask=target_mask,
        original_batch_size=batch_size,
    )


def scatter_trace_prefix_scores(
    reference_batch: TraceReferenceBatch | None,
    ref_log_probs: torch.Tensor | None,
) -> np.ndarray:
    """Average target-token log-probabilities and scatter them by rollout."""
    result = np.empty(
        0 if reference_batch is None else reference_batch.original_batch_size,
        dtype=object,
    )
    if reference_batch is None:
        return result
    result[:] = [[] for _ in range(reference_batch.original_batch_size)]
    if ref_log_probs is None:
        raise ValueError("TRACE reference log-probabilities are required")
    scores = ref_log_probs.detach().cpu().to(torch.float32)
    if scores.shape != reference_batch.target_mask.shape:
        raise ValueError(
            "TRACE reference log-probability shape does not match target mask: "
            f"{tuple(scores.shape)} != {tuple(reference_batch.target_mask.shape)}"
        )

    target_mask = reference_batch.target_mask
    token_counts = target_mask.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("TRACE reference targets must contain at least one token")
    avg_scores = torch.where(target_mask, scores, 0.0).sum(dim=1) / token_counts
    if not torch.isfinite(avg_scores).all():
        raise ValueError("TRACE reference produced non-finite prefix scores")

    for score, (sample_index, _state_index) in zip(
        avg_scores.tolist(),
        reference_batch.sample_state_indices,
        strict=True,
    ):
        result[sample_index].append(float(score))
    return result
