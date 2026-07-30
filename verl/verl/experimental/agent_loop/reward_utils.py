"""Utilities for aligning scalar interaction rewards with generated response tokens."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


def materialize_turn_rewards(
    turn_reward_spans: Sequence[Sequence[dict[str, Any]] | None],
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place each turn reward on the last trainable token in its response span."""
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")
    if len(turn_reward_spans) != response_mask.shape[0]:
        raise ValueError(
            "turn_reward_spans batch size does not match response_mask: "
            f"{len(turn_reward_spans)} != {response_mask.shape[0]}"
        )

    rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    event_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    response_length = response_mask.shape[1]

    for batch_index, spans in enumerate(turn_reward_spans):
        for span in spans or []:
            start = max(0, int(span["start"]))
            end = min(response_length, int(span["end"]))
            reward = float(span["reward"])
            if not math.isfinite(reward):
                raise ValueError(f"turn reward must be finite, got {reward}")
            if end <= start:
                continue

            active_offsets = torch.nonzero(
                response_mask[batch_index, start:end] > 0,
                as_tuple=False,
            ).flatten()
            if active_offsets.numel() == 0:
                continue
            token_index = start + int(active_offsets[-1].item())
            rewards[batch_index, token_index] += reward
            event_mask[batch_index, token_index] = True

    return rewards, event_mask
