"""Utilities for representing assistant generations as Turn-PPO actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def append_assistant_turn(spans: list[tuple[int, int]], start: int, end: int) -> None:
    """Append one non-empty, non-overlapping assistant response span."""
    start = int(start)
    end = int(end)
    if end <= start:
        raise ValueError(f"assistant turn span must have positive length, got {(start, end)}")
    if start < 0:
        raise ValueError(f"assistant turn span must start at or after zero, got {(start, end)}")
    if spans and start < spans[-1][1]:
        raise ValueError(f"assistant turn spans overlap: previous={spans[-1]}, current={(start, end)}")
    spans.append((start, end))


def clip_assistant_turn_spans(
    spans: Sequence[tuple[int, int]],
    response_length: int,
) -> list[tuple[int, int]]:
    """Clip ordered assistant spans to the response tensor boundary."""
    response_length = int(response_length)
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")

    clipped: list[tuple[int, int]] = []
    for start, end in spans:
        start = int(start)
        end = int(end)
        if end <= start:
            raise ValueError(f"assistant turn span must have positive length, got {(start, end)}")
        if start < 0:
            raise ValueError(f"assistant turn span must start at or after zero, got {(start, end)}")
        if clipped and start < clipped[-1][1]:
            raise ValueError(f"assistant turn spans overlap: previous={clipped[-1]}, current={(start, end)}")
        if start >= response_length:
            break
        clipped.append((start, min(end, response_length)))
    return clipped


def materialize_turn_ids(
    assistant_turn_spans: Sequence[Sequence[tuple[int, int]] | None],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Assign a one-based turn id to every trainable assistant token."""
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")
    if len(assistant_turn_spans) != response_mask.shape[0]:
        raise ValueError(
            "assistant turn span batch size does not match response_mask: "
            f"{len(assistant_turn_spans)} != {response_mask.shape[0]}"
        )

    turn_ids = torch.zeros_like(response_mask, dtype=torch.long)
    response_length = response_mask.shape[1]

    for batch_index, spans in enumerate(assistant_turn_spans):
        previous_end = 0
        next_turn_id = 1
        for raw_span in spans or []:
            if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 2:
                raise ValueError(f"invalid assistant turn span at batch {batch_index}: {raw_span!r}")
            start, end = int(raw_span[0]), int(raw_span[1])
            if start < 0 or end <= start:
                raise ValueError(f"invalid assistant turn span at batch {batch_index}: {(start, end)}")
            if start < previous_end:
                raise ValueError(
                    f"assistant turn spans overlap at batch {batch_index}: "
                    f"previous_end={previous_end}, current={(start, end)}"
                )
            if start >= response_length:
                break

            end = min(end, response_length)
            active_tokens = response_mask[batch_index, start:end].bool()
            turn_ids[batch_index, start:end] = active_tokens.to(torch.long) * next_turn_id
            previous_end = end
            next_turn_id += 1

    uncovered = response_mask.bool() & turn_ids.eq(0)
    if uncovered.any():
        locations = torch.nonzero(uncovered, as_tuple=False)[:8].tolist()
        raise ValueError(f"trainable response tokens are not covered by assistant turn spans: {locations}")

    return turn_ids
