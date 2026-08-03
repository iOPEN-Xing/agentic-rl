# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for adapting interactive rollouts to TRACE state transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceLayout:
    """TRACE metadata for one rollout.

    ``state_boundaries`` are response-token offsets immediately before each
    assistant generation. Consecutive boundaries therefore contain one policy
    decision and its following tool/user observation. The last assistant span
    is treated as the final-answer tail and has no local TRACE transition.
    """

    state_boundaries: list[int]
    turn_spans: list[tuple[int, int]]
    final_answer_span: tuple[int, int] | None


def build_trace_layout(
    assistant_turn_spans: list[tuple[int, int]],
    response_length: int,
) -> TraceLayout:
    """Validate and clip assistant spans, then form TRACE transitions."""
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")

    clipped: list[tuple[int, int]] = []
    previous_end = 0
    for turn_index, span in enumerate(assistant_turn_spans):
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            raise ValueError(f"Invalid assistant span at turn {turn_index}: {span!r}")
        start, end = int(span[0]), int(span[1])
        if start < previous_end:
            raise ValueError(f"Assistant turn spans overlap: {clipped[-1]} and {(start, end)}")
        if start < 0 or end <= start:
            raise ValueError(f"Invalid assistant turn span: {(start, end)}")
        previous_end = end
        if start >= response_length:
            break
        clipped_end = min(end, response_length)
        if clipped_end > start:
            clipped.append((start, clipped_end))

    if not clipped:
        return TraceLayout(state_boundaries=[], turn_spans=[], final_answer_span=None)

    return TraceLayout(
        state_boundaries=[start for start, _ in clipped],
        turn_spans=clipped[:-1],
        final_answer_span=clipped[-1],
    )
