# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math


def append_assistant_turn(
    spans: list[tuple[int, int]],
    rewards: list[float],
    start: int,
    end: int,
) -> None:
    """Append one assistant generation span and its initially empty event reward."""
    if start < 0 or end <= start:
        raise ValueError(f"Invalid assistant turn span: {(start, end)}")
    if spans and start < spans[-1][1]:
        raise ValueError(f"Assistant turn spans overlap: {spans[-1]} and {(start, end)}")
    spans.append((start, end))
    rewards.append(0.0)


def accumulate_latest_turn_reward(rewards: list[float], reward: float | None) -> bool:
    """Attach a tool or interaction reward to the assistant turn that caused it."""
    if reward is None:
        return False
    value = float(reward)
    if not math.isfinite(value):
        raise ValueError(f"Turn reward must be finite, got {reward!r}")
    if not rewards:
        raise ValueError("Received a turn reward before any assistant generation")
    rewards[-1] += value
    return True


def clip_turn_reward_events(
    spans: list[tuple[int, int]],
    rewards: list[float],
    response_length: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Clip aligned turn events to the response tokens returned by the rollout."""
    if len(spans) != len(rewards):
        raise ValueError(f"Turn span/reward length mismatch: {len(spans)} != {len(rewards)}")
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")

    clipped_spans: list[tuple[int, int]] = []
    clipped_rewards: list[float] = []
    for (start, end), reward in zip(spans, rewards, strict=True):
        if start >= response_length:
            break
        clipped_end = min(end, response_length)
        if clipped_end > start:
            clipped_spans.append((start, clipped_end))
            clipped_rewards.append(float(reward))
    return clipped_spans, clipped_rewards
