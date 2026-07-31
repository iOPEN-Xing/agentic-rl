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
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig


def compute_hybrid_alpha(
    global_step: int,
    total_steps: int,
    session_weight_start: float = 0.8,
    session_weight_end: float = 0.2,
    schedule: str = "linear",
) -> float:
    """Return the scheduled session-level weight for hybrid advantage fusion."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= session_weight_start <= 1.0:
        raise ValueError(f"session_weight_start must be in [0, 1], got {session_weight_start}")
    if not 0.0 <= session_weight_end <= 1.0:
        raise ValueError(f"session_weight_end must be in [0, 1], got {session_weight_end}")

    progress = min(max(float(global_step) / float(total_steps), 0.0), 1.0)
    if schedule == "linear":
        fraction = progress
    elif schedule == "cosine":
        fraction = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported hybrid advantage schedule: {schedule!r}")
    return session_weight_start + (session_weight_end - session_weight_start) * fraction


def normalize_advantage_component(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Normalize one advantage component on the tokens where it is defined."""
    mask_bool = mask.bool()
    normalized = torch.zeros_like(advantages)
    valid = advantages[mask_bool]
    if valid.numel() == 0:
        return normalized

    centered = valid - valid.mean()
    std = centered.square().mean().sqrt()
    if not torch.isfinite(std) or std <= epsilon:
        normalized[mask_bool] = valid
    else:
        normalized[mask_bool] = centered / (std + epsilon)
    return normalized


def compute_turn_level_mc_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,
    assistant_turn_spans: np.ndarray | list,
    assistant_turn_rewards: np.ndarray | list,
    gamma: float = 0.9,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute group-relative turn reward-to-go and map it to assistant tokens.

    The returned mask identifies turns with at least two comparable rollouts for
    the same (uid, turn index). This is Monte Carlo credit assignment over
    observed turn rewards, not GAE: no value baseline is invented.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"turn gamma must be in [0, 1], got {gamma}")

    batch_size, response_length = response_mask.shape
    if len(index) != batch_size:
        raise ValueError(f"uid count must match batch size: {len(index)} != {batch_size}")
    if len(assistant_turn_spans) != batch_size or len(assistant_turn_rewards) != batch_size:
        raise ValueError("Turn event batch dimensions must match response_mask")

    grouped_turns: dict[tuple[Any, int], list[tuple[int, int, int, float]]] = defaultdict(list)
    with torch.no_grad():
        for batch_index in range(batch_size):
            # Infrastructure-failed trajectories have a zero response mask and
            # must not affect either returns-to-go or comparison statistics.
            if not response_mask[batch_index].bool().any():
                continue
            sample_spans = assistant_turn_spans[batch_index]
            sample_rewards = assistant_turn_rewards[batch_index]
            sample_spans = [] if sample_spans is None else list(sample_spans)
            sample_rewards = [] if sample_rewards is None else list(sample_rewards)
            if len(sample_spans) != len(sample_rewards):
                raise ValueError(
                    f"Turn span/reward mismatch for sample {batch_index}: "
                    f"{len(sample_spans)} != {len(sample_rewards)}"
                )

            parsed_spans: list[tuple[int, int]] = []
            parsed_rewards: list[float] = []
            previous_end = 0
            for turn_index, (span, reward) in enumerate(zip(sample_spans, sample_rewards, strict=True)):
                if not isinstance(span, (tuple, list, np.ndarray)) or len(span) != 2:
                    raise ValueError(f"Invalid turn span for sample {batch_index}, turn {turn_index}: {span!r}")
                start, end = int(span[0]), int(span[1])
                if start < previous_end or start < 0 or end <= start or end > response_length:
                    raise ValueError(
                        f"Invalid or overlapping turn span for sample {batch_index}, "
                        f"turn {turn_index}: {(start, end)}"
                    )
                value = float(reward)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite turn reward for sample {batch_index}, turn {turn_index}: {reward!r}"
                    )
                parsed_spans.append((start, end))
                parsed_rewards.append(value)
                previous_end = end

            reward_to_go = [0.0] * len(parsed_rewards)
            running_return = 0.0
            for turn_index in range(len(parsed_rewards) - 1, -1, -1):
                running_return = parsed_rewards[turn_index] + gamma * running_return
                reward_to_go[turn_index] = running_return

            uid = index[batch_index]
            for turn_index, ((start, end), turn_return) in enumerate(zip(parsed_spans, reward_to_go, strict=True)):
                if response_mask[batch_index, start:end].bool().any():
                    grouped_turns[(uid, turn_index)].append((batch_index, start, end, turn_return))

        turn_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        turn_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        for records in grouped_turns.values():
            if len(records) < 2:
                continue
            values = torch.tensor(
                [record[3] for record in records],
                device=response_mask.device,
                dtype=torch.float32,
            )
            centered = values - values.mean()
            std = centered.square().mean().sqrt()
            if not torch.isfinite(std) or std <= epsilon:
                continue
            normalized_values = centered / (std + epsilon)
            for normalized_value, (batch_index, start, end, _) in zip(
                normalized_values, records, strict=True
            ):
                span_mask = response_mask[batch_index, start:end].to(torch.float32)
                turn_advantages[batch_index, start:end] = normalized_value * span_mask
                turn_mask[batch_index, start:end] = span_mask

    return turn_advantages, turn_mask


def _hybrid_config_value(config: Optional[AlgoConfig], key: str, default: Any) -> Any:
    if config is None:
        return default
    hybrid_config = getattr(config, "hybrid_advantage", None)
    if hybrid_config is None:
        return default
    if hasattr(hybrid_config, "get"):
        return hybrid_config.get(key, default)
    return getattr(hybrid_config, key, default)


def _hybrid_config_has_value(config: Optional[AlgoConfig], key: str) -> bool:
    if config is None:
        return False
    hybrid_config = getattr(config, "hybrid_advantage", None)
    if hybrid_config is None:
        return False
    if hasattr(hybrid_config, "get"):
        return key in hybrid_config and hybrid_config.get(key) is not None
    return getattr(hybrid_config, key, None) is not None


def _normalize_session_component_by_group(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float,
) -> torch.Tensor:
    """Normalize one scalar per rollout, without weighting longer responses more."""
    normalized = torch.zeros_like(advantages, dtype=torch.float32)
    grouped: dict[Any, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    with torch.no_grad():
        for batch_index in range(response_mask.shape[0]):
            active = response_mask[batch_index].bool()
            if active.any():
                scalar = advantages[batch_index][active][0].to(torch.float32)
                grouped[index[batch_index]].append((batch_index, scalar))

        for records in grouped.values():
            values = torch.stack([value for _, value in records])
            centered = values - values.mean()
            std = centered.square().mean().sqrt()
            normalized_values = values if not torch.isfinite(std) or std <= epsilon else centered / (std + epsilon)
            for normalized_value, (batch_index, _) in zip(normalized_values, records, strict=True):
                normalized[batch_index] = normalized_value * response_mask[batch_index].to(torch.float32)
    return normalized


def resolve_hybrid_session_weight(
    config: Optional[AlgoConfig],
    global_step: int,
    total_steps: Optional[int],
) -> float:
    """Resolve fixed turn residual or scheduled session weight, never both."""
    if _hybrid_config_has_value(config, "turn_weight"):
        schedule_keys = ("session_weight_start", "session_weight_end", "schedule")
        conflicts = [key for key in schedule_keys if _hybrid_config_has_value(config, key)]
        if conflicts:
            raise ValueError(
                "hybrid_advantage.turn_weight is mutually exclusive with scheduled fields: "
                f"{conflicts}"
            )
        turn_weight = float(_hybrid_config_value(config, "turn_weight", 0.1))
        if not 0.0 <= turn_weight <= 1.0:
            raise ValueError(f"turn_weight must be in [0, 1], got {turn_weight}")
        return 1.0 - turn_weight

    configured_total_steps = int(_hybrid_config_value(config, "total_steps", 1))
    effective_total_steps = configured_total_steps if total_steps is None else int(total_steps)
    return compute_hybrid_alpha(
        global_step=global_step,
        total_steps=effective_total_steps,
        session_weight_start=float(_hybrid_config_value(config, "session_weight_start", 0.8)),
        session_weight_end=float(_hybrid_config_value(config, "session_weight_end", 0.2)),
        schedule=str(_hybrid_config_value(config, "schedule", "linear")),
    )


def compute_grpo_hybrid_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    assistant_turn_spans: np.ndarray | list,
    assistant_turn_rewards: np.ndarray | list,
    global_step: int = 0,
    total_steps: Optional[int] = None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse normalized session GRPO and turn-level Monte Carlo advantages."""
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    norm_session_by_std = True
    if config is not None:
        if hasattr(config, "get"):
            norm_session_by_std = bool(config.get("norm_adv_by_std_in_grpo", True))
        else:
            norm_session_by_std = bool(getattr(config, "norm_adv_by_std_in_grpo", True))
    if not norm_session_by_std:
        raise ValueError(
            "grpo_hybrid requires norm_adv_by_std_in_grpo=true so session and turn components use comparable scales"
        )

    session_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_session_by_std,
        config=config,
    )

    turn_gamma = float(_hybrid_config_value(config, "turn_gamma", 0.9))
    turn_advantages, turn_mask = compute_turn_level_mc_advantage(
        response_mask=response_mask,
        index=index,
        assistant_turn_spans=assistant_turn_spans,
        assistant_turn_rewards=assistant_turn_rewards,
        gamma=turn_gamma,
        epsilon=epsilon,
    )

    # Both components are normalized over rollout comparison units, never over
    # repeated token values. Turn MC already normalizes each (uid, turn) group.
    session_advantages = _normalize_session_component_by_group(
        session_advantages, response_mask, index, epsilon
    )

    if torch.any(turn_mask.bool()):
        alpha = resolve_hybrid_session_weight(config, global_step, total_steps)
        fused_turn_tokens = alpha * session_advantages + (1.0 - alpha) * turn_advantages
        final_advantages = torch.where(turn_mask.bool(), fused_turn_tokens, session_advantages)
    else:
        # Missing or non-comparable turns fall back exactly to session GRPO.
        final_advantages = session_advantages

    final_advantages = final_advantages * response_mask
    return final_advantages, final_advantages.clone()
