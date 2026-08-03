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

"""TRACE-style hybrid outcome/turn advantage.

The implementation follows Algorithm 1 of:
TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon
Agents, arXiv:2607.13988v1.

`trace_prefix_avg_log_probs` must be produced by a frozen reference model with
teacher forcing over the same training-only gold target at every state prefix.
The code here only constructs credit and maps it to policy-generated tokens; it
does not treat environment step rewards or a learned critic as TRACE values.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig


def _hybrid_config_value(config: Optional[AlgoConfig], key: str, default: Any) -> Any:
    if config is None:
        return default
    hybrid_config = getattr(config, "hybrid_advantage", None)
    if hybrid_config is None:
        return default
    if hasattr(hybrid_config, "get"):
        return hybrid_config.get(key, default)
    return getattr(hybrid_config, key, default)


def compute_trace_log_ratio_values(
    avg_gold_log_probs: torch.Tensor,
    gap_epsilon: float = 0.1,
) -> torch.Tensor:
    """Map average gold-target log-probabilities to TRACE state values.

    For prefix ``k``:

        d_k = -mean_log_p_k + epsilon
        V_k = log(d_0 / d_k)

    ``V_0`` is therefore exactly zero. The offset is part of the remaining-gap
    definition, not a numerical epsilon added after taking the logarithm.
    """
    if avg_gold_log_probs.ndim != 1:
        raise ValueError(
            "TRACE prefix log-probabilities must be one-dimensional, "
            f"got shape {tuple(avg_gold_log_probs.shape)}"
        )
    if avg_gold_log_probs.numel() == 0:
        return avg_gold_log_probs.to(dtype=torch.float32)
    if not math.isfinite(gap_epsilon) or gap_epsilon <= 0:
        raise ValueError(f"gap_epsilon must be finite and positive, got {gap_epsilon}")

    scores = avg_gold_log_probs.to(dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("TRACE prefix log-probabilities must all be finite")
    if torch.any(scores > 1e-6):
        raise ValueError("Average log-probabilities must be non-positive")

    remaining_gap = -scores + float(gap_epsilon)
    if torch.any(remaining_gap <= 0):
        raise ValueError("TRACE remaining gaps must be positive")
    return torch.log(remaining_gap[0] / remaining_gap)


def compute_trace_turn_rewards(
    state_values: torch.Tensor,
    outcome_advantage: float | torch.Tensor,
    td_horizon: int = 3,
    td_gamma: float = 0.8,
    terminal_scale: float = 2.0,
) -> torch.Tensor:
    """Construct normalized K-step TD credit plus terminal-outcome fill.

    There are ``T + 1`` state values for ``T`` decision/observation
    transitions. ``td_horizon=0`` is the paper's ablation shorthand for
    disabling dense TD credit; only the final transition receives terminal
    fill in that case.
    """
    if state_values.ndim != 1:
        raise ValueError(f"TRACE state values must be one-dimensional, got {state_values.shape}")
    if td_horizon < 0:
        raise ValueError(f"td_horizon must be non-negative, got {td_horizon}")
    if not 0.0 <= td_gamma <= 1.0:
        raise ValueError(f"td_gamma must be in [0, 1], got {td_gamma}")
    if not math.isfinite(terminal_scale) or terminal_scale < 0:
        raise ValueError(f"terminal_scale must be finite and non-negative, got {terminal_scale}")
    if state_values.numel() < 2:
        return torch.zeros(0, device=state_values.device, dtype=torch.float32)
    if not torch.isfinite(state_values).all():
        raise ValueError("TRACE state values must all be finite")

    values = state_values.to(dtype=torch.float32)
    deltas = values[1:] - values[:-1]
    transition_count = int(deltas.numel())
    outcome = torch.as_tensor(
        outcome_advantage,
        device=values.device,
        dtype=torch.float32,
    )
    if outcome.numel() != 1 or not torch.isfinite(outcome):
        raise ValueError("outcome_advantage must be one finite scalar")

    rewards = torch.zeros_like(deltas)
    for turn_index in range(transition_count):
        if td_horizon == 0:
            horizon_end = turn_index
            local_credit = torch.zeros((), device=values.device, dtype=torch.float32)
        else:
            horizon_end = min(turn_index + td_horizon - 1, transition_count - 1)
            offsets = torch.arange(
                horizon_end - turn_index + 1,
                device=values.device,
                dtype=torch.float32,
            )
            discounts = float(td_gamma) ** offsets
            local_credit = (
                discounts * deltas[turn_index : horizon_end + 1]
            ).sum() / discounts.sum()

        terminal_fill = torch.zeros((), device=values.device, dtype=torch.float32)
        if horizon_end == transition_count - 1:
            terminal_fill = (
                float(terminal_scale)
                * float(td_gamma) ** (transition_count - turn_index)
                * outcome
            )
        rewards[turn_index] = local_credit + terminal_fill
    return rewards


def _trajectory_outcome_advantage(
    session_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    batch_index: int,
) -> torch.Tensor:
    valid_mask = response_mask[batch_index].bool()
    if not torch.any(valid_mask):
        return torch.zeros((), device=session_advantages.device, dtype=torch.float32)
    # GRPO outcome advantage is broadcast across all response tokens. Taking
    # the mean makes the assumption explicit and avoids depending on which
    # assistant span appears first.
    return session_advantages[batch_index][valid_mask].to(torch.float32).mean()


def compute_trace_group_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    *,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute Algorithm 1's per-prompt outcome advantage.

    TRACE defines ``sigma_R`` with a population variance (division by ``G``),
    whereas the upstream veRL GRPO helper currently calls ``torch.std`` with
    Bessel correction. Keeping this local helper avoids silently changing the
    paper's outcome scale.
    """
    scores = token_level_rewards.sum(dim=-1).to(torch.float32)
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for batch_index, uid in enumerate(index):
        # Infrastructure-failed rollouts are represented by an all-zero
        # response mask. They must not change a valid group's outcome mean or
        # variance, even if stale/non-zero rewards are present in the batch.
        if response_mask[batch_index].bool().any():
            grouped_indices[uid].append(batch_index)

    normalized_scores = torch.zeros_like(scores)
    with torch.no_grad():
        for sample_indices in grouped_indices.values():
            group_scores = scores[sample_indices]
            centered = group_scores - group_scores.mean()
            if normalize_by_std:
                std = centered.square().mean().sqrt()
                if torch.isfinite(std) and std > epsilon:
                    centered = centered / std
                else:
                    centered = torch.zeros_like(centered)
            normalized_scores[sample_indices] = centered
    return normalized_scores.unsqueeze(-1) * response_mask.to(torch.float32)


def compute_grpo_hybrid_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trace_turn_spans: np.ndarray | list,
    trace_prefix_avg_log_probs: np.ndarray | list,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse GRPO outcome advantage with trajectory-local TRACE turn credit.

    Unlike the previous heuristic implementation, this function deliberately:

    * does not consume τ-bench incremental environment rewards;
    * does not normalize TRACE values across rollout groups;
    * does not schedule outcome weight toward zero during training; and
    * leaves the final-answer tail on outcome advantage only.
    """
    batch_size, response_length = response_mask.shape
    if len(index) != batch_size:
        raise ValueError(f"uid count must match batch size: {len(index)} != {batch_size}")
    if len(trace_turn_spans) != batch_size or len(trace_prefix_avg_log_probs) != batch_size:
        raise ValueError("TRACE metadata batch dimensions must match response_mask")

    norm_session_by_std = True
    if config is not None:
        if hasattr(config, "get"):
            norm_session_by_std = bool(config.get("norm_adv_by_std_in_grpo", True))
        else:
            norm_session_by_std = bool(getattr(config, "norm_adv_by_std_in_grpo", True))

    session_advantages = compute_trace_group_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        normalize_by_std=norm_session_by_std,
    )

    outcome_weight = float(_hybrid_config_value(config, "outcome_weight", 1.0))
    turn_weight = float(_hybrid_config_value(config, "turn_weight", 0.2))
    gap_epsilon = float(_hybrid_config_value(config, "gap_epsilon", 0.1))
    td_horizon = int(_hybrid_config_value(config, "td_horizon", 3))
    td_gamma = float(_hybrid_config_value(config, "td_gamma", 0.8))
    terminal_scale = float(_hybrid_config_value(config, "terminal_scale", 2.0))
    if not math.isfinite(outcome_weight) or outcome_weight < 0:
        raise ValueError(f"outcome_weight must be finite and non-negative, got {outcome_weight}")
    if not math.isfinite(turn_weight) or turn_weight < 0:
        raise ValueError(f"turn_weight must be finite and non-negative, got {turn_weight}")

    final_advantages = outcome_weight * session_advantages.to(torch.float32)
    with torch.no_grad():
        for batch_index in range(batch_size):
            # The rollout post-processor zeros response_mask for infrastructure
            # failures. Skip their metadata entirely so malformed partial TRACE
            # events cannot turn an already-excluded row into a trainer error.
            if not response_mask[batch_index].bool().any():
                continue
            spans = trace_turn_spans[batch_index]
            scores = trace_prefix_avg_log_probs[batch_index]
            spans = [] if spans is None else list(spans)
            scores = [] if scores is None else list(scores)

            if not spans:
                if len(scores) not in (0, 1):
                    raise ValueError(
                        f"Sample {batch_index} has no TRACE turns but {len(scores)} prefix scores"
                    )
                continue
            if len(scores) != len(spans) + 1:
                raise ValueError(
                    "TRACE requires one more prefix score than transition span for "
                    f"sample {batch_index}: {len(scores)} != {len(spans)} + 1"
                )

            previous_end = 0
            parsed_spans: list[tuple[int, int]] = []
            for turn_index, span in enumerate(spans):
                if not isinstance(span, (tuple, list, np.ndarray)) or len(span) != 2:
                    raise ValueError(
                        f"Invalid TRACE span for sample {batch_index}, turn {turn_index}: {span!r}"
                    )
                start, end = int(span[0]), int(span[1])
                if start < previous_end or start < 0 or end <= start or end > response_length:
                    raise ValueError(
                        f"Invalid or overlapping TRACE span for sample {batch_index}, "
                        f"turn {turn_index}: {(start, end)}"
                    )
                parsed_spans.append((start, end))
                previous_end = end

            score_tensor = torch.as_tensor(
                scores,
                device=response_mask.device,
                dtype=torch.float32,
            )
            state_values = compute_trace_log_ratio_values(
                score_tensor,
                gap_epsilon=gap_epsilon,
            )
            turn_rewards = compute_trace_turn_rewards(
                state_values=state_values,
                outcome_advantage=_trajectory_outcome_advantage(
                    session_advantages,
                    response_mask,
                    batch_index,
                ),
                td_horizon=td_horizon,
                td_gamma=td_gamma,
                terminal_scale=terminal_scale,
            )
            for turn_reward, (start, end) in zip(turn_rewards, parsed_spans, strict=True):
                span_mask = response_mask[batch_index, start:end].to(torch.float32)
                final_advantages[batch_index, start:end] += turn_weight * turn_reward * span_mask

    final_advantages = final_advantages * response_mask.to(torch.float32)
    return final_advantages, final_advantages.clone()
