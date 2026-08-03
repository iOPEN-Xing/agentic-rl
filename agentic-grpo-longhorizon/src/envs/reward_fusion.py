"""Judge signal calibration and cross-trajectory consistency diagnostics.

The filename is kept for configuration/import compatibility with the original
experiment branch.  The training semantics deliberately avoid fusing the
rule-based environment outcome, PRM-Lite, and LLM judge into one session score:

* the environment remains the sole source of the terminal task outcome;
* a valid judge response may provide a bounded, centered auxiliary signal;
* an unavailable judge produces no event instead of changing reward weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.envs.user_simulator_judge import JudgeFeedback


@dataclass(frozen=True)
class JudgeSignalPolicy:
    """Map a valid judge score in ``[0, 1]`` to a signal in ``[-1, 1]``.

    ``neutral_score`` is mapped to zero. Scores below/above it are linearly
    scaled to -1/+1. This keeps the auxiliary contribution interpretable and
    prevents a merely available judge from creating an automatic positive
    reward. Invalid or unavailable feedback is represented by ``None`` so the
    token-alignment layer can leave its event mask unset.
    """

    neutral_score: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.neutral_score) or not 0.0 < self.neutral_score < 1.0:
            raise ValueError("judge neutral_score must be finite and strictly between 0 and 1")

    def compute(self, feedback: JudgeFeedback) -> Optional[float]:
        if not feedback.valid:
            return None

        score = float(feedback.score)
        if not math.isfinite(score):
            raise ValueError("valid judge score must be finite")
        score = max(0.0, min(1.0, score))
        if score >= self.neutral_score:
            signal = (score - self.neutral_score) / (1.0 - self.neutral_score)
        else:
            signal = (score - self.neutral_score) / self.neutral_score
        return max(-1.0, min(1.0, signal))


def reward_consistency(
    environment_rewards: list[float],
    judge_rewards: list[float],
    *,
    minimum_samples: int = 3,
    warning_threshold: float = 0.3,
) -> dict[str, float | int | bool | None]:
    """Return a dependency-free Pearson diagnostic across trajectories.

    The two inputs must be paired trajectory-level observations, for example
    terminal outcomes and mean valid judge scores from a rollout batch. A
    within-trajectory sequence of mostly-zero incremental rewards is not a
    meaningful input for this diagnostic.
    """

    if len(environment_rewards) != len(judge_rewards):
        raise ValueError("environment and judge reward lengths must match")
    sample_count = len(environment_rewards)
    if sample_count < minimum_samples:
        return {
            "sample_count": sample_count,
            "correlation": None,
            "conflict": False,
        }

    env_mean = sum(environment_rewards) / sample_count
    judge_mean = sum(judge_rewards) / sample_count
    covariance = sum(
        (env - env_mean) * (judge - judge_mean)
        for env, judge in zip(environment_rewards, judge_rewards, strict=True)
    )
    env_variance = sum((value - env_mean) ** 2 for value in environment_rewards)
    judge_variance = sum((value - judge_mean) ** 2 for value in judge_rewards)
    denominator = math.sqrt(env_variance * judge_variance)
    correlation = covariance / denominator if denominator > 0.0 else None
    return {
        "sample_count": sample_count,
        "correlation": correlation,
        "conflict": correlation is not None and correlation < warning_threshold,
    }
