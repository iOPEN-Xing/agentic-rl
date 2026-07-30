"""Bounded multi-source reward fusion and consistency diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.envs.user_simulator_judge import JudgeFeedback


@dataclass(frozen=True)
class RewardFusionWeights:
    environment: float = 0.6
    judge: float = 0.3
    prm: float = 0.1

    def __post_init__(self) -> None:
        values = (self.environment, self.judge, self.prm)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("reward fusion weights must be finite and non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one reward fusion weight must be positive")


class HybridRewardCalculator:
    def __init__(self, weights: RewardFusionWeights = RewardFusionWeights()) -> None:
        self.weights = weights

    @staticmethod
    def _unit_interval(value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("reward source must be finite")
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _normalized_prm(cls, prm_score: float) -> float:
        return cls._unit_interval(float(prm_score) + 0.5)

    def _fuse(
        self,
        *,
        environment_score: float,
        judge_score: Optional[float],
        prm_score: float,
    ) -> tuple[float, dict[str, float | bool]]:
        sources = [
            (self.weights.environment, self._unit_interval(environment_score)),
            (self.weights.prm, self._normalized_prm(prm_score)),
        ]
        judge_available = judge_score is not None
        if judge_available:
            sources.append((self.weights.judge, self._unit_interval(judge_score)))

        active_weight = sum(weight for weight, _ in sources)
        fused = sum(weight * score for weight, score in sources) / active_weight
        return self._unit_interval(fused), {
            "environment_score": self._unit_interval(environment_score),
            "judge_score": self._unit_interval(judge_score) if judge_available else 0.0,
            "prm_score": float(prm_score),
            "judge_available": judge_available,
            "active_weight": active_weight,
        }

    def compute_turn_reward(
        self,
        *,
        environment_reward: float,
        judge_feedback: JudgeFeedback,
        prm_score: float,
    ) -> tuple[float, dict[str, float | bool]]:
        judge_score = judge_feedback.score if judge_feedback.valid else None
        return self._fuse(
            environment_score=environment_reward,
            judge_score=judge_score,
            prm_score=prm_score,
        )

    def compute_session_reward(
        self,
        *,
        outcome: float,
        judge_feedback: list[JudgeFeedback],
        prm_score: float,
        last_k: int = 5,
    ) -> tuple[float, dict[str, float | bool]]:
        valid_scores = [feedback.score for feedback in judge_feedback if feedback.valid]
        if last_k > 0:
            valid_scores = valid_scores[-last_k:]
        judge_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        return self._fuse(
            environment_score=outcome,
            judge_score=judge_score,
            prm_score=prm_score,
        )


def reward_consistency(
    environment_rewards: list[float],
    judge_rewards: list[float],
    *,
    minimum_samples: int = 3,
    warning_threshold: float = 0.3,
) -> dict[str, float | int | bool | None]:
    """Return a dependency-free Pearson correlation diagnostic."""
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
