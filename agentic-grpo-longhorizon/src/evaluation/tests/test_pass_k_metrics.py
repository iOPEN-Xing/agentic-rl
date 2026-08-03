from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.evaluation.pass_k_eval import estimate_pass_at_k, estimate_pass_power_k


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_pass_at_k_and_pass_power_k_are_distinct_for_k_greater_than_one():
    # Two successes among four samples: any two contain a success with
    # probability 5/6, while both are successful with probability 1/6.
    assert estimate_pass_at_k(4, 2, 1) == pytest.approx(0.5)
    assert estimate_pass_power_k(4, 2, 1) == pytest.approx(0.5)
    assert estimate_pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert estimate_pass_power_k(4, 2, 2) == pytest.approx(1 / 6)


def test_pass_estimators_handle_all_failure_and_all_success():
    assert estimate_pass_at_k(8, 0, 4) == pytest.approx(0.0)
    assert estimate_pass_power_k(8, 0, 4) == pytest.approx(0.0)
    assert estimate_pass_at_k(8, 8, 4) == pytest.approx(1.0)
    assert estimate_pass_power_k(8, 8, 4) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "n,c,k",
    [(0, 0, 1), (4, -1, 1), (4, 5, 1), (4, 2, 0), (4, 2, 5)],
)
def test_pass_estimators_reject_invalid_counts(n, c, k):
    with pytest.raises(ValueError):
        estimate_pass_at_k(n, c, k)
    with pytest.raises(ValueError):
        estimate_pass_power_k(n, c, k)


@pytest.mark.parametrize("step", [50, 100, 150, 200])
def test_judge_eval_config_matches_vanilla_sampling_contract(step):
    vanilla_path = PROJECT_ROOT / f"configs/eval/vanilla_grpo/eval_vanilla_step{step}.yaml"
    judge_path = PROJECT_ROOT / f"configs/eval/user_simulator_judge/eval_user_simulator_judge_step{step}.yaml"
    vanilla = yaml.safe_load(vanilla_path.read_text())
    judge = yaml.safe_load(judge_path.read_text())

    assert judge["env"] == vanilla["env"]
    assert judge["policy"] == vanilla["policy"]
    assert judge["policy"]["model_name"] == "agentic-rl-policy"
    assert judge["eval"] == vanilla["eval"]
    assert judge["output"]["dir"] == f"experiments/user_simulator_judge/eval_step_{step}"
