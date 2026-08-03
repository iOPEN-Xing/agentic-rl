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
def test_vanilla_eval_contract_is_fixed_across_checkpoints(step):
    path = PROJECT_ROOT / f"configs/eval/vanilla_grpo/eval_vanilla_step{step}.yaml"
    config = yaml.safe_load(path.read_text())

    assert config["env"]["name"] == "airline"
    assert config["env"]["user_model"] == "/data/xjz/model/qwen3-14b"
    assert config["policy"] == {
        "model_name": "agentic-rl-policy",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    assert config["eval"] == {
        "num_tasks": 50,
        "num_samples_per_task": 4,
        "max_turns": 30,
        "num_workers": 2,
    }
    assert config["output"]["dir"] == f"experiments/vanilla/eval_step_{step}"
