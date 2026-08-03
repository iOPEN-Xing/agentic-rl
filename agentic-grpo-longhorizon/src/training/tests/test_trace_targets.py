import json

from scripts.train.grpo.build_grpo_parquet import (
    build_rows,
    serialize_tau_bench_trace_target,
)
from tau_bench.envs.airline.tasks_test import TASKS


def test_task_zero_trace_target_is_deterministic_and_training_only():
    target = serialize_tau_bench_trace_target(TASKS[0])
    parsed = json.loads(target)

    assert parsed["version"] == "tau_bench_airline_actions_outputs_v1"
    assert parsed["required_actions"][0]["name"] == "book_reservation"
    assert parsed["required_actions"][0]["kwargs"]["flights"][0]["flight_number"] == "HAT136"
    assert parsed["required_outputs"] == []
    assert "reactive" not in target.lower()

    row = build_rows([0], split="seen")[0]
    assert row["reward_model"]["ground_truth"] == target
    assert target not in json.dumps(row["prompt"])


def test_trace_target_preserves_required_outputs_and_action_order():
    target = json.loads(serialize_tau_bench_trace_target(TASKS[2]))

    assert len(target["required_actions"]) == 5
    assert target["required_outputs"] == ["23553"]
