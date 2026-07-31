from __future__ import annotations

import pytest

from src.envs.tau_bench_interaction import (
    _compute_progress_potential,
    _compute_turn_reward,
    compute_potential_shaping,
)


def _action(tool: str, *, error: bool = False, params=None, entities=None):
    return {
        "tool": tool,
        "parameters": params or {},
        "param_str": "{}",
        "is_error": error,
        "extracted_entities": entities or {},
    }


def _state(actions=None):
    return {
        "total_reward": 0.0,
        "action_history": list(actions or []),
        "turn_reward_action_cursor": 0,
        "turn_reward_prev_potential": 0.0,
        "turn_rewards": [],
    }


def test_potential_shaping_uses_training_discount():
    assert compute_potential_shaping(0.2, 0.5, gamma=0.99) == pytest.approx(0.295)
    with pytest.raises(ValueError):
        compute_potential_shaping(0.0, 1.0, gamma=1.1)


def test_turn_reward_does_not_recount_previous_actions():
    state = _state([_action("get_user_details")])
    first_reward, first_metadata = _compute_turn_reward(state)
    second_reward, second_metadata = _compute_turn_reward(state)

    assert first_metadata["turn_new_actions"] == 1
    assert second_metadata["turn_new_actions"] == 0
    assert state["turn_reward_action_cursor"] == 1
    assert first_reward > second_reward


def test_progress_potential_rewards_verifiable_data_chain():
    actions = [
        _action(
            "get_reservation_details",
            params={"reservation_id": "ABC123"},
            entities={"reservation_id": ["ABC123"]},
        ),
        _action("cancel_reservation", params={"reservation_id": "ABC123"}),
    ]
    state = _state(actions)
    potential = _compute_progress_potential(state)
    reward, metadata = _compute_turn_reward(state)

    assert potential > 0.0
    assert metadata["turn_process_score"] > 0.0
    assert -1.0 <= reward <= 1.0


def test_progress_potential_does_not_leak_terminal_outcome():
    state = _state([_action("get_user_details")])
    failed_potential = _compute_progress_potential(state)

    state["total_reward"] = 1.0
    assert _compute_progress_potential(state) == pytest.approx(failed_potential)

    state["done"] = True
    assert _compute_progress_potential(state) == pytest.approx(0.0)


def test_failed_placeholder_action_is_penalized():
    state = _state([
        _action(
            "cancel_reservation",
            error=True,
            params={"reservation_id": "previous_reservation"},
        )
    ])
    reward, metadata = _compute_turn_reward(state)
    assert metadata["turn_process_score"] < 0.0
    assert reward < 0.0
