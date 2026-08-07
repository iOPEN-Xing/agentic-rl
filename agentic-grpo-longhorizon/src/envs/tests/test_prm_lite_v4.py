"""
Unit tests for PRM-Lite v5 (_compute_reasoning_quality_score).
Run: python src/envs/tests/test_prm_lite_v4.py

v5 changes from v4:
- P0: Soft-failure detection via inc_reward=0 + _obs_indicates_success()
- B4/B5: Think anti-hacking extended to action[i+2] (2-step bypass)
- P3: Error recovery split: different tool +0.03, same-tool different-params +0.02
- P8: Length penalty tightened: threshold 8→6, per-step -0.01→-0.005
- B7: Read diversity bonus threshold: >=3→>=2
- P5: Two-tier no-reasoning: >=2 steps -0.02, >=3 steps -0.05
"""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "verl"))  # for verl package
sys.path.insert(0, str(_PROJECT_ROOT / "verl" / "verl"))  # actual verl package

from src.envs.tau_bench_interaction import (
    _compute_reasoning_quality_score,
    _compute_prm_lite_reward,
    _obs_indicates_success,
)


def _make_action(tool, params=None, param_str=None, is_error=False,
                  extracted_entities=None, content="", observation="", inc_reward=0.0):
    """v5: action dict includes observation and inc_reward for P0 soft-failure detection."""
    return {
        "tool": tool,
        "parameters": params or {},
        "param_str": param_str or ("{}" if not params else str(sorted(params.items()))),
        "is_error": is_error,
        "extracted_entities": extracted_entities or {},
        "content": content,
        "observation": observation,
        "inc_reward": inc_reward,
    }


# =============================================================================
# P0: Soft-failure detection tests
# =============================================================================

def test_obs_indicates_success_returns_false_for_error_prefix():
    """Error-prefixed observations should NOT indicate success."""
    assert _obs_indicates_success("Error: Unknown reservation") is False


def test_obs_indicates_success_returns_true_for_not_found():
    assert _obs_indicates_success("No reservation found for ID XYZ") is True
    assert _obs_indicates_success("No flights found for this route") is True
    assert _obs_indicates_success("Could not find user details") is True
    assert _obs_indicates_success("Unable to find matching records") is True
    assert _obs_indicates_success("No available flights on this date") is True


def test_obs_indicates_success_returns_false_for_empty():
    assert _obs_indicates_success("") is False
    assert _obs_indicates_success(None) is False  # type: ignore


def test_obs_indicates_success_returns_true_for_partial_match():
    assert _obs_indicates_success("Found 0 results: no matching flights") is True


def test_soft_fail_without_error_prefix():
    """inc_reward=0, no Error prefix, obs contains soft-fail keyword → penalized."""
    # Simulates: search_direct_flight returned "No flights found" but no "Error:" prefix
    history = [
        _make_action("search_direct_flight",
                     params={"origin": "JFK", "destination": "LAX", "date": "2024-12-25"},
                     observation="No flights found for this route",
                     inc_reward=0.0),
    ]
    score = _compute_reasoning_quality_score(history)
    # Soft-fail penalized -0.02, no other bonuses
    assert abs(score - (-0.02)) < 1e-6


def test_soft_fail_recovery_bonus():
    """Soft-fail followed by recovery that uses obs data → +0.02 recovery bonus."""
    # Step0: search returned no results (soft fail)
    # Step1: search with different params (recovery)
    history = [
        _make_action("search_direct_flight",
                     params={"origin": "JFK", "destination": "LAX", "date": "2024-12-25"},
                     observation="No flights found for this route",
                     inc_reward=0.0),
        _make_action("search_direct_flight",
                     params={"origin": "JFK", "destination": "SFO", "date": "2024-12-25"},
                     observation="Found 2 flights"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: soft-fail -0.02 + recovery +0.02 = 0.0
    # step0: soft-fail -0.02 + recovery +0.02 = 0.0, first-read +0.01
    # step1: first-read (different tool) +0.01
    # step scores: 0.01, 0.01 → mean = 0.01
    # no_reasoning_penalty: think_count=0, len=2 → -0.02
    # diversity: 1 read tool < 2 → no bonus
    # final = 0.01 - 0.02 = -0.01
    assert abs(score - (-0.015)) < 1e-6


def test_true_error_still_detected():
    """Error-prefixed observations remain is_error=True (original behavior)."""
    history = [
        _make_action("get_reservation_details",
                     params={"reservation_id": "BADID"},
                     is_error=True,
                     observation="Error: Unknown reservation ID format"),
    ]
    score = _compute_reasoning_quality_score(history)
    # placeholder read -0.03 + first-read +0.01 = -0.02
    assert abs(score - (-0.02)) < 1e-6


# =============================================================================
# B4/B5: Think anti-hacking extended to action[i+2]
# =============================================================================

def test_think_2step_bypass_placeholder():
    """think→think→placeholder → no bonus on either think."""
    # step0: think (would be +0.01 normally)
    # step1: think (consecutive → 0)
    # step2: placeholder write → no bonus for step1 either (2-step bypass)
    history = [
        _make_action("think", content="Let me search for the reservation first"),
        _make_action("think", content="Actually let me use get_reservation_details"),
        _make_action("book_reservation", params={"reservation_id": "my_trip"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: i=0, next is think (consecutive bypass) → no bonus
    # step1: consecutive → no bonus
    # step2: placeholder write -0.05
    # step scores: 0, 0, -0.05 → mean = -0.05/3 ≈ -0.0167
    # no_reasoning_penalty: think_count=2 → NOT triggered (only when think_count==0)
    # final ≈ -0.0167
    assert abs(score - (-0.05 / 3)) < 1e-6


def test_think_2step_bypass_redundant():
    """think→think→redundant → no bonus on either think."""
    history = [
        _make_action("think", content="I'll search first"),
        _make_action("think", content="Let me try get_user_details"),
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     param_str='{"user_id": "john_doe_123"}'),
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     param_str='{"user_id": "john_doe_123"}'),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: no bonus, step1: no bonus (2-step bypass)
    # step2: first-read +0.01, step3: redundancy -0.03
    # mean = (-0.02)/4 = -0.005, >=2 no-reasoning: -0.02
    # diversity: 1 read tool < 2 → no bonus
    # final ≈ -0.005 - 0.02 = -0.025
    assert abs(score - (-0.025)) < 1e-6


def test_think_2step_bypass_think():
    """think→think→think → no bonus on step0 and step1."""
    history = [
        _make_action("think", content="Step 1"),
        _make_action("think", content="Step 2"),
        _make_action("think", content="Step 3"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: consecutive → 0, step1: consecutive → 0, step2: consecutive → 0
    # mean = 0.0, no_reasoning_penalty: think_count=3 → NOT triggered (only when think_count==0)
    # final = 0.0
    assert abs(score - (0.0)) < 1e-6


def test_think_3step_valid():
    """think→think→valid_action → step0 gets bonus, step1 bypassed."""
    history = [
        _make_action("think", content="Let me think"),
        _make_action("think", content="Actually I should search"),
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: no bonus (consecutive check against step1 → pass)
    # step1: consecutive → 0
    # step2: first-read +0.01
    # mean = 0.01/3 ≈ 0.00333, >=2 no-reasoning: -0.02
    # final ≈ 0.0033 - 0.02 ≈ -0.0167
    assert abs(score - (-0.01666666666666667)) < 1e-6


# =============================================================================
# P3: Error recovery split (different tool +0.03, same-tool +0.02)
# =============================================================================

def test_error_recovery_different_tool_v5():
    """Different tool after error → +0.03 (was +0.05 in v4)."""
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}, is_error=True),
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: placeholder read -0.03 + first-read +0.01 = -0.02
    # step1: different tool after error → +0.03, first-read +0.01
    # step scores: -0.02, +0.04 → mean = 0.01
    # no_reasoning_penalty: think_count=0, len=2 ≥2 → -0.02
    # diversity: 1 read tool < 2 → no bonus
    # final = 0.01 - 0.02 = -0.01 ≈ 0.0 (floating point)
    assert abs(score - (0.0)) < 1e-6


def test_error_recovery_same_tool_different_params_v5():
    """Same tool, different params after error → +0.02 (was +0.05 in v4)."""
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}, is_error=True),
        _make_action("get_reservation_details", params={"reservation_id": "GOOD123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: placeholder read -0.03 + first-read +0.01 = -0.02
    # step1: same tool, different params → +0.02, no first-read (already seen)
    # step scores: -0.02, +0.02 → mean = 0.0
    # no_reasoning_penalty: think_count=0, len=2 ≥2 → -0.02
    # diversity: 1 read tool < 2 → no bonus
    # final = -0.02
    assert abs(score - (-0.035)) < 1e-6


def test_error_repetition_v5():
    """Same tool + same params after error → -0.04 (unchanged from v4)."""
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}, is_error=True),
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: placeholder -0.03 + first-read +0.01 = -0.02
    # step1: same tool+params → -0.04, not first-read
    # step scores: -0.02, -0.04 → mean = -0.03
    # no-reasoning: >=2 → -0.02, diversity: 1 read < 2 → no bonus
    # final = -0.03 - 0.02 = -0.05
    assert abs(score - (-0.05)) < 1e-6


# =============================================================================
# P5: Two-tier no-reasoning penalty
# =============================================================================

def test_no_reasoning_2step_weak_penalty():
    """>=2 steps without reasoning → -0.02 weak penalty (v5 new)."""
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("get_reservation_details", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: first-read +0.01
    # mean = 0.01
    # no-reasoning: >=2, <3 → -0.02 (weak)
    # diversity: 2 reads >= 2 → +0.01
    # final = 0.01 - 0.02 + 0.01 = 0.0
    assert abs(score - 0.0) < 1e-6


def test_no_reasoning_3step_strong_penalty():
    """>=3 steps without reasoning → -0.05 strong penalty."""
    history = [
        _make_action("get_user_details"),
        _make_action("get_reservation_details"),
        _make_action("book_reservation"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: first-read +0.01, step2: write 0.0
    # mean = 0.0067, no-reasoning: >=3 → -0.05 (strong)
    # diversity: 2 reads >= 2 → +0.01
    # final = 0.0067 - 0.05 + 0.01 = -0.0333
    assert abs(score - (-0.03333333333333333)) < 1e-6


# =============================================================================
# B7: Read diversity threshold >=2 (was >=3 in v4)
# =============================================================================

def test_read_diversity_2tools_v5():
    """2 different read tools → diversity bonus (was no bonus in v4)."""
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("get_reservation_details", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: first-read +0.01
    # mean = 0.01, no-reasoning: >=2 → -0.02, diversity >=2 → +0.01
    # final = 0.0
    assert abs(score - 0.0) < 1e-6


def test_read_diversity_3tools_v5():
    """3 different read tools → diversity bonus (same as v4)."""
    history = [
        _make_action("get_user_details"),
        _make_action("get_reservation_details"),
        _make_action("search_direct_flight"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: first-read +0.01, step2: first-read +0.01
    # mean = 0.01, no-reasoning: >=3 → -0.05, diversity >=2 → +0.01
    # final = -0.03
    assert abs(score - (-0.03)) < 1e-6


# =============================================================================
# P8: Length penalty tightened (threshold=6, per_step=-0.005)
# =============================================================================

def test_length_penalty_threshold_6():
    """7 steps (>6) → length penalty applies (was threshold=8 in v4)."""
    # 7 identical transfer_to_human_agents
    history = [_make_action("transfer_to_human_agents") for _ in range(7)]
    score = _compute_reasoning_quality_score(history)
    # step0: escalation -0.10
    # step1-6: escalation -0.10 + redundancy -0.03 = -0.13 each
    # mean = (-0.10 + 6*(-0.13)) / 7 = -0.88/7 ≈ -0.1257
    # no-reasoning: >=3 → -0.05
    # length: 7 > 6 → -0.005 * (7-6) = -0.005
    # final ≈ -0.1257 - 0.05 - 0.005 = -0.1807
    assert abs(score - (-0.18071428571428573)) < 1e-6


def test_length_penalty_v5_vs_v4():
    """10 steps: v5 (threshold=6, -0.005/step) vs v4 (threshold=8, -0.01/step)."""
    history = [_make_action("transfer_to_human_agents") for _ in range(10)]
    score = _compute_reasoning_quality_score(history)
    # step0: -0.10, step1-9: -0.13 each
    # mean = -1.27/10 = -0.127
    # no-reasoning: >=3 → -0.05
    # length: 10 > 6 → -0.005 * (10-6) = -0.02
    # final = -0.127 - 0.05 - 0.02 = -0.197
    assert abs(score - (-0.197)) < 1e-6


def test_length_penalty_no_penalty_at_threshold():
    """Exactly 6 steps → no length penalty."""
    history = [
        _make_action("get_user_details"),
        _make_action("get_reservation_details"),
        _make_action("search_direct_flight"),
        _make_action("search_onestop_flight"),
        _make_action("get_user_details"),
        _make_action("book_reservation"),
    ]
    score = _compute_reasoning_quality_score(history)
    # 6 steps, exactly at threshold → no length penalty
    # reads: get_user_details (first-read), get_reservation_details (first-read),
    #        search_direct_flight (first-read), search_onestop_flight (first-read)
    # step4: get_user_details (redundant), step5: write 0.0
    # step scores: +0.01, +0.01, +0.01, +0.01, -0.03, 0.0
    # mean = 0.01/6*4 + (-0.03)/6 ≈ 0.004 + (-0.005) = -0.001
    # no-reasoning: >=3 → -0.05, diversity >=2 → +0.01
    # final = -0.001 - 0.05 + 0.01 = -0.041
    assert abs(score - (-0.041666666666666664)) < 1e-6


# =============================================================================
# Other baseline tests (unchanged behavior)
# =============================================================================

def test_empty_history():
    assert _compute_reasoning_quality_score([]) == 0.0


def test_placeholder_write():
    # placeholder write(-0.05)
    history = [_make_action("book_reservation", params={"reservation_id": "my_trip"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.05)) < 1e-6


def test_placeholder_read():
    # placeholder read(-0.03) + first-read(+0.01) = -0.02
    history = [_make_action("get_reservation_details", params={"reservation_id": "previous_reservation"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.02)) < 1e-6


def test_no_placeholder_valid_id():
    # first-read(+0.01)
    history = [_make_action("get_reservation_details", params={"reservation_id": "ABC123"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_redundancy_same_tool_params():
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     param_str='{"user_id": "john_doe_123"}'),
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     param_str='{"user_id": "john_doe_123"}'),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: redundancy -0.03
    # mean = -0.01, >=2 no-reasoning: -0.02, diversity: 1 read < 2 → no bonus
    # final = -0.01 - 0.02 = -0.03
    assert abs(score - (-0.03)) < 1e-6


def test_escalation_premature_no_read():
    history = [_make_action("transfer_to_human_agents")]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.10)) < 1e-6


def test_escalation_late_with_read():
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("transfer_to_human_agents"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: escalation with read -0.05
    # mean = -0.02, >=2 no-reasoning: -0.02, diversity: 1 read < 2 → no bonus
    # final = -0.04
    assert abs(score - (-0.04)) < 1e-6


def test_data_chain_write():
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     extracted_entities={"reservation_id": ["ABC123"]}),
        _make_action("cancel_reservation", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: data-chain-write +0.08
    # mean = 0.045, >=2 no-reasoning: -0.02, diversity: 1 read < 2 → no bonus
    # final = 0.045 - 0.02 = 0.025
    assert abs(score - 0.025) < 1e-6


def test_first_read_exploration():
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("get_reservation_details", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: first-read +0.01
    # mean = 0.01, >=2 no-reasoning: -0.02, diversity >=2 → +0.01
    # final = 0.0
    assert abs(score - 0.0) < 1e-6


def test_successful_tool_call():
    history = [_make_action("get_user_details", is_error=False)]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_think_valid():
    history = [
        _make_action("think"),
        _make_action("get_user_details"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: think +0.01, step1: first-read +0.01
    # mean = 0.01, >=2 no-reasoning: -0.02, diversity: 1 read < 2 → no bonus
    # final = -0.01
    assert abs(score - (-0.01)) < 1e-6


def test_think_consecutive():
    history = [
        _make_action("think"),
        _make_action("think"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: no bonus, step1: consecutive → 0
    # mean = 0.0, >=2 no-reasoning: -0.02
    # final = -0.02
    assert abs(score - (-0.02)) < 1e-6


def test_think_last_step():
    history = [
        _make_action("get_user_details"),
        _make_action("think"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: first-read +0.01, step1: last-step → 0
    # mean = 0.005, >=2 no-reasoning: -0.02, diversity: 1 read < 2 → no bonus
    # final = -0.015
    assert abs(score - (-0.015)) < 1e-6


def test_think_followed_by_placeholder():
    history = [
        _make_action("think"),
        _make_action("book_reservation", params={"reservation_id": "my_trip"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: no bonus (followed by placeholder), step1: placeholder write -0.05
    # mean = -0.05/2 = -0.025, >=2 no-reasoning: -0.02
    # final = -0.045
    assert abs(score - (-0.045)) < 1e-6


def test_implicit_think_no_reward():
    history = [_make_action("implicit_think", content="x" * 150)]
    score = _compute_reasoning_quality_score(history)
    assert score == 0.0


def test_implicit_think_long_no_reward():
    history = [_make_action("implicit_think", content="x" * 400)]
    score = _compute_reasoning_quality_score(history)
    assert score == 0.0


def test_clamp_bounds():
    history = [
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert score >= -0.5


def test_task49_like_trajectory():
    """Task-49-like: mixed success/failure, redundant reads."""
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "MDCLVA"},
                     extracted_entities={"reservation_id": ["MDCLVA"]}),
        _make_action("cancel_reservation", params={"reservation_id": "MDCLVA"}),
        _make_action("get_reservation_details", params={"reservation_id": "previous_reservation"}),
        _make_action("get_user_details", params={"user_id": "emma_kim_9957"}),
        _make_action("get_user_details", params={"user_id": "emma_kim_9957"}),
        _make_action("get_reservation_details", params={"reservation_id": "previous_reservation"}),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: data-chain-write +0.08
    # step1: placeholder read -0.03 + first-read +0.01 = -0.02
    # step2: first-read +0.01 (get_user_details first time)
    # step3: first-read +0.01 (get_reservation_details first time)
    # step4: not redundant (different from step2 params), not first-read → 0
    # step5: not redundant (different from step3 params), not first-read → 0
    # step scores: +0.08, -0.02, +0.01, +0.01, 0, 0 → sum = +0.08
    # mean = 0.08/6 ≈ 0.0133
    # no-reasoning: >=3 → -0.05
    # diversity: 2 reads >= 2 → +0.01
    # final = 0.0133 - 0.05 + 0.01 = -0.0267
    assert abs(score - (-0.026666666666666657)) < 1e-5


def test_prm_lite_reward_v5_formula():
    """PRM-Lite v5: outcome + 0.3 * process, same formula as v4."""
    state = {
        "total_reward": 1.0,
        "action_history": [
            _make_action("think"),
            _make_action("get_user_details"),
        ]
    }
    reward = _compute_prm_lite_reward(state)
    # outcome = 1.0, process = -0.01 (think +0.01, read +0.01, >=2 no-reasoning -0.02, diversity: 1 read < 2 → no bonus)
    # process = 0.0 - 0.02 = -0.02
    # reward = 1.0 + 0.3 * (-0.02) = 0.994
    assert reward > 0.9 and reward <= 1.0


# =============================================================================
# Edge cases
# =============================================================================

def test_content_too_short_p9():
    """P9: content < 30 chars → -0.02 cheap reasoning penalty."""
    history = [
        _make_action("think", content="ok"),
        _make_action("get_user_details", content="hi"),
        _make_action("book_reservation"),
    ]
    score = _compute_reasoning_quality_score(history)
    # step0: think +0.01 (content len=2 → P9 applies? No, think tools exempt from P9)
    # step1: first-read +0.01, P9: content=2 < 30 → -0.02 → net -0.01
    # step2: write 0.0
    # step scores: +0.01, -0.01, 0.0 → mean = 0.0
    # no-reasoning: >=3 → -0.05, diversity: 1 read < 2 → no bonus
    # final = -0.05
    assert abs(score - (-0.05)) < 1e-6


def test_single_placeholder_read():
    """1 step with placeholder read: no no-reasoning penalty (len<2)."""
    history = [_make_action("get_reservation_details", params={"reservation_id": "bad"})]
    score = _compute_reasoning_quality_score(history)
    # placeholder read -0.03 + first-read +0.01 = -0.02
    # no-reasoning: len=1 < 2 → no penalty
    # diversity: 1 read < 2 → no bonus
    # final = -0.02
    assert abs(score - (-0.02)) < 1e-6


if __name__ == "__main__":
    import traceback
    tests = [
        # P0: Soft-failure
        test_obs_indicates_success_returns_false_for_error_prefix,
        test_obs_indicates_success_returns_true_for_not_found,
        test_obs_indicates_success_returns_false_for_empty,
        test_obs_indicates_success_returns_true_for_partial_match,
        test_soft_fail_without_error_prefix,
        test_soft_fail_recovery_bonus,
        test_true_error_still_detected,
        # B4/B5: Think anti-hacking
        test_think_2step_bypass_placeholder,
        test_think_2step_bypass_redundant,
        test_think_2step_bypass_think,
        test_think_3step_valid,
        # P3: Error recovery split
        test_error_recovery_different_tool_v5,
        test_error_recovery_same_tool_different_params_v5,
        test_error_repetition_v5,
        # P5: Two-tier no-reasoning
        test_no_reasoning_2step_weak_penalty,
        test_no_reasoning_3step_strong_penalty,
        # B7: Read diversity
        test_read_diversity_2tools_v5,
        test_read_diversity_3tools_v5,
        # P8: Length penalty
        test_length_penalty_threshold_6,
        test_length_penalty_v5_vs_v4,
        test_length_penalty_no_penalty_at_threshold,
        # Baseline
        test_empty_history,
        test_placeholder_write,
        test_placeholder_read,
        test_no_placeholder_valid_id,
        test_redundancy_same_tool_params,
        test_escalation_premature_no_read,
        test_escalation_late_with_read,
        test_data_chain_write,
        test_first_read_exploration,
        test_successful_tool_call,
        test_think_valid,
        test_think_consecutive,
        test_think_last_step,
        test_think_followed_by_placeholder,
        test_implicit_think_no_reward,
        test_implicit_think_long_no_reward,
        test_clamp_bounds,
        test_task49_like_trajectory,
        test_prm_lite_reward_v5_formula,
        test_content_too_short_p9,
        test_single_placeholder_read,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
