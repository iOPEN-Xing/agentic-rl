"""
Edge case: REAL failure-then-recovery trajectory (sample_idx=15)
=============================================================
This trajectory has a real tool error:
  "Error: payment amount does not add up, total price is 255, but paid 305"
followed by successful recovery (corrected payment to 250+5).

This is the perfect test case to verify PRM-Lite v5 P0 / P3 / v4 vs v5 differences.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from math import exp
from pathlib import Path

WORKSPACE = Path("/Users/xingjiezheng/Desktop/my_cursor-main/agentic-grpo-longhorizon-main")
TASK_DATA = WORKSPACE / "agentic-grpo-longhorizon" / "experiments" / "sft_collect_airline" / "task_0000.jsonl"

# Re-import the v4 and v5 functions
sys.path.insert(0, str(WORKSPACE / "agentic-rl-codex-turn-level-reward" / "agentic-grpo-longhorizon" / "scripts" / "test"))
from turn_ppo_real_data_dryrun import (
    _compute_reasoning_quality_score_v4,
    _compute_reasoning_quality_score_v5,
    _PRM_LITE_ACTION_ERROR_PENALTY,
    _READ_TOOLS,
    _WRITE_TOOLS,
    _THINK_TOOLS,
    _has_placeholder,
    _is_redundant,
    _param_str,
    extract_entities_from_obs,
)

print("=" * 75)
print("PRM-Lite v4 vs v5: FAILURE-THEN-RECOVERY case (task_0000 sample_idx=15)")
print("=" * 75)

# Load sample_idx=15
with open(TASK_DATA) as f:
    samples = [json.loads(line) for line in f]
sample = next(s for s in samples if s["sample_idx"] == 15)
print(f"\nLoaded sample: success={sample['success']} num_turns={sample['num_turns']} num_tool_calls={sample['num_tool_calls']}")

# Build action_history with REAL error from the trajectory
# Trace through tool calls:
# 1. search_direct_flight  →  obs has flights, inc_reward > 0, success
# 2. search_onestop_flight →  obs has flights, inc_reward > 0, success
# 3. get_user_details      →  obs has user profile, inc_reward > 0, success
# 4. book_reservation      →  obs: "Error: payment amount does not add up, total price is 255, but paid 305", is_error=True, inc_reward=0
# 5. book_reservation      →  obs has reservation_id, success (corrected)

action_history_recovery = [
    {
        "tool": "search_direct_flight",
        "parameters": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
        "param_str": _param_str({"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}),
        "inc_reward": 0.5,
        "is_error": False,
        "observation": '[{"flight_number": "HAT069", ...}, {"flight_number": "HAT083", ...}]',
        "extracted_entities": extract_entities_from_obs('[{"flight_number": "HAT069", "..."}]'),
        "content": "Sure, I can help with that. Let's start by searching for direct flights.",
    },
    {
        "tool": "search_onestop_flight",
        "parameters": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
        "param_str": _param_str({"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}),
        "inc_reward": 0.5,
        "is_error": False,
        "observation": '[[{...HAT057...HAT039...}, ...]]',
        "extracted_entities": extract_entities_from_obs('[[{...HAT057...HAT039...}, ...]]'),
        "content": "I found several one-stop flight options.",
    },
    {
        "tool": "get_user_details",
        "parameters": {"user_id": "mia_li_3668"},
        "param_str": _param_str({"user_id": "mia_li_3668"}),
        "inc_reward": 0.5,
        "is_error": False,
        "observation": '{"name": {"first_name": "Mia", "last_name": "Li"}, ...}',
        "extracted_entities": extract_entities_from_obs('{"name": {"first_name": "Mia", "last_name": "Li"}, ...}'),
        "content": "Let me check your account details.",
    },
    {
        # FAILED attempt: paid 305 (sum of 250+55) instead of 255
        "tool": "book_reservation",
        "parameters": {
            "user_id": "mia_li_3668", "origin": "JFK", "destination": "SEA",
            "flight_type": "one_way", "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}, {"flight_number": "HAT039", "date": "2024-05-20"}],
            "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
            "payment_methods": [
                {"payment_id": "certificate_7504069", "amount": 250},
                {"payment_id": "credit_card_4421486", "amount": 55},  # WRONG: should be 5
            ],
            "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
        },
        "param_str": _param_str({
            "user_id": "mia_li_3668", "origin": "JFK", "destination": "SEA",
            "flight_type": "one_way", "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}, {"flight_number": "HAT039", "date": "2024-05-20"}],
            "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
            "payment_methods": [
                {"payment_id": "certificate_7504069", "amount": 250},
                {"payment_id": "credit_card_4421486", "amount": 55},  # WRONG: should be 5
            ],
            "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
        }),
        "inc_reward": 0.0,
        "is_error": True,
        "observation": "Error: payment amount does not add up, total price is 255, but paid 305",
        "extracted_entities": {},
        "content": "Let me book the reservation now.",
    },
    {
        # RECOVERY: same tool, corrected amount
        "tool": "book_reservation",
        "parameters": {
            "user_id": "mia_li_3668", "origin": "JFK", "destination": "SEA",
            "flight_type": "one_way", "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}, {"flight_number": "HAT039", "date": "2024-05-20"}],
            "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
            "payment_methods": [
                {"payment_id": "certificate_7504069", "amount": 250},
                {"payment_id": "credit_card_4421486", "amount": 5},  # CORRECTED
            ],
            "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
        },
        "param_str": _param_str({
            "user_id": "mia_li_3668", "origin": "JFK", "destination": "SEA",
            "flight_type": "one_way", "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}, {"flight_number": "HAT039", "date": "2024-05-20"}],
            "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
            "payment_methods": [
                {"payment_id": "certificate_7504069", "amount": 250},
                {"payment_id": "credit_card_4421486", "amount": 5},  # CORRECTED
            ],
            "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
        }),
        "inc_reward": 1.0,
        "is_error": False,
        "observation": '{"reservation_id": "HATHAT", "user_id": "mia_li_3668", ...}',
        "extracted_entities": extract_entities_from_obs('{"reservation_id": "HATHAT", "user_id": "mia_li_3668", ...}'),
        "content": "I see there was an error, let me correct the payment amount.",
    },
]

print(f"\n  Reconstructed {len(action_history_recovery)} actions:")
for i, a in enumerate(action_history_recovery):
    err = "ERROR" if a["is_error"] else "ok"
    inc = a["inc_reward"]
    obs_preview = a["observation"][:50]
    print(f"    [{i}] {a['tool']:25s} inc_reward={inc:.1f} [{err}] obs={obs_preview!r}")

# Score
score_v4 = _compute_reasoning_quality_score_v4(action_history_recovery)
score_v5, breakdown_v5 = _compute_reasoning_quality_score_v5(action_history_recovery)

print(f"\n  PRM-Lite v4 score: {score_v4:+.6f}")
print(f"  PRM-Lite v5 score: {score_v5:+.6f}")
print(f"  Δ: {score_v5 - score_v4:+.6f}")

print(f"\n  v5 per-step breakdown:")
for s in breakdown_v5["per_step_scores"]:
    print(f"    step {s['i']} [{s['tool']:25s}] score={s['score']:+.4f}  components={s['components']}")
print(f"\n  v5 trajectory adjustments: {breakdown_v5['trajectory_adjustments']}")

# Analyze where v4 vs v5 differ
print("\n" + "=" * 75)
print("v4 vs v5 DIFFERENCE ANALYSIS")
print("=" * 75)
print("""
Key differences in this trajectory:
  1. step[3] book_reservation FAILED (inc_reward=0, obs starts with "Error:")
     - v4: P0 fires because is_error=True → -0.10
     - v5: P0 fires because _is_tool_call_failure(0, obs, tool) → True → -0.10
     → SAME in this case (string-based and inc_reward-based both flag it)

  2. step[4] book_reservation RECOVERY (same tool, different params)
     - v4: P3 check: prev.is_error=True, curr_sig != prev_sig (params differ)
       → +0.05 (recovery bonus)
     - v5: P3 check: prev_is_failure=True (v5 way), tool == prev.tool, params differ
       → +0.02 (same-tool-different-params, conservative)

  3. v4 vs v5 differ by: (0 - 0.05) + (0.02 - 0)  wait, let me recompute:
     - v4 step[3] P0 = -0.10; v5 step[3] P0 = -0.10 (same)
     - v4 step[4] P3 = +0.05 (different sig); v5 step[4] P3 = +0.02 (same tool)
     - v4 total: -0.10 + 0.05 = -0.05 (per-step at this point)
     - v5 total: -0.10 + 0.02 = -0.08 (per-step at this point)
     - v5 score should be LOWER by 0.03
""")

# Math verification
per_step_v4 = [0.0] * 5
# step 0: B2 first read = +0.01
per_step_v4[0] = 0.01
# step 1: B2 first read = +0.01
per_step_v4[1] = 0.01
# step 2: B2 first read = +0.01
per_step_v4[2] = 0.01
# step 3: P0 (is_error) = -0.10
per_step_v4[3] = -0.10
# step 4: P3 (different sig) = +0.05
per_step_v4[4] = 0.05

per_step_v5 = [0.0] * 5
# step 0: B2 first read = +0.01
per_step_v5[0] = 0.01
# step 1: B2 first read = +0.01
per_step_v5[1] = 0.01
# step 2: B2 first read = +0.01
per_step_v5[2] = 0.01
# step 3: P0 (inc_reward=0) = -0.10
per_step_v5[3] = -0.10
# step 4: P3 (same tool, different params) = +0.02
per_step_v5[4] = 0.02

mean_v4_pre = sum(per_step_v4) / 5  # 0.01+0.01+0.01-0.10+0.05 = -0.02, /5 = -0.004
mean_v5_pre = sum(per_step_v5) / 5  # 0.01+0.01+0.01-0.10+0.02 = -0.05, /5 = -0.01

# P5 (v4: only if >=3 steps AND no think) - both versions fire, both -0.05
# B7 (v4: >=3 reads; v5: >=2 reads): only 3 reads (search_direct, search_onestop, get_user_details)
#   v4: 3 reads >= 3 → +0.01; v5: 3 reads >= 2 → +0.01 (same)
# P8: 5 steps <= threshold (6 for v5, 8 for v4) → no penalty in either

v4_total = mean_v4_pre - 0.05 + 0.01
v5_total = mean_v5_pre - 0.05 + 0.01

print(f"\n  Step-by-step manual computation:")
print(f"    v4 per-step: {[round(x, 4) for x in per_step_v4]}")
print(f"    v5 per-step: {[round(x, 4) for x in per_step_v5]}")
print(f"    v4 mean pre-clip: {mean_v4_pre:+.4f}")
print(f"    v5 mean pre-clip: {mean_v5_pre:+.4f}")
print(f"    v4 with P5+B7: {v4_total:+.4f}")
print(f"    v5 with P5+B7: {v5_total:+.4f}")

# Validate against actual function
print(f"\n  Actual function outputs:")
print(f"    v4: {score_v4:+.4f}")
print(f"    v5: {score_v5:+.4f}")
print(f"    Δ: {score_v5 - score_v4:+.4f}")

# Final reward (in turn_ppo context: outcome=1 + 0.3 * process)
print(f"\n  Final turn-ppo reward (turn_level_reward.enabled=True):")
print(f"    v4: outcome + 0.3 * v4_score = 1.0 + 0.3 * ({score_v4:.4f}) = {1.0 + 0.3 * score_v4:+.4f}")
print(f"    v5: outcome + 0.3 * v5_score = 1.0 + 0.3 * ({score_v5:.4f}) = {1.0 + 0.3 * score_v5:+.4f}")

print("\n" + "=" * 75)
print("CONCLUSION: v5's P3 split provides finer-grained recovery reward:")
print("  - 'same tool, different params' (parameter correction) = +0.02 (was 0 in v4)")
print("  - 'completely different tool' (strategy change) = +0.03 (was +0.05 in v4)")
print("  Net effect: v5 is more conservative on 'just try again' vs 'correct'.")
print("=" * 75)
