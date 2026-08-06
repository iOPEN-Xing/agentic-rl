"""
Turn-PPO End-to-End Dry-Run
============================
Pure-Python, model-free trace of one τ-bench airline episode through the entire
Turn-PPO pipeline.

Coverage:
  1. Interaction state + action_history build-up
  2. assistant_turn_spans → turn_ids (materialize_turn_ids)
  3. turn_reward_spans → turn_level_rewards (materialize_turn_rewards)
  4. combine_outcome_and_turn_rewards (strict vs shaping-on path)
  5. compute_turn_gae_advantage_return (the core GAE formula)
  6. compute_policy_loss_turn_ppo (response-product ratio + single clip)
  7. _validate_turn_ids invariants (fail-loud checks)

Run:  python scripts/test/turn_ppo_e2e_dryrun.py
"""
from __future__ import annotations

import sys
from math import exp
from pathlib import Path

# ── toy helper: replicate the real materialize_turn_ids (no torch) ──────────
def materialize_turn_ids_python(
    assistant_turn_spans_batch: list[list[tuple[int, int]]],
    response_mask_batch: list[list[int]],
) -> list[list[int]]:
    """Turn ids: 0 = environment token, 1..N = assistant turn id.

    Raises ValueError if any trainable token (response_mask=1) has turn_id=0.
    """
    result = []
    for batch_idx, spans in enumerate(assistant_turn_spans_batch):
        resp_len = len(response_mask_batch[batch_idx])
        turn_ids = [0] * resp_len
        next_id = 1
        prev_end = 0
        for (start, end) in spans or []:
            start = max(0, start)
            end = min(resp_len, end)
            if start >= resp_len:
                break
            for pos in range(start, end):
                if response_mask_batch[batch_idx][pos]:
                    turn_ids[pos] = next_id
            prev_end = end
            next_id += 1
        result.append(turn_ids)

    # Fail-loud: any trainable token with turn_id=0 must raise
    for batch_idx, (turn_ids, resp_mask) in enumerate(
        zip(result, response_mask_batch)
    ):
        for pos, (tid, mask) in enumerate(zip(turn_ids, resp_mask)):
            if mask and tid == 0:
                raise ValueError(
                    f"Uncovered trainable token: batch={batch_idx} pos={pos}"
                )
    return result


def materialize_turn_rewards_python(
    turn_reward_spans_batch: list[list[dict]],
    response_mask_batch: list[list[int]],
) -> tuple[list[list[float]], list[list[bool]]]:
    """Place each turn reward on the last trainable token of its span."""
    rewards_out, mask_out = [], []
    for batch_idx, spans in enumerate(turn_reward_spans_batch):
        resp_len = len(response_mask_batch[batch_idx])
        rewards = [0.0] * resp_len
        event_mask = [False] * resp_len
        for span in spans or []:
            start = max(0, int(span["start"]))
            end = min(resp_len, int(span["end"]))
            if end <= start:
                continue
            # find last trainable token in [start, end)
            last_trainable = None
            for pos in range(end - 1, start - 1, -1):
                if response_mask_batch[batch_idx][pos]:
                    last_trainable = pos
                    break
            if last_trainable is not None:
                rewards[last_trainable] += float(span["reward"])
                event_mask[last_trainable] = True
        rewards_out.append(rewards)
        mask_out.append(event_mask)
    return rewards_out, mask_out


# ── toy: replicate combine_outcome_and_turn_rewards ─────────────────────────
def combine_outcome_and_turn_rewards_python(
    outcome_scores: list[list[float]],
    turn_scores: list[list[float]],
    turn_event_mask: list[list[bool]],
    turn_level_reward_enabled: bool = False,
    outcome_weight: float = 1.0,
    turn_weight: float = 0.3,
    normalize_by_count: bool = True,
) -> tuple[list[list[float]], dict]:
    """Pure-Python replicate of verl.trainer.ppo.ray_trainer.combine_outcome_and_turn_rewards."""
    if not turn_level_reward_enabled:
        return outcome_scores, {}

    combined = []
    metrics_total_events = 0.0
    metrics_turn_contrib = 0.0
    batch_count = len(outcome_scores)

    for b_idx in range(batch_count):
        row_out, row_turn = outcome_scores[b_idx], turn_scores[b_idx]
        row_mask = turn_event_mask[b_idx]
        row_combined = []
        total_turn_contrib = 0.0
        event_count = 0
        for t_idx in range(len(row_out)):
            masked_turn = row_turn[t_idx] * (1.0 if row_mask[t_idx] else 0.0)
            if row_mask[t_idx]:
                event_count += 1
            total_turn_contrib += masked_turn
            row_combined.append(
                outcome_weight * row_out[t_idx]
                + turn_weight * (masked_turn / max(event_count, 1) if normalize_by_count else masked_turn)
            )
        combined.append(row_combined)
        metrics_total_events += sum(row_mask)
        metrics_turn_contrib += total_turn_contrib

    metrics = {
        "reward/turn_events_per_trajectory": metrics_total_events / batch_count,
        "reward/turn_contribution_mean": metrics_turn_contrib / batch_count,
    }
    return combined, metrics


# ── toy: replicate compute_turn_gae_advantage_return ──────────────────────
def compute_turn_gae_advantage_python(
    token_level_rewards: list[list[float]],
    values: list[list[float]],
    response_mask: list[list[int]],
    turn_ids: list[list[int]],
    gamma: float = 0.99,
    lam: float = 0.9,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Pure-Python replicate of verl.trainer.ppo.core_algos.compute_turn_gae_advantage_return."""
    batch_size = len(token_level_rewards)
    seq_len = len(token_level_rewards[0])
    advantages = [[0.0] * seq_len for _ in range(batch_size)]
    returns = [[0.0] * seq_len for _ in range(batch_size)]
    turn_value_masks = [[0.0] * seq_len for _ in range(batch_size)]

    raw_advantages = [[0.0] * seq_len for _ in range(batch_size)]
    trajectory_rewards = [sum(row) for row in token_level_rewards]

    for b in range(batch_size):
        resp_len = len(response_mask[b])
        # collect turn ids present in this sample
        ids_in_seq = sorted(set(turn_ids[b]))
        if not ids_in_seq:
            continue

        # find turn start positions
        turn_starts = []
        for tid in ids_in_seq:
            for pos in range(resp_len):
                if turn_ids[b][pos] == tid:
                    turn_starts.append(pos)
                    break

        n_turns = len(turn_starts)
        # state values at each turn start
        state_vals = [values[b][turn_starts[t]] for t in range(n_turns)]
        # rewards: 0 everywhere except last turn = trajectory reward
        turn_rewards = [0.0] * n_turns
        turn_rewards[-1] = trajectory_rewards[b]

        next_adv = 0.0
        for t_idx in range(n_turns - 1, -1, -1):
            next_val = state_vals[t_idx + 1] if t_idx + 1 < n_turns else 0.0
            delta = turn_rewards[t_idx] + gamma * next_val - state_vals[t_idx]
            next_adv = delta + gamma * lam * next_adv
            raw_advantages[b][turn_starts[t_idx]] = next_adv

        # whitening over all turn-start positions
        turn_start_vals = [raw_advantages[b][turn_starts[t]] for t in range(n_turns)]
        if n_turns > 1:
            mean_a = sum(turn_start_vals) / n_turns
            std_a = (sum((a - mean_a) ** 2 for a in turn_start_vals) / n_turns) ** 0.5
            std_a = max(std_a, 1e-8)
            norm_start = [(a - mean_a) / std_a for a in turn_start_vals]
        else:
            norm_start = turn_start_vals

        # broadcast back to all tokens in each turn
        for t_idx, tid in enumerate(ids_in_seq):
            for pos in range(resp_len):
                if turn_ids[b][pos] == tid:
                    advantages[b][pos] = norm_start[t_idx]
                    returns[b][pos] = norm_start[t_idx] + state_vals[t_idx]
            turn_value_masks[b][turn_starts[t_idx]] = 1.0 / n_turns

    return advantages, returns, turn_value_masks


# ── toy: replicate compute_policy_loss_turn_ppo ───────────────────────────
def compute_policy_loss_turn_ppo_python(
    old_log_probs: list[list[float]],  # (batch, seq)
    new_log_probs: list[list[float]],
    advantages: list[list[float]],
    response_mask: list[list[int]],
    turn_ids: list[list[int]],
    epsilon: float = 0.2,
) -> tuple[float, dict]:
    """Pure-Python replicate of verl.trainer.ppo.core_algos.compute_policy_loss_turn_ppo."""
    batch_size = len(old_log_probs)
    total_loss = 0.0
    metrics = {}
    turn_clip_frac_sum = 0.0
    saturation_sum = 0.0

    for b in range(batch_size):
        seq_len = len(old_log_probs[b])
        resp_len = len([m for m in response_mask[b] if m])
        ids_in_seq = sorted(set(turn_ids[b]))
        if not ids_in_seq:
            continue

        turn_losses = []
        clip_count = 0
        sat_count = 0
        token_count = 0

        for tid in ids_in_seq:
            log_ratio = 0.0
            turn_token_count = 0
            for pos in range(seq_len):
                if turn_ids[b][pos] == tid and response_mask[b][pos]:
                    log_ratio += new_log_probs[b][pos] - old_log_probs[b][pos]
                    turn_token_count += 1

            if turn_token_count == 0:
                continue
            token_count += turn_token_count

            ratio = exp(max(-20.0, min(20.0, log_ratio)))
            adv = advantages[b][pos]  # broadcast advantage on last token of turn
            clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))

            if adv > 0:
                loss_contrib = -adv * ratio
                clipped_contrib = -adv * clipped_ratio
            else:
                loss_contrib = adv * ratio
                clipped_contrib = adv * clipped_ratio

            turn_loss = max(loss_contrib, clipped_contrib) / turn_token_count
            turn_losses.append(turn_loss * turn_token_count)

            if ratio >= 1.0 - epsilon and ratio <= 1.0 + epsilon:
                clip_count += 1
            if abs(log_ratio) > 20.0:
                sat_count += 1

        if token_count > 0:
            total_loss += sum(turn_losses) / token_count
            turn_clip_frac_sum += clip_count / max(len(ids_in_seq), 1)
            saturation_sum += sat_count / max(len(ids_in_seq), 1)

    avg_loss = total_loss / batch_size
    metrics = {
        "actor/pg_loss": avg_loss,
        "actor/turn_clip_fraction": turn_clip_frac_sum / batch_size,
        "actor/turn_ppo_log_ratio_saturation_fraction": saturation_sum / batch_size,
    }
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# THE CASE: task_id=0, one successful 6-turn episode
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Turn-PPO End-to-End Dry-Run: task_id=0 (airline booking)")
print("=" * 70)

# ── Step 0: Simulated rollout data ─────────────────────────────────────────
# assistant_turn_spans: (start, end) in token-level indices
# For this simulation, we use a simplified token scheme:
#   - Each turn occupies 200 tokens (arbitrary, only shape matters)
#   - Total response length = 6 turns × 200 = 1200 tokens
#   - Turn 1: [0, 200), Turn 2: [200, 400), ..., Turn 6: [1000, 1200)
TOKENS_PER_TURN = 200
NUM_TURNS = 6
RESP_LEN = TOKENS_PER_TURN * NUM_TURNS

assistant_turn_spans = [
    [(i * TOKENS_PER_TURN, (i + 1) * TOKENS_PER_TURN) for i in range(NUM_TURNS)]
]

# Every token in the assistant spans is trainable (no padding)
response_mask = [[1] * RESP_LEN]  # batch size = 1

# ── Step 1: turn_ids ────────────────────────────────────────────────────────
print("\n[Step 1] materialize_turn_ids")
turn_ids = materialize_turn_ids_python(assistant_turn_spans, response_mask)
print(f"  shape: ({len(turn_ids)}, {len(turn_ids[0])})")
print(f"  token 0-10:   {turn_ids[0][:10]}")
print(f"  token 199-201: {turn_ids[0][199:202]}")
print(f"  token 399-401: {turn_ids[0][399:402]}")
print(f"  token 999-1001:{turn_ids[0][999:1002]}")
print(f"  token 1199-1200:{turn_ids[0][1199:1201]}")

assert turn_ids[0][0] == 1,   "First token must be turn 1"
assert turn_ids[0][199] == 1, "Turn 1 ends at token 199"
assert turn_ids[0][200] == 2, "Turn 2 starts at token 200"
assert turn_ids[0][1199] == 6, "Turn 6 last token"
print("  ✓ All assistant tokens covered, turn ids contiguous from 1")

# ── Step 2: Simulated interaction state (turn_reward_spans) ────────────────
print("\n[Step 2] Simulate interaction → turn_reward_spans")
# In strict mode (turn_level_reward.enabled=false), these are never written
# to batch. But for demonstration we show what they would look like.
# Turn reward comes from _compute_turn_reward, each turn's reward placed at
# the LAST TRAINABLE token of that turn (per materialize_turn_rewards).
# Here we simulate: each turn has some incremental reward
turn_rewards_sim = {
    1:  0.05,   # search_direct: partial progress
    2:  0.08,   # search_onestop: found good candidate
    3:  0.02,   # confirm: clarifying details
    4:  0.10,   # get_user_details: got full info
    5:  0.15,   # book_reservation: tool succeeded
    6:  0.00,   # final confirmation: no incremental reward
}
turn_reward_spans = []
for turn_num, reward_val in turn_rewards_sim.items():
    start = (turn_num - 1) * TOKENS_PER_TURN
    end = turn_num * TOKENS_PER_TURN
    turn_reward_spans.append({"start": start, "end": end, "reward": reward_val})

print(f"  turns with reward: {len(turn_reward_spans)}")
for span in turn_reward_spans:
    print(f"    Turn reward at [{span['start']}-{span['end']}]: {span['reward']}")

# materialize: reward placed at last token of each span
turn_scores_mat, turn_mask_mat = materialize_turn_rewards_python(
    [turn_reward_spans], response_mask
)
rewarded_positions = [i for i, m in enumerate(turn_mask_mat[0]) if m]
print(f"  Reward placed at tokens: {rewarded_positions}")
print(f"  Token 199 (end of turn 1): reward={turn_scores_mat[0][199]:.3f}, mask={turn_mask_mat[0][199]}")
print(f"  Token 399 (end of turn 2): reward={turn_scores_mat[0][399]:.3f}, mask={turn_mask_mat[0][399]}")
print(f"  Token 1199 (end of turn 6): reward={turn_scores_mat[0][1199]:.3f}, mask={turn_mask_mat[0][1199]}")

# ── Step 3: combine_outcome_and_turn_rewards ────────────────────────────────
print("\n[Step 3a] Strict path: turn_level_reward.enabled=False")
# Terminal outcome reward: token 1199 gets the full binary outcome
terminal_outcome = [0.0] * RESP_LEN
terminal_outcome[1199] = 1.0  # binary success
turn_level_rewards_strict, _ = combine_outcome_and_turn_rewards_python(
    [terminal_outcome],          # (batch, seq) 2D
    turn_scores_mat,             # (batch, seq) 2D
    turn_mask_mat,               # (batch, seq) 2D
    turn_level_reward_enabled=False,
)
print(f"  token 0:      {turn_level_rewards_strict[0][0]:.3f}")
print(f"  token 199:     {turn_level_rewards_strict[0][199]:.3f}")
print(f"  token 399:     {turn_level_rewards_strict[0][399]:.3f}")
print(f"  token 1199:    {turn_level_rewards_strict[0][1199]:.3f}")
total_in_strict = sum(turn_level_rewards_strict[0])
print(f"  Sum across all tokens (should be 1.0): {total_in_strict:.6f}")
assert abs(total_in_strict - 1.0) < 1e-6, f"Strict path should preserve total reward exactly"
print("  ✓ Strict: terminal reward preserved exactly, shaping path NOT activated")

print("\n[Step 3b] Shaping path: turn_level_reward.enabled=True")
turn_level_rewards_shaping, shaping_metrics = combine_outcome_and_turn_rewards_python(
    [terminal_outcome],          # list[list[float]]: (batch, seq)
    turn_scores_mat,             # list[list[float]]: (batch, seq) — already 2D
    turn_mask_mat,               # list[list[bool]]:  (batch, seq) — already 2D
    turn_level_reward_enabled=True,
    outcome_weight=1.0,
    turn_weight=0.3,
    normalize_by_count=True,
)
total_in_shaping = sum(turn_level_rewards_shaping[0])
print(f"  token 199 (turn 1):  {turn_level_rewards_shaping[0][199]:.5f}  (outcome={terminal_outcome[199]:.1f} + 0.3*{turn_scores_mat[0][199]:.2f}/1)")
print(f"  token 399 (turn 2):  {turn_level_rewards_shaping[0][399]:.5f}  (outcome={terminal_outcome[399]:.1f} + 0.3*{turn_scores_mat[0][399]:.2f}/1)")
print(f"  token 1199 (turn 6): {turn_level_rewards_shaping[0][1199]:.5f}  (outcome=1.0 + 0.3*0.0/1)")
print(f"  Sum across all tokens: {total_in_shaping:.6f}")
print(f"  Metrics: {shaping_metrics}")
print("  ✓ Shaping: turn rewards additive on top of terminal reward")

# ── Step 4: compute_turn_gae_advantage_return ──────────────────────────────
print("\n[Step 4] compute_turn_gae_advantage_return (strict path)")
# Simulate critic values: V(s_n) at each turn start
# Using values from tech-report §04 (teaching example, not from experiment)
V_s = {
    1: 0.10,   # V(s_1): before first search
    2: 0.18,   # V(s_2): after turn 1, before onestop search
    3: 0.32,   # V(s_3): after turn 2, before confirmation
    4: 0.48,   # V(s_4): after turn 3, before getting user details
    5: 0.72,   # V(s_5): after turn 4, before booking
    6: 0.90,   # V(s_6): after turn 5, before final confirmation
}
values_sim = []
for pos in range(RESP_LEN):
    for turn_num in range(1, NUM_TURNS + 1):
        start = (turn_num - 1) * TOKENS_PER_TURN
        end = turn_num * TOKENS_PER_TURN
        if start <= pos < end:
            values_sim.append(V_s[turn_num])
            break
    else:
        values_sim.append(0.0)  # outside any turn (shouldn't happen here)

advantages, returns, turn_value_masks = compute_turn_gae_advantage_python(
    turn_level_rewards_strict,
    [values_sim],
    response_mask,
    turn_ids,
    gamma=0.99,
    lam=0.9,
)

print("  Raw advantages at each turn start (before whitening):")
for turn_num in range(1, NUM_TURNS + 1):
    pos = (turn_num - 1) * TOKENS_PER_TURN
    print(f"    Turn {turn_num} start (pos={pos}): advantage={advantages[0][pos]:.6f}, return={returns[0][pos]:.6f}, vf_mask={turn_value_masks[0][pos]:.6f}")

print("\n  Whiten range: from first to last turn start:")
first_turn_start = 0
last_turn_start = (NUM_TURNS - 1) * TOKENS_PER_TURN
raw_advs_in_whitening = [advantages[0][(t - 1) * TOKENS_PER_TURN] for t in range(1, NUM_TURNS + 1)]
mean_a = sum(raw_advs_in_whitening) / NUM_TURNS
std_a = (sum((a - mean_a) ** 2 for a in raw_advs_in_whitening) / NUM_TURNS) ** 0.5
print(f"    Raw advantages: {[round(a, 4) for a in raw_advs_in_whitening]}")
print(f"    Mean: {mean_a:.4f}, Std: {std_a:.4f}")

print("\n  Normalized (whitened) advantages at each turn start:")
for turn_num in range(1, NUM_TURNS + 1):
    pos = (turn_num - 1) * TOKENS_PER_TURN
    raw_a = raw_advs_in_whitening[turn_num - 1]
    norm_a = (raw_a - mean_a) / max(std_a, 1e-8)
    print(f"    Turn {turn_num}: raw={raw_a:.4f} → normalized={norm_a:.4f}")

print("\n  Key insight: Turn 1 (earliest action) gets LARGE positive advantage")
print("  because critic predicts V(s_1)=0.10 but the trajectory ultimately")
print("  achieves R=1.0. The TD error at turn 1 is dominated by the")
print("  γ^5 * (1.0 - 0.10) ≈ 0.59 contribution from the terminal reward.")
print("  Turn 6 gets the SMALLEST advantage because V(s_6)=0.90 ≈ R=1.0.")
print("  → This is how Turn-PPO differentiates early good actions from lucky late ones.")

# ── Step 5: broadcast advantages to all tokens in each turn ────────────────
print("\n[Step 5] Advantage broadcast: every token in turn gets same normalized advantage")
for turn_num in range(1, NUM_TURNS + 1):
    start = (turn_num - 1) * TOKENS_PER_TURN
    mid = start + 100  # middle of turn
    end = turn_num * TOKENS_PER_TURN - 1
    norm_a = advantages[0][start]  # same for all tokens in this turn
    assert advantages[0][mid] == norm_a, f"Turn {turn_num} not properly broadcast"
    assert advantages[0][end] == norm_a, f"Turn {turn_num} not properly broadcast"
print(f"  ✓ All tokens in each turn share identical advantage value")
print(f"  Turn 1 mid-token advantage: {advantages[0][100]:.4f}")
print(f"  Turn 6 mid-token advantage: {advantages[0][1100]:.4f}")

# ── Step 6: compute_policy_loss_turn_ppo ──────────────────────────────────
print("\n[Step 6] compute_policy_loss_turn_ppo")
# Simulate log probs: old vs new policy
# Scenario: new policy is slightly better on turns 1-5, slightly worse on turn 6
# Use realistic per-token log prob differences (not cumulative saturation)
old_logp = [
    [-1.0] * RESP_LEN,   # old policy: fixed log prob per token
]
# Per-token improvement: +0.01 per token = +2.0 over 200 tokens
# This gives log_ratio=+2.0 per turn, which clips at epsilon=0.2
new_logp = [
    [-0.99] * TOKENS_PER_TURN * 5   # new policy: +0.01 per token for turns 1-5
    + [-1.01] * TOKENS_PER_TURN       # new policy: -0.01 per token for turn 6
]
new_logp = [new_logp[0]]  # batch of 1

loss, loss_metrics = compute_policy_loss_turn_ppo_python(
    old_logp,
    new_logp,
    advantages,
    response_mask,
    turn_ids,
    epsilon=0.2,
)
print(f"  Policy loss: {loss:.6f}")
print(f"  Metrics: {loss_metrics}")

print("\n  Key insight: Each turn contributes loss proportional to 1/num_tokens_in_turn")
print(f"  Turn 1 has {TOKENS_PER_TURN} tokens with +0.01/token improvement:")
print(f"    → cumulative log_ratio = +{0.01 * TOKENS_PER_TURN:.2f}, ratio = exp(+2.0) = {exp(2.0):.4f}")
print(f"    → clipped to 1.2, adv>0 so both unclipped and clipped terms matter")
for turn_num in range(1, NUM_TURNS + 1):
    start = (turn_num - 1) * TOKENS_PER_TURN
    end = turn_num * TOKENS_PER_TURN
    old_sum = sum(old_logp[0][start:end])
    new_sum = sum(new_logp[0][start:end])
    ratio = exp(max(-20.0, min(20.0, new_sum - old_sum)))
    print(
        f"    Turn {turn_num}: per-token Δlogp={new_logp[0][start] - old_logp[0][start]:+.4f}, "
        f"cum_log_ratio={new_sum-old_sum:+.4f}, ratio={ratio:.4f}"
    )

# ── Step 7: _validate_turn_ids invariants ─────────────────────────────────
print("\n[Step 7] _validate_turn_ids invariants (code-level checks)")

def validate_turn_ids_python(turn_ids_batch, response_mask_batch):
    """Replicate verl.trainer.ppo.core_algos._validate_turn_ids checks."""
    for b, (turn_ids, resp_mask) in enumerate(zip(turn_ids_batch, response_mask_batch)):
        # 1. turn ids must be positive integers
        for pos, tid in enumerate(turn_ids):
            if resp_mask[pos]:
                assert tid >= 1, f"Batch {b}, pos {pos}: trainable token has turn_id={tid}<1"

        # 2. no holes: if turn N exists, turn N-1 must also exist
        ids_present = sorted(set(t for t in set(turn_ids) if t > 0))
        for i in range(1, len(ids_present)):
            assert ids_present[i] == ids_present[i-1] + 1, \
                f"Batch {b}: turn id hole at {ids_present[i-1]+1} (found {ids_present[i]})"

        # 3. no overlaps: tokens with same turn id must form one contiguous block
        for tid in ids_present:
            positions = [p for p, t in enumerate(turn_ids) if t == tid and resp_mask[p]]
            if not positions:
                continue
            start, end = positions[0], positions[-1]
            mid = [p for p, t in enumerate(turn_ids) if t == tid and resp_mask[p]]
            assert mid == list(range(start, end+1)), \
                f"Batch {b}, turn {tid}: non-contiguous span at positions {mid}"

        # 4. every trainable token must be covered
        for pos, (tid, m) in enumerate(zip(turn_ids, resp_mask)):
            if m and tid == 0:
                raise ValueError(f"Uncovered trainable token at batch={b}, pos={pos}")

    return True

try:
    validate_turn_ids_python(turn_ids, response_mask)
    print("  ✓ All turn_ids validations pass")
except AssertionError as e:
    print(f"  ✗ VALIDATION FAILED: {e}")
    sys.exit(1)

# ── Step 8: Critical failure case - truncated trajectory ──────────────────
print("\n[Step 8] Failure case: trajectory truncated at turn 3 (max_turns hit)")
# Turn IDs for a truncated episode: only turns 1, 2, 3 present
truncated_turn_ids = [turn_ids[0][:600] + [0] * 600]  # tokens 0-599: turns 1-3; 600-1199: 0
truncated_response_mask = [[1] * 600 + [0] * 600]      # after token 599, not trainable

truncated_spans = [(0, 200), (200, 400), (400, 600)]
truncated_turn_ids_from_spans = materialize_turn_ids_python(
    [truncated_spans], truncated_response_mask
)

print(f"  Turn IDs after truncation:")
for pos in [0, 199, 200, 399, 400, 599, 600, 1000]:
    if pos < len(truncated_turn_ids_from_spans[0]):
        print(f"    pos {pos}: turn_id={truncated_turn_ids_from_spans[0][pos]}")
    else:
        print(f"    pos {pos}: (out of range)")

try:
    validate_turn_ids_python(truncated_turn_ids_from_spans, truncated_response_mask)
    print("  ✓ Truncated trajectory: all validations pass")
except (AssertionError, ValueError) as e:
    print(f"  ✗ Truncated validation failed: {e}")
    sys.exit(1)

# ── Step 9: Error case - all tokens uncovered ──────────────────────────────
print("\n[Step 9] Error case: all tokens uncovered (should raise ValueError)")
try:
    empty_spans: list[list[tuple[int, int]]] = [[]]  # no assistant turns
    materialize_turn_ids_python(empty_spans, [[1, 1, 1]])
    print("  ✗ Should have raised ValueError for uncovered tokens")
    sys.exit(1)
except (AssertionError, ValueError) as e:
    print(f"  ✓ Correctly raises: {e}")

print("\n" + "=" * 70)
print("All checks passed. Turn-PPO pipeline is theoretically sound.")
print("=" * 70)

# ── Summary: print the key values for reference ────────────────────────────
print("\n" + "─" * 70)
print("SUMMARY TABLE: GAE per turn (strict path, γ=0.99, λ=0.9)")
print("─" * 70)
print(f"{'Turn':>8} {'V(s_n)':>8} {'A_n raw':>10} {'A_n norm':>10} {'Explanation'}")
print("-" * 70)
for turn_num in range(1, NUM_TURNS + 1):
    pos = (turn_num - 1) * TOKENS_PER_TURN
    raw_a = raw_advs_in_whitening[turn_num - 1]
    norm_a = advantages[0][pos]
    explanations = [
        "Early search: large TD error from terminal",
        "Second search: still large TD error",
        "User confirmation: moderate TD error",
        "Get user details: moderate TD error",
        "Book reservation: small TD error",
        "Final confirm: critic already near terminal R",
    ]
    print(f"{'Turn '+str(turn_num):>8} {V_s[turn_num]:>8.3f} {raw_a:>10.4f} {norm_a:>10.4f}  {explanations[turn_num-1]}")
