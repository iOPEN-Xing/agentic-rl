# Hybrid Advantage Adaptation

## Scope

Hybrid Advantage combines an outcome-level GRPO signal with incremental
turn-level environment feedback. It is selected with
`algorithm.adv_estimator=grpo_hybrid`. The implementation is Monte Carlo credit
assignment; it does not invent a value baseline and is not GAE.

The current training configuration uses a small fixed turn residual so that the
verified session outcome remains dominant:

```yaml
algorithm:
  adv_estimator: grpo_hybrid
  hybrid_advantage:
    turn_gamma: 0.9
    turn_weight: 0.1
```

## Algorithm

For valid rollout `i` in prompt group `g`, let `R_i` be the session score. The
session component is normalized across rollouts, with each rollout counted once:

```text
A_session(i) = (R_i - mean_g(R)) / (std_g(R) + epsilon)
```

For assistant turn `t`, the rollout stores the incremental rewards caused by
that assistant action. Its Monte Carlo return is:

```text
G(i,t) = sum_{k=t..T_i} gamma^(k-t) * r(i,k)
A_turn(i,t) = (G(i,t) - mean_{g,t}(G)) / (std_{g,t}(G) + epsilon)
```

`A_turn(i,t)` exists only when at least two valid rollouts contain the same turn
index and the comparison has non-zero variance. It is broadcast only over that
assistant turn's policy tokens. Missing later turns, singleton comparisons, and
zero-variance comparisons fall back to `A_session`.

With fixed `turn_weight=w`, fusion is:

```text
A_final = (1 - w) * A_session + w * A_turn
```

The shipped `w=0.1` therefore gives exact coefficients `0.9 / 0.1`. A scheduled
session weight is also supported when `turn_weight` is absent:

```yaml
hybrid_advantage:
  turn_gamma: 0.9
  session_weight_start: 0.8
  session_weight_end: 0.2
  schedule: cosine  # linear or cosine
```

Fixed and scheduled fields are mutually exclusive and fail fast if mixed. All
weights and `turn_gamma` are range-checked. Component normalization operates on
rollout/turn comparison units before token broadcast, so longer responses do not
receive extra statistical weight merely by repeating a scalar over more tokens.
Because comparable component scales are part of the method, Hybrid requires
`norm_adv_by_std_in_grpo: true` and fails fast if it is disabled.

## Reward Ownership

Session and turn rewards have separate storage and are never arithmetically
added before advantage fusion:

- `Interaction.calculate_score()` owns the terminal session score placed in
  `token_level_rewards`.
- Successful tool and user-simulator steps expose only their incremental
  environment reward through `assistant_turn_rewards`.
- Multiple tool results caused by one assistant generation accumulate on that
  same assistant turn.
- A terminal step's increment belongs to the turn component; the final outcome
  is independently evaluated by `calculate_score()` for the session component.

The two components can be correlated because both observe the same environment,
but no reward value is counted twice inside either estimator.

## Rollout And Tool Contract

1. `TauBenchInteraction.start_interaction()` resets the environment and stores
   the returned initial user observation.
2. `ToolAgentLoop` injects that observation into a system-only prompt, or checks
   that an existing first user message matches it exactly.
3. Every non-empty assistant generation records one response-token span and one
   initially zero turn reward.
4. A successful tool or user event accumulates its incremental reward on the
   latest assistant span.
5. `done=True` from a tool terminates the state machine before another model
   generation. The interaction is finalized in a `finally` block on every path.

Tool failures are classified by ownership:

| Event | Training treatment | PRM-Lite treatment |
|---|---|---|
| Valid tool call | Train normally | Existing process rules |
| Unknown tool | Trainable policy error, zero turn reward | Recorded with direct `-0.10` action penalty |
| Malformed call/arguments | Trainable policy error, error response returned to policy | Recorded with direct `-0.10` action penalty |
| Tool/environment runtime exception | Infrastructure failure; terminate trajectory | No policy penalty |
| User-simulator exception | Infrastructure failure; terminate trajectory | No policy penalty |

Infrastructure failures retain diagnostics but set `valid_for_training=false`.
Post-processing zeros their entire `response_mask`; both Session GRPO and Turn-MC
also exclude zero-mask rows from group means and variances. An all-invalid group
returns zero advantages without consuming malformed event metadata.

## Evaluation Metric Correction

`src/evaluation/pass_k_eval.py` now imports tau-bench lazily, allowing metric-only
tests without the environment package. It reports the distinct estimators:

```text
pass@k = 1 - C(n-c, k) / C(n, k)   # at least one success
pass^k = C(c, k) / C(n, k)         # all k successes
```

Legacy output aliases remain, but are named explicitly in the report.

## CPU Validation

The focused server suite covers exact fixed/scheduled coefficients, reward-scale
invariance, discounted future events, variable trajectory lengths, missing later
turns, invalid/all-invalid groups, trainer wiring, initial-observation injection,
finalization, immediate tool termination, malformed and unknown actions,
infrastructure exclusion, PRM-Lite error penalties, and pass@k/pass^k formulas.

Validation on the server (CPU-only, `CUDA_VISIBLE_DEVICES=`):

- Relevant trainer, rollout, environment, parser, PRM-Lite, and evaluation tests:
  `84 passed, 1 skipped`. The skipped test is an existing async-marker test for
  the unrelated GPT-OSS parser; the new Hermes malformed-call test ran normally.
- Python compilation: passed for every changed Python module and test.
- Hydra composition: resolved `adv_estimator: grpo_hybrid`, `turn_gamma: 0.9`,
  and `turn_weight: 0.1`.
- `bash -n` for `run_hybrid_advantage.sh` and `git diff --check`: passed.

No model weights, vLLM service, CUDA kernel, or training job is started by these
checks. Full training quality still requires a controlled server experiment and
comparison against the session-only baseline.
