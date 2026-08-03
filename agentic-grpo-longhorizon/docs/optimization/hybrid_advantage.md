# TRACE-Style Hybrid Advantage Adaptation

## Scope

This branch combines the terminal, verifier-backed GRPO outcome advantage with
TRACE-style turn credit. It is selected with
`algorithm.adv_estimator=grpo_hybrid`.

The dense component is **not** a discounted sum of tau-bench step rewards and
is not GAE. Each state is scored by a frozen reference model under teacher
forcing on the same training-only gold target. Changes in the remaining target
gap estimate which assistant transition made the task easier or harder.

This distinction matters for the airline data: environment increments can be
sparse, delayed, or coupled to a write action, while TRACE asks whether a turn
improves the reference model's ability to produce the known target state.

## Algorithm

For rollout `i` in prompt group `g`, let `R_i` be the terminal score. Invalid
infrastructure trajectories are excluded before group statistics are computed.
The outcome component uses population standard deviation, matching the TRACE
definition:

```text
A_outcome(i) = (R_i - mean_g(R)) / (std_population_g(R) + epsilon)
```

For prefix state `s_t`, the frozen reference model returns the average
teacher-forced log-probability `ell_t` of the same gold target `y*`. Define:

```text
d_t = -ell_t + gap_epsilon
V_t = log(d_0 / d_t)
delta_t = V_(t+1) - V_t
```

There are `T + 1` prefix scores for `T` transition spans. For horizon `K`, the
local credit is the normalized discounted average of the available deltas:

```text
c_local(t) = sum_(j=t..h) gamma^(j-t) delta_j
             / sum_(j=t..h) gamma^(j-t)
h = min(t + K - 1, T - 1)
```

When the horizon reaches the terminal transition, the terminal outcome anchor
is added:

```text
c_terminal(t) = terminal_scale * gamma^(T-t) * A_outcome(i)
```

The final token advantage is:

```text
A_final = outcome_weight * A_outcome + turn_weight * c_t
```

`c_t` is broadcast only to the policy tokens in the corresponding transition
turn. The final-answer tail receives outcome advantage only. TRACE turn credit
is trajectory-local and is deliberately not normalized across rollout groups.
`td_horizon=0` disables dense TD deltas but keeps the final transition's outcome
anchor.

Current defaults:

```yaml
algorithm:
  adv_estimator: grpo_hybrid
  hybrid_advantage:
    outcome_weight: 1.0
    turn_weight: 0.2
    gap_epsilon: 0.1
    td_horizon: 3
    td_gamma: 0.8
    terminal_scale: 2.0
    score_temperature: 1.0
    max_scoring_length: 24576
    max_target_tokens: 2048
```

## Data Contract

Training parquet rows must contain a non-empty gold target in
`reward_model.ground_truth`. The target is for the frozen scorer only; it is not
inserted into the actor prompt. Vanilla parquet intentionally has no such
target, so `run_hybrid_advantage.sh` and the H200 launcher rebuild
method-specific parquet instead of silently reusing vanilla data.

During rollout, `ToolAgentLoop` records assistant token spans and state
boundaries. After rollout, the frozen reference worker teacher-forces the same
target at every prefix and emits:

- `trace_turn_spans`: the policy-token interval for each transition;
- `trace_prefix_avg_log_probs`: exactly one more prefix score than transition
  spans;
- `trace_final_answer_span`: the outcome-only tail, used for diagnostics.

Scoring refuses silent context truncation because truncating a state changes the
quantity being scored. A target exceeding `max_target_tokens`, a composed input
exceeding `max_scoring_length`, non-finite scores, missing frozen-reference
metadata, or span/score misalignment fails fast.

Ground-truth targets must be restricted to the training split. Evaluation
targets or hidden business outcomes must never enter the scorer, otherwise the
dense reward leaks evaluation information.

## Rollout and Failure Contract

1. `TauBenchInteraction.start_interaction()` resets the environment and exposes
   the actual initial user observation.
2. `ToolAgentLoop` injects it once into a system-only prompt, or verifies that a
   pre-existing first user message is exactly identical.
3. Every non-empty assistant generation records one policy-token span.
4. `done=True` terminates the state machine before another model generation.
5. Interaction finalization runs on success and failure paths.

Failures are separated by ownership:

| Event | Training treatment | PRM-Lite treatment |
|---|---|---|
| Valid tool call | Train normally | Existing process rules |
| Unknown tool | Trainable policy error | Direct `-0.10` action penalty |
| Malformed call/arguments | Trainable policy error with an error observation | Direct `-0.10` action penalty |
| Tool/environment runtime exception | Infrastructure failure; exclude rollout | No policy penalty |
| User-simulator or finalization exception | Infrastructure failure; exclude rollout | No policy penalty |

Infrastructure failures retain diagnostics and set
`valid_for_training=false`. Post-processing zeros the complete
`response_mask`; the outcome group statistics and TRACE metadata parser both
skip those rows. An all-invalid group therefore produces zero advantages,
without attempting to interpret partial TRACE metadata.

Tau-bench incremental rewards are still recorded for environment diagnostics
and other algorithms, but `grpo_hybrid` does not consume them as TRACE values.

## Airline Case

Suppose the user asks to change a reservation after confirming identity:

1. At `s_0`, the target state is still difficult for the frozen scorer and its
   average gold log-probability is low.
2. The policy reads the reservation. At `s_1`, the scorer's remaining gap falls;
   `delta_0 > 0`, so the read/verification turn receives positive local credit.
3. The policy calls a write tool with a wrong flight number. At `s_2`, the
   remaining gap grows; `delta_1 < 0`, assigning negative credit to that turn
   even before the final verifier fails.
4. The terminal GRPO outcome remains the authoritative business signal and is
   added to every valid response token. The last answer receives only this
   outcome component.

This provides denser attribution without letting a heuristic potential inspect
the terminal result or allowing an LLM judge to replace the environment's
business verifier.

## Evaluation and Validation

`src/evaluation/pass_k_eval.py` reports distinct estimators:

```text
pass@k = 1 - C(n-c, k) / C(n, k)   # at least one success
pass^k = C(c, k) / C(n, k)         # all k successes
```

Focused CPU/static checks:

```bash
pytest -q verl/tests/trainer/ppo/test_hybrid_advantage.py
pytest -q verl/tests/trainer/ppo/test_trace_reference.py
pytest -q verl/tests/experimental/agent_loop/test_interaction_contract.py
pytest -q agentic-grpo-longhorizon/src/training/tests/test_trace_targets.py
pytest -q agentic-grpo-longhorizon/src/evaluation/tests/test_pass_k_metrics.py
bash -n agentic-grpo-longhorizon/scripts/train/grpo/run_hybrid_advantage.sh
bash -n agentic-grpo-longhorizon/scripts/train/grpo/run_hybrid_advantage_h200_4gpu.sh
git diff --check
```

These checks do not start a model server, load model weights, or run training.
Training quality still requires controlled comparisons against outcome-only
GRPO, with terminal success, pass@k/pass^k, reward/length correlation, invalid
rollout rate, scorer truncation rate, and turn-credit scale logged separately.
