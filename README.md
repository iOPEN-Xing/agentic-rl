# Agentic RL on τ-bench Airline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository studies long-horizon, multi-turn tool use with GRPO on the
50-task τ-bench airline domain.  This README describes the behavior of the
current code after a static training-logic audit.  In particular, it separates
trajectory-level reward shaping from genuine turn-level credit assignment.

## Scope and Audited Status

The `main` branch contains the original ablation track:

| Track | Reward presented to GRPO | Token advantage | What it really changes |
|---|---|---|---|
| Vanilla | binary τ-bench terminal outcome | scalar group-relative advantage broadcast to all assistant tokens | outcome-only baseline |
| Turn-discount | binary terminal outcome | scalar advantage × absolute response-position weights | positional prior; no semantic turn boundary |
| Legacy LATA | binary terminal outcome | scalar advantage × position weights ÷ `sqrt(L)` | positional prior plus extra length damping |
| PRM-Lite | `outcome + 0.3 × process_score` | one group-relative scalar broadcast to all assistant tokens | rule-based trajectory shaping |
| PRM-Lite + LATA | shaped trajectory score | position-weighted and length-damped scalar | combination of the two mechanisms above |

All five paths are internally executable, but three claims from the earlier
README were stronger than the implementation supports:

1. PRM-Lite calculates per-action features, then averages them into one
   trajectory score.  It is not a learned PRM and does not attach a distinct
   return to each assistant turn.
2. Legacy LATA does not replace actor loss normalization `1/L` with
   `1/sqrt(L)`.  It multiplies token advantages by an additional
   `1/sqrt(L)` before the default token-mean reducer, so long responses receive
   additional damping.
3. Neither turn-discount nor legacy LATA parses assistant/tool boundaries.
   “Turn” in these names refers to a response-position heuristic, not an MDP
   decision boundary.

For true boundary-aware alternatives, see the repository branches
[`codex/turn-level-reward`](https://github.com/iOPEN-Xing/agentic-rl/tree/codex/turn-level-reward),
[`codex/hybrid-advantage`](https://github.com/iOPEN-Xing/agentic-rl/tree/codex/hybrid-advantage),
and
[`codex/user-simulator-judge`](https://github.com/iOPEN-Xing/agentic-rl/tree/codex/user-simulator-judge).

## Training Signal, End to End

```text
τ-bench task + database snapshot
          │
          ▼
Qwen user simulator ↔ policy ↔ airline tools
          │                    │
          │                    └─ action history / tool observations
          ▼
τ-bench terminal verifier: success ∈ {0, 1}
          │
          ├─ binary mode ───────────────────────────┐
          │                                         │
          └─ PRM-Lite rules → mean process score ───┤
                                                    ▼
                                      one trajectory reward R_i
                                                    │
                     group by original task, n=8 rollouts
                                                    │
                                                    ▼
                                  A_i = (R_i - μ_g)/(σ_g + ε)
                                                    │
                         ┌──────────────────────────┴─────────────────┐
                         ▼                                            ▼
                vanilla broadcast                       position/length transform
                         │                                            │
                         └──────────── clipped actor objective ───────┘
```

This distinction matters on τ-bench.  Suppose an airline task requires the
agent to read a reservation, verify passenger intent, update two flights and
then communicate the result.  A failed trajectory may contain three correct
tool calls and one invalid write.  PRM-Lite can make its final scalar less bad
than a trajectory that repeatedly calls invalid tools, helping an all-zero
GRPO group recover some variance.  The final scalar is nevertheless broadcast
to all assistant tokens, so the optimizer still cannot know that only the last
write decision caused failure.

## The Four Main Ablations

### Vanilla GRPO

For rollouts generated from the same task, the code sums terminal token rewards,
computes the group mean and standard deviation, and broadcasts the normalized
scalar to response tokens.  Binary rewards produce zero gradient when every
rollout in a group succeeds or every rollout fails.  This is the principal
sparse-reward limitation, not an implementation bug.

### Turn-discount

The legacy estimator uses

```text
w_t ∝ α^(L - 1 - t),     mean_t(w_t) = 1
A_token,t = A_GRPO × w_t
```

with `alpha=1.05`.  It prefers early response positions.  It can act as a
regularizer, but it cannot distinguish an early correct read from an early
hallucinated write, and its semantics move when message serialization changes.

### Legacy LATA

The implemented estimator is

```text
A_token,t = A_GRPO × w_t × mask_t / sqrt(L)
```

where the actor still uses a token-mean loss.  Consequently this is best viewed
as a length-damped positional prior.  Historical LATA results in this repository
measure that implementation; they are not evidence that `1/L` was replaced by
`1/sqrt(L)`.

### PRM-Lite

`src/envs/tau_bench_interaction.py` scores action-history properties such as
placeholder arguments, repeated errors, read-before-write, recovery and tool
chains.  Individual action scores are averaged and clipped to `[-0.5, 0.5]`:

```text
process_score = clip(mean(action_rule_scores) + trajectory_adjustments)
R = terminal_outcome + 0.3 × process_score
```

This is a deterministic heuristic reward shaper.  It is cheap and interpretable,
but has four important limits:

- it can reward a plausible process that does not satisfy the database verifier;
- rule coverage is domain-specific and must be re-audited for every tool schema;
- mean aggregation loses the exact temporal location of a good or bad action;
- the policy may learn the proxy unless rule-hacking rates are monitored.

A production PRM should instead be trained or calibrated on step-level human,
verifier or preference labels, report calibration and conflict with terminal
success, and expose turn-local scores to an estimator that preserves their
location.

## Historical Results: Interpretation Rules

The repository contains an earlier single-run ablation table in which the best
recorded joint checkpoint reported a success fraction of `0.240`, versus
`0.175` for the selected vanilla checkpoint.  Treat these as historical
observations, not a statistically established algorithmic gain:

- checkpoint selection differed across tracks;
- the benchmark has only 50 tasks and four samples per task in this protocol;
- no seed confidence interval is reported;
- the old evaluator used ambiguous `pass_hat_k` names;
- the joint run evaluates the actual length-damped LATA implementation described
  above, not the older README's theoretical formula.

The corrected evaluator now reports both estimator families explicitly.

## Unified Evaluation Contract

All method launchers use the same airline contract unless explicitly overridden:

| Setting | Value |
|---|---|
| environment | τ-bench `airline` |
| tasks | fixed split file, 50 tasks |
| samples per task | 4 |
| policy sampling | temperature `0.7`, top-p `0.9`, max tokens `4096` |
| user simulator | fixed Qwen3-14B endpoint on port `8001` |
| policy endpoint | served alias `agentic-rl-policy` on port `8000` |
| interaction budget | 30 turns |

For a task with `n` samples and `c` successes:

```text
pass@k = 1 - C(n-c, k) / C(n, k)   # at least one of k succeeds
pass^k = C(c, k) / C(n, k)         # all k samples succeed
```

These metrics are not interchangeable.  The report also writes
`any_success = 1[c > 0]`.  Legacy aliases remain in JSON only with explicit
names so old analysis scripts can migrate without silently changing meaning.

The common launcher refuses to kill an unknown process on port `8000`; it owns
and cleans up only the vLLM process it starts.  Evaluation checkpoints can be
selected with `AGENTIC_RL_EVAL_STEPS`, and the task split can be pinned with
`AGENTIC_RL_EVAL_SPLIT_FILE`.

```bash
cd agentic-grpo-longhorizon

# Standard vanilla checkpoints: 50, 100, 150, 200
bash scripts/eval/eval_vanilla.sh

# Legacy ablation checkpoints default to 200, 250, 300
bash scripts/eval/eval_exp1_turn_discount.sh
bash scripts/eval/eval_exp2_lata.sh
bash scripts/eval/eval_exp3_prm_lite.sh
bash scripts/eval/eval_exp4_prm_lite_lata.sh

# Example override
AGENTIC_RL_EVAL_STEPS="100 200" \
AGENTIC_RL_EVAL_SPLIT_FILE=/absolute/path/to/split.json \
bash scripts/eval/eval_vanilla.sh
```

## One-Node 4×H200 Launcher

The additive H200 launcher leaves the original YAML and launchers unchanged.
It reserves physical GPU 0 for the existing Qwen3-14B user simulator and makes
GPUs 1, 2 and 3 visible to the Qwen3-8B veRL trainer.  The rollout uses TP=1,
so the three training GPUs form three rollout data-parallel ranks instead of an
invalid or inefficient TP=3 model shard.

```bash
cd agentic-grpo-longhorizon

# Inspect every resolved override without loading models or creating a run.
AGENTIC_RL_DRY_RUN=1 \
bash scripts/train/grpo/run_vanilla_h200_4gpu.sh

# Start the isolated Vanilla GRPO experiment.
bash scripts/train/grpo/run_vanilla_h200_4gpu.sh
```

The H200 defaults increase prompt/response limits to `12288/16384` and the
policy/user-simulator contexts to `32768`, while preserving reward, advantage,
rollout group size, learning rate and training horizon.  Results are stored at
`experiments/h200_4gpu/vanilla_grpo/<run-tag>/`, including checkpoints,
`train.log`, `user_simulator.log`, `launch_manifest.env`, the resolved command
and local W&B files.  Online metrics are sent to
[`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon).

Useful non-destructive overrides:

```bash
AGENTIC_RL_RUN_TAG=vanilla_seed1 \
AGENTIC_RL_VANILLA_DATA_ROOT=/absolute/path/to/experiments/vanilla \
bash scripts/train/grpo/run_vanilla_h200_4gpu.sh
```

Reusing the same run tag enables veRL/W&B resume behavior.  To prevent unknown
GPU overlap, an existing service on port 8001 is rejected by default.  Set
`AGENTIC_RL_REUSE_USER_SIMULATOR=1` only after verifying that service occupies
GPU 0 alone; otherwise the launcher starts its own simulator on GPU 0 and stops
only that owned process when training exits.

## Quick Start

The model paths, GPU topology and user-simulator service are configured by the
existing training scripts.  They are intentionally not duplicated here.

```bash
bash setup.sh
conda activate agentrl
cd agentic-grpo-longhorizon/scripts/train/grpo

bash run_vanilla.sh
bash run_exp1_turn_discount.sh
bash run_exp2_lata.sh
bash run_exp3_prm_lite.sh
bash run_exp4_prm_lite_lata.sh
```

## Important Files

| File | Role |
|---|---|
| `agentic-grpo-longhorizon/src/envs/tau_bench_interaction.py` | terminal reward and PRM-Lite rule aggregation |
| `verl/verl/trainer/ppo/core_algos.py` | GRPO, turn-discount and legacy LATA estimators |
| `agentic-grpo-longhorizon/configs/train/grpo/` | audited training contracts |
| `agentic-grpo-longhorizon/src/evaluation/pass_k_eval.py` | task execution and corrected pass metrics |
| `agentic-grpo-longhorizon/scripts/eval/eval_checkpoint_series.sh` | shared safe checkpoint evaluator |
| `agentic-grpo-longhorizon/src/evaluation/tests/test_pass_k_metrics.py` | metric regression contract |

## Recommended Experimental Order

1. Reproduce vanilla with the fixed split and record per-task reward variance.
2. Use PRM-Lite only as a diagnostic shaper; log outcome, process score and their
   conflicts separately.
3. Compare semantic turn-aware branches against the same vanilla checkpoints.
4. Report at least three seeds or bootstrap task-level confidence intervals.
5. Gate any learned PRM or LLM judge on held-out calibration, tool-error
   sensitivity, counterfactual ranking and proxy-hacking tests.

The most informative diagnostics are group saturation rate, task success,
tool-call error rate, valid-write precision, trajectory length, reward/outcome
conflict, and gradient contribution by assistant turn.  Training reward alone is
not a reliable model-selection metric.

## References

- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- [veRL](https://github.com/volcengine/verl)
- [Turn-PPO](https://arxiv.org/abs/2512.17008)
- [TRACE](https://arxiv.org/abs/2607.13988)
- [UserRL](https://arxiv.org/abs/2509.19736)

## License

See [LICENSE](LICENSE).
