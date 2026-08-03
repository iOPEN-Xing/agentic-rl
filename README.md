# Hybrid Advantage: TRACE-Style Credit for τ-bench Airline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This branch implements a TRACE-style hybrid of group-relative terminal outcome
advantage and turn-local temporal-difference credit.  The implementation uses
real assistant/tool-cycle boundaries and a frozen reference model; it does not
use the earlier hand-written potential function.

The branch is an adaptation of
[TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents](https://arxiv.org/abs/2607.13988),
not a claim of byte-for-byte reproduction.  The most important adaptation is
that τ-bench provides required database actions and outputs rather than one
canonical natural-language final answer.

## Audit Verdict

The current training chain is logically consistent with TRACE Algorithm 1:

- state boundaries are recorded before assistant decisions;
- every state is scored against the same training-only target by a frozen
  reference policy with teacher forcing;
- state value is a log ratio of the remaining target-prediction gap;
- turn reward is normalized K-step TD credit plus a terminal outcome fill;
- terminal GRPO outcome advantage remains the dominant signal;
- turn credit is mapped only to the assistant span that produced that
  transition;
- the final-answer tail receives outcome advantage only.

The removed heuristic potential contained a terminal-success indicator and
could leak final outcome into nominally intermediate state value.  No such term
is present in the current implementation.

## Algorithm

For each rollout group belonging to task `g`, the terminal verifier produces
`R_i`.  The implementation uses TRACE's population standard deviation:

```text
A_out,i = (R_i - mean_g(R)) / sqrt(mean_g((R - mean_g(R))²))
```

For every state prefix `s_t`, the frozen reference model teacher-forces the same
target `y*` and calculates average target-token log probability `ell_t`.  With
`epsilon=0.1`:

```text
d_t = -ell_t + epsilon
V_t = log(d_0 / d_t)
delta_t = V_(t+1) - V_t
```

Positive `delta_t` means the tool interaction made the required target easier
for the frozen model to predict.  For the default `K=3`, `gamma=0.8`, the local
credit is a normalized discounted average:

```text
local_t = Σ_(j=0..K-1) gamma^j delta_(t+j) / Σ_(j=0..K-1) gamma^j
```

When a K-step window reaches the terminal transition, it also receives
`2 × gamma^(T-t) × A_out`.  The policy advantage for tokens in assistant turn
`t` is:

```text
A_token,t = 1.0 × A_out + 0.2 × A_trace,t
```

TRACE turn credit is intentionally not normalized across the rollout group:
normalizing it would change a trajectory-local progress estimate into a
relative ranking signal and can erase its scale semantics.

## End-to-End Data Flow

```text
τ-bench task
  │
  ├─ required database actions + required outputs
  │          │
  │          └─ deterministic canonical JSON target y* (training metadata)
  │
  └─ policy rollout ↔ user simulator ↔ airline tools
             │
             ├─ assistant spans [start_t, end_t)
             ├─ pre-assistant state boundaries b_t
             └─ terminal verifier outcome R
                         │
       ┌─────────────────┴──────────────────┐
       │                                    │
       ▼                                    ▼
group-relative outcome A_out       frozen-reference scoring batch
                                            │
                               prefix(s_t) + teacher-forced y*
                                            │
                                            ▼
                                ell_t → V_t → K-step TD credit
       │                                    │
       └─────────────────┬──────────────────┘
                         ▼
             hybrid assistant-token advantage
                         │
                         ▼
                  standard clipped actor loss
```

The target is never included in the actor prompt or rollout.  It appears only
in the separate reference-scoring batch.  The builder fails on missing targets,
empty tokenization and over-length prefix/target combinations instead of
silently truncating the value definition.

## τ-bench Adaptation

### Canonical target

`scripts/train/grpo/build_grpo_parquet.py` serializes a stable training target:

```json
{
  "required_actions": [
    {"name": "update_reservation_flights", "kwargs": {"...": "..."}}
  ],
  "required_outputs": ["..."]
}
```

This is a defensible surrogate for “what remains to be achieved” because it is
derived from the same task specification used by the τ-bench verifier.  It also
creates a different inductive bias from the paper's natural-language answer:
the value model may become especially sensitive to JSON surface form, action
ordering and long argument strings.  Therefore this branch should be reported
as **TRACE-style τ-bench adaptation**, not exact TRACE reproduction.

### Concrete airline case

Consider a task requiring: read reservation, identify the correct passenger,
change the outbound flight, preserve the return flight and report the result.

1. Before any tool call, the frozen scorer has low likelihood for the canonical
   action/output target, so `V_0=0` by construction.
2. A correct `get_reservation` reveals flight and passenger identifiers.  The
   target becomes easier to predict; `V_1>V_0`, yielding positive TD credit for
   the assistant turn that selected the read.
3. A redundant read may barely change target likelihood, yielding near-zero
   credit without requiring a hand-written redundancy penalty.
4. A write to the wrong reservation can make the target harder to predict,
   yielding negative local credit even before final failure.
5. Final verifier success still supplies the main group-relative advantage, so
   a locally plausible but globally wrong trajectory is not treated as success.

This is the main advantage over PRM-Lite: the target likelihood provides a
task-conditioned progress estimate, and the score remains located at the
assistant decision that caused the observation change.

## Paper-to-Code Correspondence

| TRACE concept | Branch implementation |
|---|---|
| state prefix at tool boundary | `tool_agent_loop.py` records `trace_state_boundaries` |
| action span receiving credit | `trace_turn_spans` over generated assistant tokens |
| frozen target scorer | `trace_reference.py` builds teacher-forced batches |
| log-ratio value | `compute_trace_log_ratio_values` |
| K-step TD + terminal fill | `compute_trace_turn_rewards` |
| outcome/turn mixture | `compute_grpo_hybrid_advantage` |
| τ-bench target | `build_grpo_parquet.py` canonical required-actions/outputs JSON |

The branch defaults match the paper's main configuration: outcome weight `1.0`,
turn weight `0.2`, `K=3`, `gamma=0.8`, gap offset `0.1`, terminal scale `2.0`
and scorer temperature `1.0`.

## Important Non-Equivalences and Risks

1. **Target semantics.** Canonical verifier JSON replaces the paper's gold
   answer.  Its usefulness must be validated, not assumed.
2. **Answer-tail weighting.** The paper reports an experimental final-answer
   token rule, but the public description does not define its exact reduction
   unambiguously.  This branch leaves final-answer tokens on outcome advantage
   only instead of silently inventing an interpretation.
3. **Reference bias.** Credit quality is bounded by the frozen model's ability
   to map a tool observation to the canonical target.  A weak or format-biased
   reference produces a weak value signal.
4. **Sequence cost.** A reference forward is required for every state prefix and
   target.  Cost grows with turns and target length.
5. **Batch padding limit.** Individually valid prefix/target pairs can still form
   an over-length padded batch.  Current code fails clearly; production code
   should bucket or micro-batch such pairs without changing their contents.
6. **Multi-tool assistant messages.** One assistant generation can contain
   multiple tool calls.  The default configuration limits parallel calls; if
   that changes, define whether the entire generation is one action or whether
   sub-call credit is required.
7. **No empirical result is claimed here.** The branch has algorithm and
   contract tests, not a completed multi-seed τ-bench comparison.

## One-Node 4×H200 Launcher

The H200 launcher places the existing Qwen3-14B user simulator on physical GPU
0 and the Qwen3-8B actor/reference/TRACE scorer and rollout workers on GPUs 1–3.
The policy rollout uses TP=1 with three data-parallel ranks.  No original YAML
or launcher is modified.

```bash
cd agentic-grpo-longhorizon

AGENTIC_RL_DRY_RUN=1 \
bash scripts/train/grpo/run_hybrid_advantage_h200_4gpu.sh

AGENTIC_RL_SPLIT_FILE=/absolute/path/to/split.json \
bash scripts/train/grpo/run_hybrid_advantage_h200_4gpu.sh
```

The policy prompt/response limits are `12288/16384`; policy, frozen-reference
scoring and user-simulator contexts are `32768`.  The launcher also raises
`algorithm.hybrid_advantage.max_scoring_length` to `32768`, while retaining the
TRACE weights, TD horizon, discount and target cap.  It builds canonical target
parquet files inside the run directory instead of overwriting shared data.

Results are isolated under
`experiments/h200_4gpu/hybrid_advantage/<run-tag>/`, including the run-specific
training data, checkpoints, logs, manifest, exact command and local W&B files.
Online records go to
[`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon).
Set `AGENTIC_RL_RUN_TAG=hybrid_seed1` to create a stable resumable identity.
An occupied port 8001 is rejected by default to prevent unknown GPU overlap;
explicit reuse requires `AGENTIC_RL_REUSE_USER_SIMULATOR=1` after confirming
that the existing service uses only GPU 0.

## Training

The launcher rebuilds method-specific parquet files because vanilla data has an
empty `reward_model.ground_truth` by design.

```bash
cd agentic-grpo-longhorizon
bash scripts/train/grpo/run_hybrid_advantage.sh
```

Primary configuration:
`configs/train/grpo/hybrid_advantage.yaml`.

Useful diagnostics include `V_t`, adjacent `delta_t`, TD reward scale,
turn/outcome advantage ratio, target length, scoring failures, group saturation
and correlation between summed turn credit and terminal success.  Always inspect
successful, failed and judge/verifier-conflict cases separately.

## Unified Evaluation

```bash
cd agentic-grpo-longhorizon
bash scripts/eval/eval_hybrid_advantage.sh
```

The default contract evaluates steps `50 100 150 200` with the same split,
Qwen3-14B user simulator, policy sampling parameters and turn budget used by the
other branches.  Overrides:

```bash
AGENTIC_RL_EVAL_STEPS="100 200" \
AGENTIC_RL_EVAL_SPLIT_FILE=/absolute/path/to/split.json \
bash scripts/eval/eval_hybrid_advantage.sh
```

The evaluator reports `any_success`, standard at-least-one `pass@k`, and
all-success `pass^k` as distinct metrics.

| 📄 Document | 📝 Content |
|----------|---------|
| [`docs/ablation/ablation_diagnosis_report.md`](docs/ablation/ablation_diagnosis_report.md) | **Main report**: training curves, eval data, mechanism analysis, hypothesis validation |
| [`docs/ablation/ablation_plan.md`](docs/ablation/ablation_plan.md) | Experiment design manual: code implementation, PRM-Lite rule set, hacking risk analysis |
| [`agentic-grpo-longhorizon/docs/optimization/hybrid_advantage.md`](agentic-grpo-longhorizon/docs/optimization/hybrid_advantage.md) | TRACE-style Hybrid Advantage equations, frozen-reference scoring contract, tool-error policy, configuration, and CPU validation |
| [`docs/vanilla_grpo/vanilla_grpo_diagnosis.md`](docs/vanilla_grpo/vanilla_grpo_diagnosis.md) | Vanilla GRPO collapse diagnosis: three root causes, five checkpoints analysis |
| [`../agentic-grpo-longhorizon-blog.md`](../agentic-grpo-longhorizon-blog.md) | 🆕 Technical blog: from training collapse to stable convergence (PRM-Lite + LATA) |

## Verification

The relevant CPU/static contracts are:

```bash
pytest -q verl/tests/trainer/ppo/test_hybrid_advantage.py
pytest -q verl/tests/trainer/ppo/test_trace_reference.py
pytest -q verl/tests/experimental/agent_loop/test_turn_reward_utils.py
pytest -q agentic-grpo-longhorizon/src/training/tests/test_trace_targets.py
pytest -q agentic-grpo-longhorizon/src/evaluation/tests/test_pass_k_metrics.py
```

They cover value math, K-step/terminal credit, span mapping, frozen-reference
batch construction, target non-leakage and metric definitions.  These tests do
not replace an end-to-end rollout or calibration experiment.

## Recommended Acceptance Gates

Before treating the adaptation as better than outcome-only GRPO:

- held-out terminal success improves under the unified evaluator;
- gain survives task bootstrap intervals and multiple seeds;
- correct intermediate actions rank above counterfactual wrong actions;
- tool-error turns receive lower average credit than matched valid turns;
- local credit does not simply correlate with trajectory length or target
  token count;
- outcome-only weight remains nonzero and conflict cases are manually audited;
- training-only targets are absent from evaluation prompts and artifacts.

## References

- [TRACE](https://arxiv.org/abs/2607.13988)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [veRL](https://github.com/volcengine/verl)

## License

See [LICENSE](LICENSE).
