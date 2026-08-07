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

## Project Highlights

This branch makes three observable improvements over naive outcome-only GRPO on τ-bench airline:

| Feature | What it solves | Where to see it in W&B |
|---------|----------------|------------------------|
| **Hybrid Advantage (TRACE)** |终局 0/1 无法区分中间动作质量 | `trace/prefix_log_prob_mean`, `trace/scored_prefixes`, `critic/advantages/mean` |
| **PRM-Lite v5** | 群组饱和（score 全 0/1）导致梯度消失 | `reward/reward_mode=prm_lite`, `process_score` in trajectory extra |
| **Population std GRPO** | veRL 默认 Bessel-corrected std 偏移 outcome scale | `critic/advantages/std` should match population not Bessel |
| **Turn-boundary masking** | token-level advantage 不小心扩散到 observation | `num_turns/mean`, `response_length/mean` |
| **Numerical guards** | fully-masked row → NaN → training divergence | `critic/advantages/min` never NaN, `response/aborted_ratio` < 5 % |
| **K=3 TD backup** | 延迟证据（后几步才显现）无法回传 | `critic/advantages/max - critic/advantages/mean` widens then narrows |

> **Baseline for comparison:** The W&B run
> [`b3wg4ezc`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/b3wg4ezc)
> shows vanilla GRPO on the same τ-bench airline split collapsing after ~150 steps
> (response length → 0, advantages → 0). The current branch is expected to sustain
> non-zero advantage variance throughout training.

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
[`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon);
new runs appear alongside the reference run
[`1ictyjgd`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/1ictyjgd)
and the legacy baseline
[`b3wg4ezc`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/b3wg4ezc).
Set `AGENTIC_RL_RUN_TAG=hybrid_seed1` to create a stable resumable identity.
An occupied port 8001 is rejected by default to prevent unknown GPU overlap;
explicit reuse requires `AGENTIC_RL_REUSE_USER_SIMULATOR=1` after confirming
that the existing service uses only GPU 0.

## Training

```bash
cd agentic-grpo-longhorizon
bash scripts/train/grpo/run_hybrid_advantage.sh
```

Primary configuration: `configs/train/grpo/hybrid_advantage.yaml`.

### Observability Dashboard — What to Watch in W&B

Every experiment should be tracked at
[`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon).
The current baseline comparison run is [`1ictyjgd`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/1ictyjgd).

**Panel layout recommendation (W&B wandb):**

```
row 1: critic/advantages/mean | critic/advantages/std | critic/advantages/min
row 2: response_length/mean   | response_length/clip_ratio | response/aborted_ratio
row 3: trace/prefix_log_prob_mean | trace/scored_prefixes | critic/score/mean
row 4: policy loss | actor grad norm | ref kl
row 5: critic/vf_explained_var (if use_critic=True)
```

**Metric dictionary — what each panel tells you:**

#### Training dynamics (must always be healthy)

| W&B metric | Healthy range | What bad looks like |
|---|---|---|
| `critic/advantages/mean` | stable ~0; not trending to ±large | drifts to +0.5+ → all rollouts converging to same score |
| `critic/advantages/std` | stable > 0.05; not trending to 0 | drops to < 0.01 → group saturation (vanilla collapse) |
| `critic/advantages/min` | not NaN, not all-zero | NaN → fully-masked row leak; all-zero → collapsed |
| `response_length/mean` | non-zero, not exploding (> 3× baseline) | drops to < 20 tokens → collapse |
| `response_length/clip_ratio` | < 0.5 | > 0.8 → hitting max too often; increase limit |
| `response/aborted_ratio` | < 5 % | > 15 % → rollout workers failing; check vLLM |
| `actor/grad_norm` | 0.5–5.0 | > 20 → exploding; < 0.01 → dead |
| `actor/policy_loss` | not NaN, not inf | NaN → check for inf in advantages |

#### TRACE-specific (only in `adv_estimator=grpo_hybrid`)

| W&B metric | Healthy range | What bad looks like |
|---|---|---|
| `trace/scored_prefixes` | > 0, increasing with turns | = 0 → prefix scoring batch all-empty; check span construction |
| `trace/prefix_log_prob_mean` | negative, trending toward 0 | trending to +1e-3+ → ref model assigning positive log-p to target → check tokenizer |
| `critic/advantages/max - critic/advantages/mean` | widens then narrows | always flat → TD credit is zero (K=0 or all prefixes same score) |
| `num_turns/mean` | 2–10 | < 1 → no multi-turn; no TRACE turns |
| `num_turns/max` | not growing unbounded | > 50 → runaway loops |

#### GRPO / advantage normalization

| W&B metric | Healthy range | What bad looks like |
|---|---|---|
| `ref/kl` | 0.01–0.3 per step | > 1.0 → ref policy diverging; reduce lr |
| `critic/score/mean` | between 0 and 1 | drifting to 0 or 1 → group saturated |
| `critic/score/all_one_frac` | < 0.3 | > 0.7 → nearly all rollouts succeed; group cannot discriminate |
| `critic/score/all_zero_frac` | < 0.3 | > 0.7 → nearly all rollouts fail; no positive signal |

#### PRM-Lite process metrics (if `reward_mode=prm_lite`)

The trainer emits `reward_extra_infos` keyed by the interaction class. In
`tau_bench_interaction.py`, `calculate_score` returns a dict with `outcome_score`,
`process_score`. These are per-trajectory scalars stored in `reward_extra_infos`:

| Diagnostic | How to get it | Healthy range |
|---|---|---|
| `process_score` distribution | log `reward_extra_infos["process_score"]` per trajectory | mean ≈ +0.1 to +0.3 on success; ≈ 0 on failure |
| `process_score == 0` fraction | fraction of trajectories with zero process score | < 30 % on success rollouts |
| `process_score` × `outcome_score` correlation | high positive → process signals real quality | if near 0 → PRM-Lite not discriminating |

> **Note on metric logging:** The per-trajectory `process_score` from `reward_extra_infos`
> is not currently aggregated by the trainer into a dedicated W&B panel. To see it in
> W&B during training, add a custom logging hook in `ray_trainer.py` that reduces
> `reward_extra_infos["process_score"]` to a mean/min/max and logs it under
> `prm_lite/process_score/mean`, etc. See the `W5 conditional PRM` comment in
> `tau_bench_interaction.py:694`.

#### Signal-of-progress checklist (per 50-step window)

Run this mental check at every evaluation checkpoint:

```
✅ advantages not all the same value
✅ response_length not collapsing (not < 20 tokens mean)
✅ trace/scored_prefixes > 0 (TRACE scoring is running)
✅ trace/prefix_log_prob_mean is negative (ref model cannot trivially predict gold)
✅ ref/kl < 1.0 (actor not diverging from ref)
✅ response/aborted_ratio < 15 % (rollouts completing)
✅ critic/score/all_one_frac < 0.7 AND all_zero_frac < 0.7 (group not saturated)
```

If any ❌, see the "Failure Modes" table below.

### Failure Modes — Diagnosis and Interventions

| Symptom (W&B) | Likely cause | Intervention |
|---|---|---|
| `critic/advantages/std → 0` | Group saturation: all rollouts same outcome | Switch to PRM-Lite reward mode |
| `response_length/mean → 0` | Vanilla collapse: model learned to stop responding | Use LATA or turn-discounted weight in adv config |
| `trace/scored_prefixes = 0` | Span construction failed or rollout truncated before first assistant turn | Check `build_trace_layout` in `turn_reward_utils.py` |
| `trace/prefix_log_prob_mean → 0` | Ref model trivially predicting gold; no information | Target too short or already in context; shorten or verify prefix isolation |
| `critic/advantages/min = NaN` | Fully-masked row leaked through (pre-fix batch norm) | Verify `compute_trace_group_outcome_advantage` skips zero-mask rows |
| `ref/kl > 1.0` | Learning rate too high or batch too small | Reduce `actor.lr`; increase `rollout.n` |
| `response_length/clip_ratio > 0.9` | Tokens hitting max limit; trajectories cut short | Increase `response_length` or reduce turn budget |
| `critic/score/all_one_frac > 0.9` | Group is trivially easy; no discrimination signal | Increase group size or switch to unseen-task-only evaluation |
| `actor/grad_norm > 20` | Exploding gradient | Add gradient clipping; check for inf in advantages |

### Offline vs Online: When Each Signal is Reliable

| Signal | Source | Reliable when |
|--------|--------|---------------|
| `critic/score/mean` | τ-bench outcome 0/1, averaged per group | always (verifier ground truth) |
| `trace/prefix_log_prob_mean` | frozen ref on gold target | offline calibration shows success/fail AUC > 0.7 |
| `trace/scored_prefixes` | count of non-empty prefix batches | always (0 = bug) |
| `critic/advantages/std` | from population std of scores | not during saturation |
| `process_score` | PRM-Lite v5 rules | verified against human annotation |

> **Golden rule:** If `critic/score/mean` and `response_length/mean` are both healthy,
> the training is proceeding correctly. If either collapses, no amount of TRACE
> telemetry will compensate — fix the collapse first.

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

**Comparing with W&B reference runs:** Evaluation results are logged to the same
project at [`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon).
The reference training run is [`1ictyjgd`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/1ictyjgd).
When evaluating new checkpoints, use the same split and `n=4` samples-per-task
to ensure metric comparability across runs. The evaluator will emit `eval/pass@1`,
`eval/pass@4`, `eval/any_success`, `eval/pass^k` — compare these to the same
metrics from previous runs at matching step counts.

| 📄 Document | 📝 Content |
|----------|---------|
| [`docs/tech-report-trace.html`](docs/tech-report-trace.html) | **Main theory doc**: paper ↔ code bidirectional audit, 5-min glossary, numerical walkthrough, PRM-Lite v5 signal analysis, edge-case guards, and 5-gate acceptance checklist |
| [`docs/ablation/ablation_diagnosis_report.md`](docs/ablation/ablation_diagnosis_report.md) | **Main report**: training curves, eval data, mechanism analysis, hypothesis validation |
| [`docs/ablation/ablation_plan.md`](docs/ablation/ablation_plan.md) | Experiment design manual: code implementation, PRM-Lite rule set, hacking risk analysis |
| [`docs/vanilla_grpo/vanilla_grpo_diagnosis.md`](docs/vanilla_grpo/vanilla_grpo_diagnosis.md) | Vanilla GRPO collapse diagnosis: three root causes, five checkpoints analysis |
| [`agentic-grpo-longhorizon/scripts/validate/numerical_walkthrough.py`](../agentic-grpo-longhorizon/scripts/validate/numerical_walkthrough.py) | Pure-Python 4-turn numerical trace — run with `python3` to reproduce every number in the tech report |
| [`agentic-grpo-longhorizon/src/envs/tests/test_prm_lite_v5.py`](../agentic-grpo-longhorizon/src/envs/tests/test_prm_lite_v5.py) | 16-test PRM-Lite v5 regression suite — run with `pytest` |
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

Before treating the adaptation as better than outcome-only GRPO, verify the following
against W&B run [`1ictyjgd`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon/runs/1ictyjgd):

### Training health gates (must all be ✅)

| Gate | W&B metric | Pass threshold |
|------|-----------|----------------|
| No collapse | `critic/advantages/std` | > 0.05 at step 50, > 0.03 at step 100 |
| No length death | `response_length/mean` | > 50 tokens at step 100 |
| TRACE running | `trace/scored_prefixes` | > 0 throughout training |
| Ref stable | `ref/kl` | < 0.5 per-step mean |
| No saturation | `critic/score/all_one_frac` and `all_zero_frac` | both < 0.7 |
| Rollouts completing | `response/aborted_ratio` | < 15 % throughout |

If any gate fails, see the Failure Modes table above.

### Evaluation gates (step 50 / 100 / 150 / 200 checkpoints)

| Gate | Method | Pass threshold |
|------|--------|---------------|
| Held-out terminal success improves | Unified evaluator on unseen 10 tasks | > baseline (vanilla or previous best) |
| Gain survives multiple seeds | ±1 seed re-run | within 2× baseline variance |
| Correct actions rank above wrong | Per-turn credit manual audit on 20 sampled trajectories | ≥ 80 % agreement with human label |
| Tool-error turns receive lower credit | `trace/prefix_log_prob_mean` on failed vs succeeded prefix | failed < succeeded |
| Credit does not correlate with length | Pearson(turn_credit, target_length) | < 0.3 |
| Outcome weight remains non-zero | `alpha_out * A_out` contribution to `A_t` | > 50 % of total advantage variance |
| Target absent from eval artifacts | Grep rollout logs | no occurrence of canonical JSON fields |

## References

- [TRACE](https://arxiv.org/abs/2607.13988)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [veRL](https://github.com/volcengine/verl)

## License

See [LICENSE](LICENSE).
