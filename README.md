# User-Simulator Judge for τ-bench Airline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This branch adds an LLM judge to every user/agent interaction cycle.  The judge
observes the task goal, recent dialogue, tool parameters, tool observations and
tool-error flags, then returns structured progress, tool-correctness and
communication scores.

The safe default is deliberately conservative:

> the judge is collected and logged, but does not change the GRPO reward unless
> `algorithm.judge_turn_reward.enabled=true` is explicitly set after held-out
> calibration.

The branch is useful as a judge-calibration and reward-shaping experiment.  It
is **not a complete implementation of UserRL** and should not be described as
turn-local UserRL credit assignment.

## Audit Verdict

The current implementation is internally consistent for its declared scope:

- the terminal score remains the rule-based τ-bench verifier outcome;
- the judge sees actual tool observations and error status instead of trusting
  the assistant's textual claim;
- strict JSON parsing, finite numeric checks and `[0,1]` clipping protect the
  interface;
- a raw score of `0.5` maps to zero, so judge availability does not create an
  automatic positive reward;
- invalid or unavailable judge output becomes a missing event, not a zero-score
  penalty;
- optional shaping is averaged over valid events and bounded to at most `0.1`
  per trajectory, preserving terminal-outcome dominance;
- valid rate, judge mean and cross-trajectory judge/outcome correlation are
  exposed as training diagnostics.

However, standard GRPO later sums token rewards into one trajectory score and
broadcasts one group-relative advantage.  Therefore enabling the optional
judge term still does **not** preserve which assistant turn earned each score.

## Signal Path

```text
task goal + recent dialogue
          │
          ├─ current cycle tool calls
          ├─ parameters
          ├─ tool observations
          └─ is_error
          │
          ▼
Qwen3-14B judge, temperature 0
          │
          ▼
{task_progress, tool_correctness, communication}
          │
          ▼
raw judge score = 0.45 progress + 0.40 correctness + 0.15 communication
          │
          ▼
center around 0.5 → turn event in [-1, 1]
          │
          ├─ default: diagnostics only
          │
          └─ optional: mean valid events × weight (weight ≤ 0.1)
                                   │
τ-bench terminal outcome ──────────┤
                                   ▼
                         one trajectory reward
                                   │
                                   ▼
                     standard outcome-only GRPO advantage
```

With the shipped training config, `interaction.judge_enabled=true`, so judge
events are produced, while `algorithm.judge_turn_reward.enabled=false`, so the
actor sees exactly the binary terminal reward.  This separation permits
calibration on the same rollout distribution before the proxy is allowed into
the objective.

## Score Semantics

The judge returns three fields:

| Field | Weight | Evidence it should use |
|---|---:|---|
| `task_progress` | 0.45 | whether the cycle moves toward the stated task goal |
| `tool_correctness` | 0.40 | actual tool parameters, observations and errors |
| `communication` | 0.15 | grounded, clear and policy-compatible response |

If a cycle contains no tool action, the prompt instructs the judge to set tool
correctness to `0.5`, avoiding invented tool evidence.  The weighted raw score
is centered by `JudgeSignalPolicy`:

```text
signal = (score - 0.5) / 0.5
```

for the default neutral point, producing `[-1,1]`.  If shaping is enabled and
there are `m` valid judge events:

```text
R_shaped = R_terminal + w × (1/m) × Σ signal_t,     0 ≤ w ≤ 0.1
```

The implementation stores contributions at event positions before GRPO sums
them.  That provenance is helpful for logging and future algorithms, but the
current estimator reduces them to the scalar above.

## Concrete Airline Case

Suppose a user asks to change one passenger's outbound flight while preserving
the return leg.

1. The policy reads the reservation with the correct identifier.  The tool
   returns valid passenger and flight data.  A calibrated judge should assign
   high correctness and positive progress.
2. The policy attempts a write with a placeholder passenger ID.  The tool
   returns an error.  Because the judge receives `is_error=true` and the actual
   observation, it should score correctness low even if the assistant says the
   change succeeded.
3. The policy recovers by reading availability and issuing the valid update.
   Progress can rise again.
4. The τ-bench verifier remains decisive: if the wrong flight was changed, the
   terminal outcome is zero regardless of a fluent final message.

This example highlights both the value and limitation of an LLM judge.  It can
distinguish grounded recovery from blind repetition more flexibly than fixed
PRM-Lite rules, but a single scalar trajectory average cannot tell the actor
that only step 2 was wrong.

## Relationship to UserRL

[UserRL](https://arxiv.org/abs/2509.19736) jointly improves the user simulator
and assistant, and its public implementation includes multi-turn credit methods
such as Equalized, R2G and EM turn-credit variants.  This branch differs:

| Dimension | This branch | Full UserRL-style system |
|---|---|---|
| user simulator | frozen external Qwen3-14B service | user policy is also optimized |
| judge | same external endpoint, separate prompt | reward/evaluation components vary by implementation |
| terminal task truth | τ-bench rule verifier | environment-dependent |
| judge event location | recorded | used by multi-turn estimator |
| advantage | standard scalar GRPO | turn-aware aggregation/credit variants |
| shipped behavior | diagnostic only | learning signal is active |

Accordingly, the accurate name is **outcome-only GRPO with user-simulator judge
diagnostics and optional bounded trajectory shaping**.

## User Simulator and Judge Determine the Ceiling

The policy does not learn in isolation.  The user simulator controls ambiguity,
clarification, corrections, termination and distribution of dialogue states.
An overly cooperative simulator leaks the desired answer; an inconsistent one
adds reward noise; a narrow one encourages overfitting to phrasing.

The judge controls the quality of dense feedback.  A useful judge must be:

- grounded in observations, not assistant self-report;
- calibrated across successful and failed trajectories;
- sensitive to wrong entities, write arguments and tool errors;
- stable to harmless paraphrases and message formatting;
- robust to prompt injection inside task text or tool output;
- sufficiently independent from the policy and simulator to avoid shared bias.

This branch currently uses the same Qwen3-14B endpoint for user simulation and
judging.  That is economical but creates correlated errors: the simulator may
generate a dialogue pattern that the same model family systematically rates
high.  A serious experiment should compare an independent judge model or a
rule/verifier ensemble.

## Calibration Before Enabling Shaping

At minimum, build a held-out set of interaction cycles with labels for valid
progress, correct tool use, harmful write and grounded communication.  Measure:

- JSON/valid response rate and latency;
- AUROC or pairwise accuracy for good versus bad turns;
- expected calibration error or reliability buckets;
- judge/outcome correlation across trajectories;
- false-positive rate on fluent but tool-invalid turns;
- score invariance under paraphrase;
- counterfactual sensitivity when one entity or tool argument is changed;
- conflict rate where judge is positive but terminal verifier fails.

Correlation alone is insufficient.  A judge can correlate with outcome by
rewarding trajectory length or fluent final answers while missing the causal
tool error.  Inspect matched counterfactuals and conflict cases.

Recommended activation gates:

1. valid rate is stable and high on held-out tasks;
2. tool-error counterfactual ranking is reliable;
3. judge-positive/verifier-failure cases have been manually categorized;
4. no strong length, verbosity or self-report shortcut is found;
5. enable `turn_weight=0.02` first, then ablate up to `0.05`; keep the hard
   `0.1` ceiling;
6. retain an outcome-only control under the identical evaluator.

## One-Node 4×H200 Launcher

The dedicated experiment launcher uses physical GPU 0 for the shared Qwen3-14B
user-simulator/Judge endpoint and GPUs 1–3 for the Qwen3-8B veRL trainer.  The
policy rollout uses TP=1 and three data-parallel ranks.  Existing YAML remains
unchanged.

```bash
cd agentic-grpo-longhorizon

AGENTIC_RL_DRY_RUN=1 \
bash scripts/train/grpo/run_user_simulator_judge_h200_4gpu.sh

AGENTIC_RL_VANILLA_DATA_ROOT=/absolute/path/to/experiments/vanilla \
bash scripts/train/grpo/run_user_simulator_judge_h200_4gpu.sh
```

Unlike the diagnostic-only YAML default, this method-specific H200 launcher
sets `algorithm.judge_turn_reward.enabled=true` with the existing bounded
weight `0.05`, so it runs a distinct Judge-shaping experiment.  To collect
diagnostics without changing the policy reward, set:

```bash
AGENTIC_RL_ENABLE_JUDGE_REWARD=false \
bash scripts/train/grpo/run_user_simulator_judge_h200_4gpu.sh
```

Prompt/response limits become `12288/16384`, and policy/user contexts become
`32768`.  Results are isolated under
`experiments/h200_4gpu/user_simulator_judge/<run-tag>/`, including checkpoints,
trainer and simulator logs, launch metadata, exact command and W&B artifacts.
Online records go to
[`jiezhengxing-aaaa/agentic-grpo-longhorizon`](https://wandb.ai/jiezhengxing-aaaa/agentic-grpo-longhorizon).

Set `AGENTIC_RL_RUN_TAG=user_judge_seed1` for a stable resumable run identity.
The launcher rejects an occupied port 8001 by default because it cannot verify
the service's physical GPU.  Set `AGENTIC_RL_REUSE_USER_SIMULATOR=1` only after
confirming it uses GPU 0 alone.  It never kills a reused service; if it starts
the service itself, it cleans up only that owned process.

## Training

```bash
cd agentic-grpo-longhorizon
bash scripts/train/grpo/run_user_simulator_judge.sh
```

Default diagnostic-only contract:

```yaml
algorithm:
  adv_estimator: grpo
  judge_turn_reward:
    enabled: false
    turn_weight: 0.05
```

Only after calibration should `enabled` be changed to `true`.  Doing so creates
bounded trajectory shaping, not turn-aware advantages.

## Unified Evaluation

```bash
cd agentic-grpo-longhorizon
bash scripts/eval/eval_user_simulator_judge.sh
```

The launcher evaluates steps `50 100 150 200` under the same task split,
Qwen3-14B user simulator, sampling settings and turn budget as the other
branches.  It reports standard at-least-one `pass@k`, all-success `pass^k` and
`any_success` separately.

```bash
AGENTIC_RL_EVAL_STEPS="100 200" \
AGENTIC_RL_EVAL_SPLIT_FILE=/absolute/path/to/split.json \
bash scripts/eval/eval_user_simulator_judge.sh
```

## Important Files

| File | Role |
|---|---|
| `agentic-grpo-longhorizon/src/envs/user_simulator_judge.py` | prompt, parsing and structured scoring |
| `agentic-grpo-longhorizon/src/envs/reward_fusion.py` | neutral centering and consistency helper |
| `agentic-grpo-longhorizon/src/envs/tau_bench_interaction.py` | per-cycle judge invocation and terminal metadata |
| `verl/verl/trainer/ppo/ray_trainer.py` | diagnostics and optional bounded fusion |
| `agentic-grpo-longhorizon/configs/interaction_config/tau_bench_airline_judge.yaml` | simulator/judge service contract |
| `agentic-grpo-longhorizon/configs/train/grpo/user_simulator_judge.yaml` | training contract |
| `agentic-grpo-longhorizon/scripts/validate/reward_consistency.py` | offline outcome/judge consistency check |

## Verification

```bash
pytest -q agentic-grpo-longhorizon/src/envs/tests/test_user_simulator_judge.py
pytest -q verl/tests/trainer/ppo/test_judge_turn_reward_fusion.py
pytest -q agentic-grpo-longhorizon/src/evaluation/tests/test_pass_k_metrics.py
```

These tests cover structured parsing, neutral centering, error-aware context,
unavailable-judge fallback, bounded fusion, outcome dominance, cross-trajectory
diagnostics and evaluation metrics.  They do not establish judge calibration.

## References

- [UserRL](https://arxiv.org/abs/2509.19736)
- [Official UserRL implementation](https://github.com/SalesforceAIResearch/UserRL)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [veRL](https://github.com/volcengine/verl)

## License

See [LICENSE](LICENSE).
