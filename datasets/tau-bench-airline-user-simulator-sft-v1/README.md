# tau-bench Airline User-Simulator SFT v1

This release contains audited conversational SFT data for the `tau-bench`
airline user simulator. It is a `tau-bench` release, not a Pi-Bench dataset.

| File | Rows | Purpose | SHA-256 |
| --- | ---: | --- | --- |
| `train.jsonl` | 1,595 | Fine-tuning split | `c1dd6fa61cc141bc6d596da38e2e7c0e44c11389ae409332e3ad88009ef50379` |
| `eval.jsonl` | 164 | Held-out evaluation split | `89be5d093484591ad804878d7e00b4575d4fcb1ed70a237893601d6fb3e86492` |

Each JSONL line has exactly one field:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Role meanings follow the runtime contract, which is intentionally role-flipped
relative to a normal assistant SFT dataset:

- `system`: the exact `LLMUserSimulationEnv` system prompt;
- `user`: text the policy agent made visible to the simulated customer;
- final `assistant`: the next simulated-customer reply, or the exact terminal
  marker `###STOP###`.

Earlier `assistant` messages are historical simulated-customer context. Compute
loss only on the final `assistant` message of each example. Do not normalize,
strip, or replace `###STOP###`: the tau-bench runtime uses that exact marker to
end an interaction.

## Load

```python
from datasets import load_dataset

data = load_dataset(
    "json",
    data_files={
        "train": "datasets/tau-bench-airline-user-simulator-sft-v1/train.jsonl",
        "eval": "datasets/tau-bench-airline-user-simulator-sft-v1/eval.jsonl",
    },
)
```

Render `messages` with the same chat template used by the deployed user
simulator, with thinking disabled when using the repository's Qwen3 setup.
For this release, keep the `eval` split immutable and apply
`last_assistant` loss masking rather than training every historical assistant
turn in the prefix.

## Split And Cleaning

The evaluation split is the natural held-out `trial07` population. It has no
identical complete example or model-input prefix in common with the released
training split.

`train.jsonl` is the user-selected cleaned version of the original 1,635-row
TRL training artifact. It omits 40 continuation examples and introduces no new
examples or terminal targets. The remaining 504 terminal targets retain the
source pipeline's threefold terminal oversampling; the training split therefore
contains 1,091 continuation and 504 terminal examples. The evaluation split is
natural: 139 continuation and 25 terminal examples.

## Provenance And Scope

The source pipeline used the tau-bench airline historical trajectories, the
40-task seen split, audited teacher targets, and a separate `trial07`
evaluation population. The associated offline audit verified runtime prompt
alignment, role alternation, no tool-role messages, no live-prefix stop marker,
no holdout tasks in SFT, and no secret-shaped values. It does not establish
online rollout quality or remove the need for frozen-policy A/B evaluation.

This package deliberately contains only the two release JSONL files and their
metadata. It does not publish the benchmark holdout cases, tool traces, source
databases, raw teacher responses, or model checkpoints.

tau-bench is MIT licensed. See [NOTICE.md](NOTICE.md) and the retained upstream
[LICENSE.tau-bench](LICENSE.tau-bench).
