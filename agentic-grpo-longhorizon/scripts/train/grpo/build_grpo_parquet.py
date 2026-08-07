"""
Build GRPO training/validation parquet files.

Usage:
    python scripts/train/grpo/build_grpo_parquet.py \
        --seen-task-ids-from experiments/sft_collect_airline/split.json \
        --output-train experiments/vanilla/train.parquet \
        --output-val experiments/vanilla/val.parquet

Design: patch v2 §3.4
- Each row = one task, rollout.n=4 expands at runtime by veRL
- prompt column: only system message (date grounding), user msg from Interaction
- extra_info: index, task_id, split, interaction_kwargs
- reward_model.ground_truth: training-only canonical TRACE target
- No traj_uid column (veRL repeat mechanism makes it non-unique)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
BUNDLE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BUNDLE_ROOT / "tau-bench"))

from tau_bench.envs.airline.tasks_test import TASKS
from tau_bench.types import RESPOND_ACTION_NAME

SYSTEM_PROMPT = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, "
    "always assume they refer to 2024. "
    "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
)

INTERACTION_NAME = "tau_bench_airline"
# Default upper bound mirrors the upstream τ-bench airline test split size.
# The actual task count is asserted against `len(TASKS)` at the start of main()
# so a future split growth surfaces as a clear error rather than silently
# dropping the unseen tasks at the end.
_NUM_AIRLINE_TASKS_DEFAULT = 50


def serialize_tau_bench_trace_target(task) -> str:
    """Serialize the verifier's stable action/output target for TRACE scoring.

    The hidden user-simulator instruction is intentionally excluded: it
    contains behavioral prose that is not checked by the τ-bench verifier.
    Dynamic post-booking identifiers are also absent because τ-bench ground
    truth specifies the required database mutation before such IDs exist.
    """
    required_actions = [
        {
            "name": action.name,
            "kwargs": action.kwargs,
        }
        for action in task.actions
        if action.name != RESPOND_ACTION_NAME
    ]
    payload = {
        "version": "tau_bench_airline_actions_outputs_v1",
        "required_actions": required_actions,
        "required_outputs": list(task.outputs),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_rows(task_ids: list[int], split: str) -> list[dict]:
    rows = []
    for idx, tid in enumerate(task_ids):
        if tid < 0 or tid >= len(TASKS):
            raise ValueError(f"Unknown airline task id {tid}; expected 0..{len(TASKS) - 1}")
        trace_target = serialize_tau_bench_trace_target(TASKS[tid])
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
            "extra_info": {
                "index": idx,
                "task_id": tid,
                "split": split,
                "interaction_kwargs": {
                    "name": INTERACTION_NAME,
                    "task_id": tid,
                },
            },
            "data_source": INTERACTION_NAME,
            # This field is consumed only by the frozen reference scorer. It
            # is never inserted into the actor prompt or user simulator.
            "reward_model": {"ground_truth": trace_target},
            "ability": INTERACTION_NAME,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build GRPO parquet datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seen-task-ids", type=str, help="Comma-separated seen task IDs")
    group.add_argument("--seen-task-ids-from", type=str, help="Path to metadata.json with seen_task_ids")
    parser.add_argument("--output-train", default="experiments/vanilla/train.parquet")
    parser.add_argument("--output-val", default="experiments/vanilla/val.parquet")
    parser.add_argument("--num-total-tasks", type=int, default=_NUM_AIRLINE_TASKS_DEFAULT)
    args = parser.parse_args()

    # Fail loud: --num-total-tasks must not exceed the actual τ-bench task set.
    # Silently truncating would silently drop unseen tasks and skew pass@k.
    if args.num_total_tasks > len(TASKS):
        raise ValueError(
            f"--num-total-tasks={args.num_total_tasks} exceeds the τ-bench airline "
            f"task count ({len(TASKS)}). Adjust the default or pass a smaller value."
        )

    if args.seen_task_ids:
        seen_ids = [int(x.strip()) for x in args.seen_task_ids.split(",")]
    else:
        meta_path = Path(args.seen_task_ids_from)
        with open(meta_path) as f:
            meta = json.load(f)
        if "seen_task_ids" in meta:
            seen_ids = meta["seen_task_ids"]
        elif "covered_task_ids" in meta:
            seen_ids = meta["covered_task_ids"]
        else:
            # Fail loud: silently falling back to ``range(40)`` would mask a
            # metadata schema drift (e.g. split.json now lives under a
            # different key). Refuse to guess.
            raise ValueError(
                f"{meta_path} contains neither 'seen_task_ids' nor 'covered_task_ids'. "
                f"Found keys: {sorted(meta.keys())}. Refusing to silently fall back "
                f"to range(40); pass --seen-task-ids explicitly to override."
            )

    all_ids = list(range(args.num_total_tasks))
    unseen_ids = [t for t in all_ids if t not in seen_ids]

    seen_set = set(seen_ids)

    # Train: seen tasks only
    train_rows = build_rows(seen_ids, split="seen")

    # Val: all tasks, split tagged by seen/unseen for correct pass^k evaluation
    val_rows = []
    for tid in all_ids:
        split_tag = "seen" if tid in seen_set else "unseen"
        val_rows.extend(build_rows([tid], split=split_tag))

    train_path = Path(args.output_train)
    val_path = Path(args.output_val)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Train: {len(train_rows)} rows (seen tasks) -> {train_path}")
    print(f"Val:   {len(val_rows)} rows (all tasks)   -> {val_path}")
    print(f"Seen task IDs ({len(seen_ids)}): {seen_ids[:10]}{'...' if len(seen_ids) > 10 else ''}")
    print(f"Unseen task IDs ({len(unseen_ids)}): {unseen_ids[:10]}{'...' if len(unseen_ids) > 10 else ''}")


if __name__ == "__main__":
    main()
