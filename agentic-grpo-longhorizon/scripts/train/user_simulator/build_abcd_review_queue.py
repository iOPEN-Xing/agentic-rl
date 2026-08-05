#!/usr/bin/env python3
"""Build a 55-intent ABCD human-review queue without calling a Teacher model."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.abcd_adapter import (  # noqa: E402
    FIXED_AGENT_GREETING,
    canonical_abcd_intent,
    select_abcd_review_queue,
)


DEFAULT_DIR = PROJECT_ROOT / "outputs/user_simulator_data/abcd_adapter"
PINNED_OFFICIAL_COMMIT = "6b8700ce67c6b37b062dd7a60abc76d7ef832a97"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = _temporary_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = _temporary_path(path)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DIR / "abcd_v1.1.official.json.gz",
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=DEFAULT_DIR / "kb.official.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DIR / "review_queue/v1",
    )
    parser.add_argument("--official-commit", default=PINNED_OFFICIAL_COMMIT)
    parser.add_argument("--conversations-per-intent", type=int, default=2)
    parser.add_argument("--targets-per-conversation", type=int, default=2)
    parser.add_argument("--seed", default="abcd-55-intent-review-v1")
    return parser.parse_args()


def _validate_splits(dataset: dict[str, Any]) -> dict[str, set[int]]:
    if set(dataset) != {"train", "dev", "test"}:
        raise RuntimeError(f"unexpected ABCD split keys: {sorted(dataset)}")
    split_ids: dict[str, set[int]] = {}
    for split, rows in dataset.items():
        if not isinstance(rows, list):
            raise RuntimeError(f"ABCD {split} must be a conversation list")
        ids = [int(row["convo_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate conversation ID in ABCD {split}")
        split_ids[split] = set(ids)
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise RuntimeError(f"ABCD conversation overlap: {left}/{right}")
    return split_ids


def _validate_queue(
    queue: list[dict[str, Any]],
    *,
    train_by_id: dict[int, dict[str, Any]],
    kb_intents: set[str],
    split_ids: dict[str, set[int]],
    conversations_per_intent: int,
    targets_per_conversation: int,
) -> dict[str, Any]:
    intent_counts = Counter(str(item["canonical_intent"]) for item in queue)
    if set(intent_counts) != kb_intents:
        raise RuntimeError(
            "review queue intents differ from KB: "
            f"missing={sorted(kb_intents - set(intent_counts))}, "
            f"unexpected={sorted(set(intent_counts) - kb_intents)}"
        )
    wrong_counts = {
        intent: count
        for intent, count in intent_counts.items()
        if count != conversations_per_intent
    }
    if wrong_counts:
        raise RuntimeError(f"review queue conversation count drift: {wrong_counts}")

    selected_ids = [int(item["source_convo_id"]) for item in queue]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("one ABCD conversation was selected more than once")
    if not set(selected_ids) <= split_ids["train"]:
        raise RuntimeError("review queue contains a non-train conversation")
    if set(selected_ids) & (split_ids["dev"] | split_ids["test"]):
        raise RuntimeError("review queue leaks an ABCD dev/test conversation")

    target_total = 0
    targets_after_hidden_action = 0
    selection_reasons: Counter[str] = Counter()
    source_opening_modes: Counter[str] = Counter()
    early_targets_at_literal_opening = 0
    later_targets_after_agent_question = 0
    selected_high_echo_targets = 0
    temporally_ordered_target_pairs = 0
    for item in queue:
        if item.get("allowed_for_generation") is not False:
            raise RuntimeError("unreviewed ABCD item was generation-enabled")
        if item.get("review", {}).get("status") != "pending":
            raise RuntimeError("new ABCD review item is not pending")
        targets = item.get("proposed_targets", [])
        source_opening_modes[str(item.get("source_opening_mode"))] += 1
        rendered_item = json.dumps(item, ensure_ascii=False).casefold()
        source_conversation = train_by_id[int(item["source_convo_id"])]
        for speaker, content in source_conversation["original"]:
            action_text = str(content).strip()
            if (
                str(speaker).casefold() == "action"
                and action_text
                and action_text.casefold() in rendered_item
            ):
                raise RuntimeError(
                    f"exact ABCD action text leaked: {item['source_convo_id']}"
                )
        if len(targets) != targets_per_conversation:
            raise RuntimeError(
                f"ABCD target count drift: {item['source_convo_id']}"
            )
        adjacent_target_pairs = list(zip(targets, targets[1:]))
        if any(
            max(left["source_turn_indices"]) >= min(right["source_turn_indices"])
            for left, right in adjacent_target_pairs
        ):
            raise RuntimeError(
                f"ABCD later target moved before the selected goal: "
                f"{item['source_convo_id']}"
            )
        temporally_ordered_target_pairs += len(adjacent_target_pairs)
        for target in targets:
            selection_reason = str(target.get("selection_reason"))
            selection_reasons[selection_reason] += 1
            early_targets_at_literal_opening += int(
                selection_reason == "early_goal_expression"
                and bool(target.get("is_opening_target"))
            )
            later_targets_after_agent_question += int(
                selection_reason == "later_goal_progress"
                and "?" in str(target.get("preceding_agent_text", ""))
            )
            selected_high_echo_targets += int(
                float(target.get("agent_echo_ratio", 0.0)) >= 0.8
            )
            prefix = target.get("student_prefix_without_system", [])
            if not prefix or prefix[-1].get("role") != "user":
                raise RuntimeError(
                    f"ABCD target lacks an Agent-ending prefix: "
                    f"{item['source_convo_id']}"
                )
            if any(message.get("role") == "system" for message in prefix):
                raise RuntimeError("unreviewed ABCD target contains a system prompt")
            expected_roles = [
                "user" if index % 2 == 0 else "assistant"
                for index in range(len(prefix))
            ]
            if [message.get("role") for message in prefix] != expected_roles:
                raise RuntimeError(
                    f"ABCD review prefix role drift: {item['source_convo_id']}"
                )
            if prefix[0].get("content") != FIXED_AGENT_GREETING:
                raise RuntimeError("ABCD review prefix greeting drift")
            source_indices = [int(index) for index in target["source_turn_indices"]]
            source_turns = [
                source_conversation["original"][index] for index in source_indices
            ]
            if any(str(turn[0]).casefold() != "customer" for turn in source_turns):
                raise RuntimeError("ABCD proposed target cites a non-customer turn")
            expected_response = re.sub(
                r"\s+",
                " ",
                " ".join(str(turn[1]) for turn in source_turns),
            ).strip()
            if expected_response != target.get("reference_response"):
                raise RuntimeError("ABCD proposed target/source text drift")
            targets_after_hidden_action += int(
                bool(target.get("hidden_action_turn_indices_before_target"))
            )
        target_total += len(targets)

    expected_conversations = len(kb_intents) * conversations_per_intent
    expected_targets = expected_conversations * targets_per_conversation
    if len(queue) != expected_conversations or target_total != expected_targets:
        raise RuntimeError(
            f"review queue size drift: conversations={len(queue)}, targets={target_total}"
        )
    if targets_after_hidden_action:
        raise RuntimeError(
            "ABCD review queue contains a target conditioned on hidden action state"
        )
    return {
        "canonical_intents": len(intent_counts),
        "selected_train_conversations": len(queue),
        "proposed_targets": target_total,
        "per_intent_conversations": dict(sorted(intent_counts.items())),
        "selected_raw_leaves": len({str(item["raw_leaf"]) for item in queue}),
        "selected_with_causal_cut": sum(
            bool(item.get("candidate_counts", {}).get("causal_cut_applied"))
            for item in queue
        ),
        "selected_targets_after_hidden_action": targets_after_hidden_action,
        "selection_reason_counts": dict(sorted(selection_reasons.items())),
        "source_opening_mode_counts": dict(sorted(source_opening_modes.items())),
        "early_targets_at_literal_opening": early_targets_at_literal_opening,
        "later_targets_after_agent_question": later_targets_after_agent_question,
        "selected_high_echo_targets": selected_high_echo_targets,
        "temporally_ordered_target_pairs": temporally_ordered_target_pairs,
        "exact_action_text_hits": 0,
    }


def _checklist_markdown(queue: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    lines = [
        "# ABCD 55-intent 扩量人工审核清单",
        "",
        "> 该清单只用于人工审核。所有记录仍为 pending，禁止送入 DeepSeek。",
        "",
        "## 审核要求",
        "",
        "每个会话必须根据源对话单独填写自然语言 `goal`、可选 `context` 和",
        "`goal_labels`。两个 target 分别判断是否只依赖客户已知场景与可见 Agent",
        "文本。任何 action 结果依赖、目标虚构、纯寒暄、领域专有策略污染或结束轮次",
        "都应拒绝。ABCD 不提供 STOP。",
        "",
        f"- 官方 commit：`{manifest['source']['official_commit']}`",
        f"- 选中会话：{manifest['queue']['selected_train_conversations']}",
        f"- 候选 targets：{manifest['queue']['proposed_targets']}",
        "- DeepSeek 调用：0",
        "- 全量生成门：关闭",
        "",
        "## 会话索引",
        "",
        (
            "| Intent | Convo ID | Opening map | Raw leaf | Target turns | "
            "Hidden-action targets | Status |"
        ),
        "|---|---:|---|---|---|---:|---|",
    ]
    for item in queue:
        target_turns = ", ".join(
            "+".join(str(index) for index in target["source_turn_indices"])
            for target in item["proposed_targets"]
        )
        hidden_action_targets = sum(
            bool(target["hidden_action_turn_indices_before_target"])
            for target in item["proposed_targets"]
        )
        lines.append(
            f"| `{item['canonical_intent']}` | {item['source_convo_id']} | "
            f"`{item['source_opening_mode']}` | `{item['raw_leaf']}` | "
            f"{target_turns} | "
            f"{hidden_action_targets} | pending |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.conversations_per_intent <= 0 or args.targets_per_conversation <= 0:
        raise RuntimeError("review queue sizes must be positive")
    with gzip.open(args.source, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    split_ids = _validate_splits(dataset)

    kb = json.loads(args.kb.read_text(encoding="utf-8"))
    if not isinstance(kb, dict) or not kb:
        raise RuntimeError("ABCD KB must be a non-empty intent mapping")
    kb_intents = {str(intent) for intent in kb}
    train_intents = {canonical_abcd_intent(row) for row in dataset["train"]}
    if train_intents != kb_intents:
        raise RuntimeError("ABCD train canonical intents differ from KB keys")

    queue = select_abcd_review_queue(
        dataset["train"],
        conversations_per_intent=args.conversations_per_intent,
        targets_per_conversation=args.targets_per_conversation,
        seed=args.seed,
    )
    queue_stats = _validate_queue(
        queue,
        train_by_id={int(row["convo_id"]): row for row in dataset["train"]},
        kb_intents=kb_intents,
        split_ids=split_ids,
        conversations_per_intent=args.conversations_per_intent,
        targets_per_conversation=args.targets_per_conversation,
    )

    output_dir = args.output_dir
    queue_path = output_dir / "review_queue.jsonl"
    manifest_path = output_dir / "manifest.json"
    checklist_path = output_dir / "REVIEW_CHECKLIST.md"
    manifest: dict[str, Any] = {
        "format_version": "abcd-55-intent-review-queue-v1",
        "source": {
            "repository": "https://github.com/asappresearch/abcd",
            "official_commit": args.official_commit,
            "compressed_file": str(args.source.resolve()),
            "compressed_sha256": _sha256(args.source),
            "kb_file": str(args.kb.resolve()),
            "kb_sha256": _sha256(args.kb),
            "source_split": "train_only",
            "dev_test_excluded": True,
        },
        "selection": {
            "seed": args.seed,
            "conversations_per_intent": args.conversations_per_intent,
            "targets_per_conversation": args.targets_per_conversation,
            "raw_leaf_diversity_preferred": True,
            "action_text_exported": False,
        },
        "queue": queue_stats,
        "gate": {
            "review_status": "pending",
            "user_pilot_acceptance_received": False,
            "allowed_for_generation": False,
            "full_scale_generation_allowed": False,
            "deepseek_calls_made": 0,
            "promotion_requires": [
                "conversation-level goal/context/goal_labels review",
                "per-target eligibility approval",
                "action-dependence and fact-grounding audit",
                "user acceptance of the existing 14-row pilot",
                "a separate fail-closed promotion validator",
            ],
        },
        "artifacts": {
            "review_queue": str(queue_path.resolve()),
            "checklist": str(checklist_path.resolve()),
        },
    }
    _write_jsonl_atomic(queue_path, queue)
    _write_json_atomic(manifest_path, manifest)
    _write_text_atomic(checklist_path, _checklist_markdown(queue, manifest))
    print(
        json.dumps(
            {
                "review_queue": str(queue_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "checklist": str(checklist_path.resolve()),
                **queue_stats,
                "allowed_for_generation": False,
                "deepseek_calls_made": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
