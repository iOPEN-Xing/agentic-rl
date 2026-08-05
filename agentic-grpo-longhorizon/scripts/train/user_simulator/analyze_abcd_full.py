#!/usr/bin/env python3
"""Audit official ABCD train coverage without generating any model labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.abcd_adapter import (  # noqa: E402
    DEFAULT_PILOT_TARGETS,
    canonical_abcd_intent,
    count_abcd_continue_candidates,
)


DEFAULT_DIR = PROJECT_ROOT / "outputs/user_simulator_data/abcd_adapter"
PINNED_OFFICIAL_COMMIT = "6b8700ce67c6b37b062dd7a60abc76d7ef832a97"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
        "--output",
        type=Path,
        default=DEFAULT_DIR / "full_analysis/train_static_audit.json",
    )
    parser.add_argument("--official-commit", default=PINNED_OFFICIAL_COMMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with gzip.open(args.source, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if set(dataset) != {"train", "dev", "test"}:
        raise RuntimeError(f"unexpected ABCD split keys: {sorted(dataset)}")
    if not all(isinstance(dataset[split], list) for split in dataset):
        raise RuntimeError("ABCD splits must all be conversation lists")

    split_ids: dict[str, set[int]] = {}
    for split, rows in dataset.items():
        ids = [int(row["convo_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"duplicate conversation ID in ABCD {split}")
        split_ids[split] = set(ids)
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_ids[left] & split_ids[right]:
            raise RuntimeError(f"ABCD conversation overlap: {left}/{right}")

    kb = json.loads(args.kb.read_text(encoding="utf-8"))
    if not isinstance(kb, dict):
        raise RuntimeError("ABCD KB must be an intent-to-action mapping")

    train = dataset["train"]
    flows: Counter[str] = Counter()
    raw_leaves: Counter[str] = Counter()
    canonical_intents: Counter[str] = Counter()
    speakers: Counter[str] = Counter()
    raw_to_canonical: dict[str, set[str]] = defaultdict(set)
    candidates_by_intent: Counter[str] = Counter()
    conversations_by_intent: Counter[str] = Counter()
    candidate_totals: Counter[str] = Counter()
    causal_cut_conversations = 0
    unmappable_opening_conversations = 0

    for conversation in train:
        scenario = conversation["scenario"]
        flow = str(scenario["flow"])
        raw_subflow = str(scenario["subflow"])
        intent = canonical_abcd_intent(conversation)
        flows[flow] += 1
        raw_leaves[f"{flow}/{raw_subflow}"] += 1
        canonical_intents[intent] += 1
        raw_to_canonical[f"{flow}/{raw_subflow}"].add(intent)
        conversations_by_intent[intent] += 1
        for speaker, _ in conversation["original"]:
            speakers[str(speaker)] += 1

        candidate = count_abcd_continue_candidates(conversation)
        candidates_by_intent[intent] += int(candidate["candidate_continue_blocks"])
        for key, value in candidate.items():
            if isinstance(value, bool):
                continue
            candidate_totals[key] += int(value)
        causal_cut_conversations += int(candidate["causal_cut_applied"])
        unmappable_opening_conversations += int(
            candidate["opening_mapping_unmappable"]
        )

    if any(len(labels) != 1 for labels in raw_to_canonical.values()):
        raise RuntimeError("one raw ABCD scenario leaf maps to multiple canonical intents")
    canonical_set = set(canonical_intents)
    if canonical_set != set(kb):
        raise RuntimeError(
            "ABCD canonical intents differ from KB keys: "
            f"missing={sorted(set(kb) - canonical_set)}, "
            f"unexpected={sorted(canonical_set - set(kb))}"
        )
    pilot_ids = set(DEFAULT_PILOT_TARGETS)
    if not pilot_ids <= split_ids["train"]:
        raise RuntimeError("a reviewed ABCD pilot conversation is absent from train")

    candidate_upper_bound = candidate_totals["candidate_continue_blocks"]
    average_prompt_tokens = 14575 / 14
    average_completion_tokens = 1973 / 14

    def token_estimate(rows: int) -> dict[str, int]:
        return {
            "rows": rows,
            "prompt_tokens": round(rows * average_prompt_tokens),
            "completion_tokens": round(rows * average_completion_tokens),
            "total_tokens": round(
                rows * (average_prompt_tokens + average_completion_tokens)
            ),
        }

    aliases = {
        raw: next(iter(labels)) for raw, labels in sorted(raw_to_canonical.items())
    }
    non_identity_aliases = {
        raw: canonical
        for raw, canonical in aliases.items()
        if raw.rsplit("/", 1)[-1] != canonical
    }
    per_intent = {
        intent: {
            "conversations": canonical_intents[intent],
            "candidate_continue_blocks_upper_bound": candidates_by_intent[intent],
        }
        for intent in sorted(canonical_intents)
    }
    report: dict[str, Any] = {
        "format_version": "abcd-full-train-static-audit-v2",
        "source": {
            "repository": "https://github.com/asappresearch/abcd",
            "official_commit": args.official_commit,
            "compressed_file": str(args.source.resolve()),
            "compressed_bytes": args.source.stat().st_size,
            "compressed_sha256": _sha256(args.source),
            "kb_file": str(args.kb.resolve()),
            "kb_sha256": _sha256(args.kb),
        },
        "split_isolation": {
            "train_conversations": len(dataset["train"]),
            "dev_conversations_excluded": len(dataset["dev"]),
            "test_conversations_excluded": len(dataset["test"]),
            "pairwise_conversation_id_overlap": 0,
            "generation_split": "train_only",
        },
        "train_structure": {
            "flows": len(flows),
            "flow_counts": dict(sorted(flows.items())),
            "raw_scenario_leaves": len(raw_leaves),
            "canonical_intents": len(canonical_intents),
            "canonical_intents_equal_kb_keys": True,
            "speaker_turn_counts": dict(sorted(speakers.items())),
            "raw_leaf_to_canonical_is_single_valued": True,
            "non_identity_raw_leaf_aliases": len(non_identity_aliases),
            "non_identity_alias_map": non_identity_aliases,
        },
        "candidate_upper_bound": {
            **dict(candidate_totals),
            "causal_cut_conversations": causal_cut_conversations,
            "unmappable_opening_conversations": unmappable_opening_conversations,
            "note": (
                "Upper bound after deterministic reachability/courtesy gates; still "
                "contains semantic small talk and must not be sent wholesale to a Teacher."
            ),
        },
        "per_canonical_intent": per_intent,
        "review_coverage": {
            "reviewed_pilot_conversations": len(pilot_ids),
            "reviewed_pilot_targets": sum(
                len(turns) for turns in DEFAULT_PILOT_TARGETS.values()
            ),
            "unreviewed_train_conversations": len(train) - len(pilot_ids),
            "conversation_level_goal_spec_required": True,
            "subflow_global_goal_templates_allowed": False,
        },
        "token_budget_estimates_from_v0_3": {
            "observed_average_prompt_tokens": round(average_prompt_tokens, 1),
            "observed_average_completion_tokens": round(average_completion_tokens, 1),
            "next_55_intent_audit_2_conversations_2_targets_each": token_estimate(220),
            "controlled_1000_row_batch": token_estimate(1000),
            "unsafe_all_candidate_average_case_estimate": token_estimate(
                candidate_upper_bound
            ),
            "billing_note": "Token estimates are not provider price estimates.",
        },
        "expansion_gate": {
            "pilot_deterministic_and_human_audit_passed": True,
            "user_pilot_acceptance_received": False,
            "full_scale_generation_allowed": False,
            "deepseek_calls_made_by_this_analysis": 0,
            "next_safe_stage": (
                "After user acceptance, review two train conversations per canonical "
                "intent and at most two task-relevant targets per conversation."
            ),
        },
    }
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_conversations": len(train),
                "raw_scenario_leaves": len(raw_leaves),
                "canonical_intents": len(canonical_intents),
                "candidate_continue_blocks_upper_bound": candidate_upper_bound,
                "causal_cut_conversations": causal_cut_conversations,
                "full_scale_generation_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
