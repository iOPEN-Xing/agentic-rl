#!/usr/bin/env python3
"""Build a reviewed, train-only ABCD pilot case set from the official sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_simulator_data.abcd_adapter import (  # noqa: E402
    ABCD_SOURCE,
    DEFAULT_PILOT_TARGETS,
    select_abcd_pilot_cases,
)


DEFAULT_DIR = PROJECT_ROOT / "outputs/user_simulator_data/abcd_adapter"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DIR / "abcd_sample.official.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIR / "pilot/source_cases.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DIR / "pilot/source_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError("ABCD source must be a JSON list of conversations")
    cases = select_abcd_pilot_cases(value)
    if len(cases) != sum(len(turns) for turns in DEFAULT_PILOT_TARGETS.values()):
        raise RuntimeError("pilot row count differs from the reviewed allowlist")
    if any(case.get("external_train_only") is not True for case in cases):
        raise RuntimeError("ABCD pilot case is not marked train-only")
    if any(case.get("expected_decision") != "continue" for case in cases):
        raise RuntimeError("ABCD pilot contains a non-CONTINUE target")
    rendered = json.dumps(cases, ensure_ascii=False).casefold()
    if '"role": "tool"' in rendered or '"speaker": "action"' in rendered:
        raise RuntimeError("ABCD action/tool state leaked into pilot cases")

    _atomic_jsonl(args.output, cases)
    counts = Counter(int(case["source_convo_id"]) for case in cases)
    manifest = {
        "format_version": "abcd-user-simulator-pilot-source-v1",
        "source": ABCD_SOURCE,
        "source_url": "https://github.com/asappresearch/abcd",
        "source_license": "MIT",
        "source_file": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "counts": {
            "conversations": len(counts),
            "cases": len(cases),
            "cases_by_conversation": {str(key): value for key, value in counts.items()},
            "continue": len(cases),
            "stop": 0,
        },
        "gates": {
            "reviewed_turn_allowlist": True,
            "external_train_only": True,
            "action_turns_visible": 0,
            "abcd_terminal_labels_used": 0,
            "human_semantic_review_required_after_generation": True,
            "full_scale_generation_allowed": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
