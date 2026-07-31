#!/usr/bin/env python3
"""Check that session and turn advantage components are scale-normalized."""

from __future__ import annotations

import argparse
import json

import torch

from verl.trainer.ppo.hybrid_advantage import normalize_advantage_component


def _stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-normalization", action="store_true")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")

    generator = torch.Generator().manual_seed(args.seed)
    session = torch.randn(args.samples, generator=generator)
    turn = 0.1 * torch.randn(args.samples, generator=generator)
    mask = torch.ones(args.samples)
    session_normalized = normalize_advantage_component(session, mask)
    turn_normalized = normalize_advantage_component(turn, mask)

    result = {
        "before": {
            "session": _stats(session),
            "turn": _stats(turn),
        },
        "after": {
            "session": _stats(session_normalized),
            "turn": _stats(turn_normalized),
        },
    }
    result["normalized"] = all(
        abs(result["after"][name]["mean"]) < 1e-5
        and abs(result["after"][name]["std"] - 1.0) < 1e-4
        for name in ("session", "turn")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["normalized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
