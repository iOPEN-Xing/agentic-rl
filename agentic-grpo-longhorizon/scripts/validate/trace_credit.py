#!/usr/bin/env python3
"""Deterministic formula checks for the local TRACE implementation."""

from __future__ import annotations

import json

import torch

from verl.trainer.ppo.hybrid_advantage import compute_trace_log_ratio_values


def main() -> int:
    appendix_first = compute_trace_log_ratio_values(
        torch.tensor([-5.1187, -1.5712]),
        gap_epsilon=0.001,
    )
    appendix_second = compute_trace_log_ratio_values(
        torch.tensor([-10.6570, -7.1061]),
        gap_epsilon=0.001,
    )
    trajectory = compute_trace_log_ratio_values(
        torch.tensor([-4.0, -3.0, -1.5, -0.8]),
        gap_epsilon=0.1,
    )
    deltas = trajectory[1:] - trajectory[:-1]

    first_credit = float(appendix_first[1] - appendix_first[0])
    second_credit = float(appendix_second[1] - appendix_second[0])
    telescope_error = float(abs(deltas.sum() - (trajectory[-1] - trajectory[0])))
    result = {
        "paper_appendix": {
            "expected": [1.1806, 0.4052],
            "actual": [first_credit, second_credit],
        },
        "telescope_error": telescope_error,
    }
    result["passed"] = (
        abs(first_credit - 1.1806) < 1e-3
        and abs(second_credit - 0.4052) < 1e-3
        and telescope_error < 1e-6
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
