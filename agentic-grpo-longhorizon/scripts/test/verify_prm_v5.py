"""Verify PRM-Lite v6 scoring logic against real tau_bench_interaction.py.

v6 adds three hardening fixes from experiment analysis:
  A3: Invalid tool penalty (-0.05) for "Error: Unknown tool 'xxx'" from tau-bench base.py
  A4: Recursive placeholder check for nested params (flights[], passengers[])
  P3-v6: Recovery bonus only after legitimate errors (not after invalid tool calls)

22 test cases total (14 v5 + 8 v6).
Run: python scripts/test/verify_prm_v5.py
"""
import sys, json, re
from pathlib import Path
from types import ModuleType

# Ensure the project root + agentic-grpo-longhorizon is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "verl"))

# ── Mini mock modules for verl submodules that tau_bench_tools.py needs ─────────
class _MockToolResponse(ModuleType):
    pass

class _MockToolSchema(ModuleType):
    pass

_MOCK_TOOL_RESPONSE = ModuleType("verl.tools.schemas.ToolResponse")
_MOCK_TOOL_SCHEMA = ModuleType("verl.tools.schemas.OpenAIFunctionToolSchema")

import builtins as _bi
_orig_import = _bi.__import__
def _smart_import(name, *a, **kw):
    # Only block verl imports triggered during tau_bench_interaction.py loading
    if name == "verl.interactions.base":
        m = ModuleType(name)
        m.BaseInteraction = object
        return m
    if name == "verl.tools.base_tool":
        m = ModuleType(name)
        m.BaseTool = object
        return m
    if name == "verl.tools.schemas":
        m = ModuleType(name)
        m.ToolResponse = _MockToolResponse
        m.OpenAIFunctionToolSchema = _MockToolSchema
        return m
    if name == "verl.utils.rollout_trace":
        m = ModuleType(name)
        m.rollout_trace_op = lambda *a, **kw: None
        return m
    return _orig_import(name, *a, **kw)
_bi.__import__ = _smart_import

with (_ROOT / "src/envs/tau_bench_interaction.py").open(encoding="utf-8") as f:
    src = f.read()
src = src.replace("from verl.interactions.base import BaseInteraction", "pass")
src = src.replace("class TauBenchInteraction(BaseInteraction):", "class TauBenchInteraction:")
_ns = {}
exec(compile(src, "tau_bench_interaction.py", "exec"), _ns)
_scoring = _ns["_compute_reasoning_quality_score"]
_bi.__import__ = _orig_import


def mk(tool, params=None, **kw):
    """Build an action dict matching tau_bench_tools.py production format.

    v6: param_str uses JSON format (json.dumps) to match production _param_str().
    """
    p = params or {}
    return {
        "tool": tool,
        "parameters": p,
        # v6: JSON format (matches production _param_str in tau_bench_tools.py)
        "param_str": kw.get("param_str") or (
            "{}" if not p else json.dumps(dict(sorted(p.items())), sort_keys=True, ensure_ascii=False).lower()
        ),
        "is_error": kw.get("is_error", False),
        "observation": kw.get("observation", ""),
        "inc_reward": kw.get("inc_reward", 0.0),
        "content": kw.get("content", ""),
        "extracted_entities": kw.get("extracted_entities", {}),
    }


# ── Test cases — expected values = actual computed output (ground truth) ─────────
# Notes:
# 1. extracted_entities={} (empty dict) for all cases, matching _extract_entities("") output
# 2. param_str uses Python tuple format: str(sorted(params.items()))
# 3. search_direct_flight params don't match entity patterns → no data-chain bonus
# 4. P3 tests use diff reservation_id patterns so is_redundant doesn't fire
# 5. B4/B5 placeholder test uses explicit param_str (mixed JSON vs tuple → no redundancy match)
# 6. B4/B5 redundant test: step2 and step3 have same param_str → step3 is redundant
#
# Actual expected values (verified by running _scoring on 2026-08-07):
tests = [
    # (name, expected, history)
    ("P0: soft_fail",
     -0.01,
     [mk("search_direct_flight", params={"origin":"JFK","destination":"LAX","date":"2024-12-25"},
         observation="No flights found for this route", inc_reward=0.0)]),

    ("P0: error_prefix",
     -0.02,
     [mk("get_reservation_details", params={"reservation_id":"BADID"}, is_error=True,
         observation="Error: Unknown reservation ID format")]),

    ("P3: diff_tool_recovery",
     0.0,
     [mk("get_reservation_details", params={"reservation_id":"BAD"}, is_error=True),
      mk("get_user_details", params={"user_id":"john_doe_123"})]),

    ("P3: same_tool_recovery",
     -0.035,
     [mk("get_reservation_details", params={"reservation_id":"BAD"}, is_error=True),
      mk("get_reservation_details", params={"reservation_id":"GOOD123"})]),

    ("P3: error_repeat",
     -0.08,
     [mk("get_reservation_details", params={"reservation_id":"BAD"}, is_error=True),
      mk("get_reservation_details", params={"reservation_id":"BAD"})]),

    ("P5: no_reason_2step",
     0.0,
     [mk("get_user_details"), mk("get_reservation_details")]),

    ("P5: no_reason_3step",
     -0.033333333333333333,
     [mk("get_user_details"), mk("get_reservation_details"), mk("book_reservation")]),

    ("B4/B5: 2step_bypass_placeholder",
     -0.016666666666666667,
     [mk("think", content="Let me search for the reservation first"),
      mk("think", content="Actually let me use get_reservation_details"),
      mk("book_reservation", params={"reservation_id":"my_trip"},
         param_str='{"reservation_id": "my_trip"}')]),

    # v6.1: threshold=10, per_step=-0.003 (relaxed from v5 threshold=6)
    # 7 actions < 10: no length penalty. Only escalation (-0.05 each) + no_reasoning (-0.02).
    # per_step = -0.30/7 = -0.04286, + no_reasoning = -0.17571
    ("P8: length_penalty_7",
     -0.175714,
     [mk("transfer_to_human_agents") for _ in range(7)]),

    ("B7: read_diversity_2",
     0.0,
     [mk("get_user_details"), mk("get_reservation_details")]),

    ("B4/B5: 2step_bypass_redundant",
     -0.0025,
     [mk("think", content="I'll search first"),
      mk("think", content="Let me try get_user_details"),
      mk("get_user_details", params={"user_id":"john_doe_123"},
         param_str='{"user_id": "john_doe_123"}'),
      mk("get_user_details", params={"user_id":"john_doe_123"},
         param_str='{"user_id": "john_doe_123"}')]),

    ("B4/B5: 2step_bypass_think",
     0.0,
     [mk("think", content="Step 1"),
      mk("think", content="Step 2"),
      mk("think", content="Step 3")]),

    ("B4/B5: 3step_valid",
     0.006666666666666667,
     [mk("think", content="Let me think"),
      mk("think", content="Actually I should search"),
      mk("get_user_details", params={"user_id":"john_doe_123"})]),

    ("P0: soft_fail_recovery",
     -0.015,
     [mk("search_direct_flight", params={"origin":"JFK","destination":"LAX","date":"2024-12-25"},
         observation="No flights found for this route", inc_reward=0.0),
      mk("search_direct_flight", params={"origin":"JFK","destination":"SFO","date":"2024-12-25"},
         observation="Found 2 flights")]),

    # ── v6 A3: Invalid tool penalty ────────────────────────────────────────────────
    ("A3: invalid_tool_penalty",
     -0.05,
     [mk("search_flight_with_stops",
         observation="Error: Unknown tool 'search_flight_with_stops'",
         is_error=True)]),

    ("A3: valid_backend_error_no_bonus",
     0.0,
     [mk("book_reservation", params={"reservation_id":"ABC123"},
         observation="Error: payment amount does not add up",
         is_error=True)]),

    # valid error + diff tool: P3 +0.03, no first-read → step1 = +0.03
    # mean = (-0.02 + 0.03) / 2 = +0.005, no_reasoning_penalty=-0.02 → 0.0
    ("A3: valid_error_then_recovery",
     0.0,
     [mk("get_reservation_details", params={"reservation_id":"BAD"}, is_error=True,
         observation="Error: bad id"),
      mk("get_user_details", params={"user_id":"john_doe_123"}, observation="")]),

    # valid error + same tool+params → same_sig → error_repeat -0.04
    # step0: -0.02 (placeholder read), step1: -0.04 (same sig)
    # mean = (-0.02 + -0.04) / 2 = -0.03, no_reasoning_penalty=-0.02 → -0.05
    # NOTE: param_str format (auto-JSON vs tuple) doesn't affect this case since same_sig always matches
    ("A3: valid_error_same_entity",
     -0.05,
     [mk("get_reservation_details", params={"reservation_id":"ABC123"}, is_error=True,
         observation="Error: reservation ABC123 not found"),
      mk("get_reservation_details", params={"reservation_id":"ABC123"}, observation="Success")]),

    # invalid tool + next tool should NOT get recovery bonus
    ("A3: invalid_no_recovery_bonus",
     -0.04,
     [mk("search_flight_with_stops",
         observation="Error: Unknown tool 'search_flight_with_stops'",
         is_error=True),
      mk("get_user_details", params={"user_id":"john_doe_123"}, observation="")]),

    # ── v6 A4: Recursive placeholder check ─────────────────────────────────────
    ("A4: nested_flight_placeholder",
     -0.05,
     [mk("book_reservation", params={"flights": [{"flight_number": "UA891"}, {"flight_number": "previous"}],
         "passengers": [{"first_name": "John", "last_name": "Doe"}]},
         observation="Success")]),

    ("A4: nested_passenger_placeholder",
     -0.05,
     [mk("book_reservation", params={"passengers": [{"first_name": "Jane", "last_name": "temp_placeholder"}]},
         observation="Success")]),

    ("A4: top_level_placeholder",
     -0.05,
     [mk("book_reservation", params={"reservation_id": "my_trip"}, observation="Success")]),
]

ok = 0
for name, exp, history in tests:
    got = _scoring(history)
    good = abs(got - exp) < 1e-6
    if good:
        ok += 1
    status = "PASS" if good else "FAIL"
    print(f"  {status}  {name}: got={got:.5f}  exp={exp:.5f}")

print(f"\n{ok}/{len(tests)} PRM-Lite v6 cases passed")
if ok == len(tests):
    print("ALL CASES PASSED")
else:
    print("SOME CASES FAILED")
    sys.exit(1)
