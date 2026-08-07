"""Stand-alone regression for the v5 _compute_reasoning_quality_score function.

Strategy: stub out verl's BaseInteraction so we can import the real source
file without installing all of verl.

In v5, an empty obs + inc_reward=0 + non-think tool triggers soft-failure
penalty (-0.10). For each test, we either:

(a) Explicitly exercise the soft-failure rule (use empty obs + inc_reward=0),
or
(b) Isolate the rule under test by giving the action a non-empty success-obs
    OR positive inc_reward.
"""
import sys
import types
import traceback
from pathlib import Path


def _stub_verl():
    """Create stub modules so we can import tau_bench_interaction.py."""
    base_mod = types.ModuleType("verl.interactions.base")

    class BaseInteraction:
        pass

    base_mod.BaseInteraction = BaseInteraction

    verl = types.ModuleType("verl")
    verl_interactions = types.ModuleType("verl.interactions")
    verl.interactions = verl_interactions
    verl_interactions.base = base_mod
    verl.utils = types.ModuleType("verl.utils")
    rollout_trace = types.ModuleType("verl.utils.rollout_trace")

    class _DecorStub:
        def __call__(self, *args, **kw):
            if args and callable(args[0]) and not kw:
                return args[0]
            def deco(fn):
                return fn
            return deco

    rollout_trace.rollout_trace_op = _DecorStub()

    tau_bench = types.ModuleType("tau_bench")
    tau_bench_types = types.ModuleType("tau_bench.types")

    class _Stub:
        pass

    tau_bench_types.Action = _Stub
    tau_bench_types.RESPOND_ACTION_NAME = "respond"
    tau_bench_types.ToolResponse = _Stub
    tau_bench_types.Task = _Stub

    tau_bench.types = tau_bench_types

    sys.modules.update({
        "verl": verl,
        "verl.interactions": verl_interactions,
        "verl.interactions.base": base_mod,
        "verl.utils": verl.utils,
        "verl.utils.rollout_trace": rollout_trace,
        "tau_bench": tau_bench,
        "tau_bench.types": tau_bench_types,
    })


# Helper: success obs for read tools (must contain a keyword)
READ_SUCCESS_OBS = "user data with { details }"
# Helper: success obs for write tools
WRITE_SUCCESS_OBS = "Reservation updated successfully."
# Helper: empty (triggers soft-failure)
EMPTY_OBS = ""


def make(tool, params=None, is_error=False, extracted_entities=None,
         content="", inc_reward=0.0, observation=EMPTY_OBS, param_str=None):
    # Default param_str to the canonical JSON serialization of params,
    # matching _param_str() in tau_bench_interaction.py.
    if param_str is None:
        if params:
            import json as _json
            param_str = _json.dumps(params, sort_keys=True, ensure_ascii=False).lower()
        else:
            param_str = "{}"
    return {
        "tool": tool, "parameters": params or {}, "param_str": param_str,
        "is_error": is_error, "extracted_entities": extracted_entities or {},
        "content": content, "inc_reward": inc_reward, "observation": observation,
    }


def assert_close(actual, expected, tol=1e-3):
    return abs(actual - expected) < tol


def main():
    _stub_verl()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from src.envs.tau_bench_interaction import (
        _compute_reasoning_quality_score,
        _is_tool_call_failure,
        _obs_indicates_success,
    )

    tests = []

    # --- empty ---
    def t_empty():
        return _compute_reasoning_quality_score([]) == 0.0
    tests.append(("empty", t_empty))

    # --- P0 hard error ---
    def t_p0_hard():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 is_error=True, observation="Error: bad id")
        ])
        return assert_close(s, -0.10, tol=1e-2)
    tests.append(("p0_hard_error", t_p0_hard))

    # --- P0 soft failure (empty obs + inc_reward=0) ---
    def t_p0_soft_empty():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation="")
        ])
        return assert_close(s, -0.10, tol=1e-2)
    tests.append(("p0_soft_empty_obs", t_p0_soft_empty))

    # --- P0 NOT a failure (success obs) ---
    def t_p0_no_failure():
        s = _compute_reasoning_quality_score([
            make("cancel_reservation", params={"reservation_id": "ABC123"},
                 observation=WRITE_SUCCESS_OBS)
        ])
        return assert_close(s, 0.0, tol=1e-3)
    tests.append(("p0_no_failure_with_success_obs", t_p0_no_failure))

    # --- P3 same-tool fixed-args recovery (+0.02 in v5) ---
    def t_p3_same_tool():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "previous_reservation"},
                 is_error=True, observation="Error: not found",
                 param_str='{"user_id": "previous_reservation"}'),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 extracted_entities={"user_id": ["mia_li_3668"]},
                 observation=READ_SUCCESS_OBS, param_str='{"user_id": "mia_li_3668"}'),
        ])
        return assert_close(s, -0.07, tol=1e-3)
    tests.append(("p3_same_tool_fixed_args_v5", t_p3_same_tool))

    # --- P3 cross-tool recovery (+0.03 in v5) ---
    def t_p3_cross_tool():
        # step0: error -0.10 + placeholder -0.03 + first-read +0.01 = -0.12
        # step1: P3 cross +0.03 + first-read +0.01 = +0.04
        # mean = (-0.12 + 0.04) / 2 = -0.04
        # P5(len>=2) -0.02; diversity (2 read tools) +0.01
        # final = -0.05
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "previous_reservation"},
                 is_error=True, observation="Error: not found"),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
        ])
        return assert_close(s, -0.05, tol=1e-3)
    tests.append(("p3_cross_tool_recovery_v5", t_p3_cross_tool))

    # --- P3 repeat same error ---
    def t_p3_repeat():
        # step0: error -0.10 + placeholder -0.03 + first-read +0.01 = -0.12
        # step1: soft-fail -0.10 + placeholder -0.03 + redundancy -0.03 + repeat -0.04
        #   = -0.20
        # mean = (-0.12 + -0.20) / 2 = -0.16
        # P5(len>=2) -0.02
        # final = -0.18
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "x"}, is_error=True,
                 observation="Error: not found"),
            make("get_user_details", params={"user_id": "x"},
                 observation="Error: not found"),
        ])
        return assert_close(s, -0.18, tol=1e-3)
    tests.append(("p3_repeat_same_error", t_p3_repeat))

    # --- B4/B5: think→think→placeholder→real tool (v5 catches 2-step attack) ---
    def t_think_hack_v5():
        s = _compute_reasoning_quality_score([
            make("think", content="ok"),
            make("think", content="ok2"),
            make("book_reservation", params={"reservation_id": "my_trip"},
                 observation=WRITE_SUCCESS_OBS),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
        ])
        return assert_close(s, -0.01, tol=1e-3)
    tests.append(("b4_b5_think_hack_v5", t_think_hack_v5))

    # --- B4/B5: think → direct non-placeholder tool rewarded ---
    def t_think_direct():
        s = _compute_reasoning_quality_score([
            make("think", content="ok"),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
        ])
        return assert_close(s, 0.01, tol=1e-3)
    tests.append(("b4_b5_think_direct_rewarded", t_think_direct))

    # --- P5: len=2 no reasoning → -0.02 (v5 only) ---
    def t_p5_len2():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
            make("cancel_reservation", params={"reservation_id": "ABC123"},
                 observation=WRITE_SUCCESS_OBS),
        ])
        return assert_close(s, -0.015, tol=1e-3)
    tests.append(("p5_len2_no_reasoning_v5", t_p5_len2))

    # --- P5: len>=3 no reasoning → -0.05 ---
    def t_p5_len3():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 extracted_entities={"user_id": ["mia_li_3668"]},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
            make("cancel_reservation", params={"reservation_id": "ABC123"},
                 observation=WRITE_SUCCESS_OBS),
        ])
        return assert_close(s, -0.0333, tol=1e-3)
    tests.append(("p5_len3_no_reasoning", t_p5_len3))

    # --- B7: diversity at 2 tools (v5; v4 required 3) ---
    def t_b7_diversity_2():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
        ])
        return assert_close(s, 0.0, tol=1e-3)
    tests.append(("b7_diversity_2_v5", t_b7_diversity_2))

    # --- P8: length penalty threshold 6 (v5; v4 was 8) ---
    def t_p8_threshold_6():
        s = _compute_reasoning_quality_score([
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "ABC123"},
                 observation=READ_SUCCESS_OBS),
        ])
        return assert_close(s, -0.06357, tol=0.001)
    tests.append(("p8_threshold_6_v5", t_p8_threshold_6))

    # --- Helper: obs_indicates_success ---
    def t_helper_obs():
        return (
            _obs_indicates_success("Reservation updated", "cancel_reservation")
            and not _obs_indicates_success("", "cancel_reservation")
            and not _obs_indicates_success("Error: bad", "cancel_reservation")
            and not _obs_indicates_success("nothing here", "cancel_reservation")
            and _obs_indicates_success("{...user: {details...}}", "get_user_details")
            and not _obs_indicates_success("plain", "get_user_details")
        )
    tests.append(("helper_obs_success", t_helper_obs))

    # --- Helper: is_tool_call_failure ---
    def t_helper_failure():
        return (
            _is_tool_call_failure(0.0, "nothing", "cancel_reservation")
            and not _is_tool_call_failure(0.0, "Reservation cancelled", "cancel_reservation")
            and not _is_tool_call_failure(1.0, "anything", "cancel_reservation")
            and not _is_tool_call_failure(0.0, "", "think")
            and not _is_tool_call_failure(0.0, "", "implicit_think")
        )
    tests.append(("helper_is_failure", t_helper_failure))

    # --- Full successful trajectory smoke test ---
    def t_full_successful():
        s = _compute_reasoning_quality_score([
            make("think", content="Looking up user."),
            make("get_user_details", params={"user_id": "mia_li_3668"},
                 extracted_entities={"user_id": ["mia_li_3668"], "reservation_id": ["Z7GOZK"]},
                 observation=READ_SUCCESS_OBS),
            make("get_reservation_details", params={"reservation_id": "Z7GOZK"},
                 extracted_entities={"reservation_id": ["Z7GOZK"]},
                 observation=READ_SUCCESS_OBS),
            make("cancel_reservation", params={"reservation_id": "Z7GOZK"},
                 inc_reward=1.0,
                 observation=WRITE_SUCCESS_OBS),
        ])
        return 0.0 < s < 0.1
    tests.append(("full_successful", t_full_successful))

    passed = failed = 0
    for name, test_fn in tests:
        try:
            ok = test_fn()
            if ok:
                print(f"  PASS {name}")
                passed += 1
            else:
                print(f"  FAIL {name}: returned False")
                failed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())