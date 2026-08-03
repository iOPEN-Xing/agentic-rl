from __future__ import annotations

import json
import unittest

from src.user_simulator_data.generation import (
    generate_one,
    replicate_cases,
    summarize_generations,
)


class FakeClient:
    def __init__(self, content):
        self.content = content

    def complete_json(self, messages):
        return {
            "content": json.dumps(self.content),
            "model": "deepseek-v4-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


def source_case(expected_decision="continue"):
    return {
        "case_id": "pilot-1",
        "task_id": 34,
        "scenario": "Cancel both bookings and tell me the exact total cost.",
        "observable_history": [
            {"turn_index": 0, "role": "agent", "content": "Both are cancelled."}
        ],
        "student_messages": [
            {"role": "system", "content": "simulate customer"},
            {"role": "user", "content": "Both are cancelled."},
        ],
        "expected_decision": expected_decision,
        "quality_gate": True,
        "privileged_reference": {
            "privileged_entities": ["HIDDEN42"],
            "required_outputs": ["1016"],
        },
    }


class GenerationTests(unittest.TestCase):
    def test_replicate_cases_produces_resume_safe_unique_ids(self):
        replicas = replicate_cases([source_case()], samples_per_case=3)

        self.assertEqual(len(replicas), 3)
        self.assertEqual(
            [case["case_id"] for case in replicas],
            ["pilot-1-rep00", "pilot-1-rep01", "pilot-1-rep02"],
        )
        self.assertTrue(all(case["base_case_id"] == "pilot-1" for case in replicas))

    def test_generate_one_outputs_plain_student_target_and_audit_record(self):
        client = FakeClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "Please also give me the exact total cost.",
                "resolved_goals": ["two cancellations"],
                "unresolved_goals": ["exact total cost"],
                "evidence": [
                    {"turn_index": 0, "fact": "Agent confirmed both cancellations."}
                ],
            }
        )

        generated = generate_one(client, source_case())

        self.assertEqual(generated["status"], "accepted")
        self.assertEqual(
            generated["sft_record"]["messages"][-1]["content"],
            "Please also give me the exact total cost.",
        )
        self.assertEqual(generated["teacher_decision"]["decision"], "continue")
        self.assertNotIn("raw_reasoning", generated)

    def test_generation_is_quarantined_on_expected_decision_mismatch(self):
        client = FakeClient(
            {
                "decision": "stop",
                "is_over": True,
                "termination_reason": "goal_satisfied",
                "goal_status": "satisfied",
                "communication_status": "complete",
                "response": "###STOP###",
                "resolved_goals": ["everything"],
                "unresolved_goals": [],
                "evidence": [],
            }
        )

        generated = generate_one(client, source_case(expected_decision="continue"))

        self.assertEqual(generated["status"], "quarantined")
        self.assertIn("expected_decision_mismatch", generated["quality_issues"])

    def test_summary_gate_requires_perfect_curated_decisions_and_no_leaks(self):
        accepted = {
            "case_id": "a",
            "status": "accepted",
            "quality_gate": True,
            "expected_decision": "continue",
            "teacher_decision": {"decision": "continue", "is_over": False},
            "quality_issues": [],
        }
        mismatch = {
            **accepted,
            "case_id": "b",
            "status": "quarantined",
            "teacher_decision": {"decision": "stop", "is_over": True},
            "quality_issues": ["expected_decision_mismatch"],
        }

        passed = summarize_generations([accepted])
        failed = summarize_generations([accepted, mismatch])

        self.assertTrue(passed["quality_gate_passed"])
        self.assertFalse(failed["quality_gate_passed"])
        self.assertEqual(failed["curated_decision_accuracy"], 0.5)

if __name__ == "__main__":
    unittest.main()
