from __future__ import annotations

import json
import unittest

from src.user_simulator_data.case_builder import build_cases_from_historical_rows


class CaseBuilderTests(unittest.TestCase):
    def test_historical_case_builder_preserves_terminal_target_for_audit_only(self):
        rows = [
        {
            "task_id": 34,
            "reward": 1.0,
            "info": {
                "task": {
                    "instruction": "Cancel two reservations and report other flight cost.",
                    "actions": [{"name": "cancel_reservation", "kwargs": {}}],
                    "outputs": [],
                }
            },
            "traj": [
                {"role": "user", "content": "Cancel both reservations."},
                {"role": "assistant", "content": "Both are cancelled. Total is $1016."},
                {"role": "user", "content": "###STOP###"},
            ],
            "trial": 0,
        }
    ]

        cases = build_cases_from_historical_rows(rows)

        self.assertEqual(len(cases), 2)
        terminal = cases[-1]
        self.assertEqual(terminal["reference_response"], "###STOP###")
        self.assertEqual(terminal["expected_decision"], "stop")
        self.assertEqual(
            terminal["privileged_reference"]["trajectory_reward"], 1.0
        )
        self.assertTrue(
            all(
                message["role"] != "tool"
                for message in terminal["observable_history"]
            )
        )
        self.assertEqual(terminal["student_messages"][-1]["role"], "user")
        self.assertTrue(
            terminal["student_messages"][-1]["content"].startswith(
                "Both are cancelled"
            )
        )


    def test_historical_case_builder_excludes_tool_results_from_student_messages(self):
        rows = [
        {
            "task_id": 0,
            "reward": 0.0,
            "info": {
                "task": {
                    "instruction": "Book a flight.",
                    "actions": [],
                    "outputs": [],
                }
            },
            "traj": [
                {"role": "user", "content": "I need a flight."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_direct_flight", "arguments": {}}}
                    ],
                },
                {"role": "tool", "name": "search_direct_flight", "content": "SECRET"},
                {"role": "assistant", "content": "What time do you prefer?"},
                {"role": "user", "content": "After 11am."},
            ],
            "trial": 2,
        }
    ]

        cases = build_cases_from_historical_rows(rows)
        target_case = cases[-1]

        self.assertNotIn("SECRET", json.dumps(target_case["student_messages"]))
        self.assertIn(
            "SECRET", json.dumps(target_case["privileged_reference"]["tool_trace"])
        )
        self.assertEqual(target_case["reference_response"], "After 11am.")


if __name__ == "__main__":
    unittest.main()
