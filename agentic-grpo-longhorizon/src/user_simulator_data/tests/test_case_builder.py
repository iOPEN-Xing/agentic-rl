from __future__ import annotations

import json
import unittest

from src.user_simulator_data.case_builder import (
    build_cases_from_historical_rows,
    prepare_curated_case,
)


class CaseBuilderTests(unittest.TestCase):
    def test_prepare_curated_case_flips_customer_and_agent_roles_for_student(self):
        case = prepare_curated_case(
            {
                "case_id": "pilot-role-flip",
                "task_id": 0,
                "scenario": "Book a flight.",
                "observable_history": [
                    {"role": "user", "content": "I need a flight."},
                    {"role": "agent", "content": "Which date?"},
                ],
                "expected_decision": "continue",
            }
        )

        self.assertEqual(
            [message["role"] for message in case["student_messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(case["student_messages"][-1]["content"], "Which date?")
        self.assertTrue(case["quality_gate"])

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

    def test_historical_case_builder_cuts_downstream_after_ungrounded_reply(self):
        rows = [
            {
                "task_id": 1,
                "reward": 0.0,
                "info": {
                    "task": {
                        "instruction": (
                            "You are olivia_gonzalez_2305 and do not remember "
                            "the reservation ID."
                        ),
                        "actions": [],
                        "outputs": [],
                    }
                },
                "traj": [
                    {"role": "user", "content": "I need help with my booking."},
                    {
                        "role": "assistant",
                        "content": "Please check your email for the reservation ID.",
                    },
                    {
                        "role": "user",
                        "content": "I found it. My reservation ID is FAKE123.",
                    },
                    {
                        "role": "assistant",
                        "content": "I cannot find reservation FAKE123.",
                    },
                    {"role": "user", "content": "Please try again."},
                ],
                "trial": 0,
            }
        ]

        cases = build_cases_from_historical_rows(rows)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[-1]["reference_response"], "I found it. My reservation ID is FAKE123.")
        self.assertIn(
            "unsupported_reservation_id:FAKE123",
            cases[-1]["reference_grounding_issues"],
        )
        self.assertTrue(all(case["reference_response"] != "Please try again." for case in cases))


if __name__ == "__main__":
    unittest.main()
