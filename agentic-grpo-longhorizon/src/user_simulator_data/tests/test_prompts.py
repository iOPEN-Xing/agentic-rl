from __future__ import annotations

import unittest

from src.user_simulator_data.prompts import PROMPT_VERSION, build_teacher_messages


class PromptTests(unittest.TestCase):
    def test_teacher_prompt_separates_observable_context_from_privileged_reference(self):
        case = {
        "case_id": "airline-0000-final",
        "task_id": 0,
        "scenario": "Book the cheapest acceptable flight and use my certificate.",
        "observable_history": [
            {"role": "user", "content": "I need a flight to Seattle."},
            {"role": "agent", "content": "Your booking is complete."},
        ],
        "privileged_reference": {
            "required_actions": [{"name": "book_reservation"}],
            "required_outputs": [],
            "tool_trace": [{"name": "book_reservation", "result": "success"}],
        },
    }

        messages = build_teacher_messages(case)
        rendered = "\n".join(message["content"] for message in messages)

        self.assertIn(PROMPT_VERSION, rendered)
        self.assertIn("OBSERVABLE_CONTEXT", rendered)
        self.assertIn("PRIVILEGED_AUDIT_REFERENCE", rendered)
        self.assertIn("must not reveal", rendered.lower())
        self.assertIn('"is_over"', rendered)
        self.assertIn('"response"', rendered)
        self.assertIn("###STOP###", rendered)


    def test_teacher_prompt_says_state_success_alone_is_not_user_visible_completion(self):
        case = {
        "case_id": "airline-0002-missing-output",
        "task_id": 2,
        "scenario": "Downgrade all flights and tell me total savings.",
        "observable_history": [
            {"role": "agent", "content": "All reservations were downgraded."}
        ],
        "privileged_reference": {
            "environment_goal_status": "satisfied",
            "required_outputs": ["23553"],
        },
    }

        rendered = "\n".join(
            message["content"] for message in build_teacher_messages(case)
        )

        self.assertIn("environment state is satisfied", rendered.lower())
        self.assertIn("requested result", rendered.lower())
        self.assertIn("continue", rendered.lower())


if __name__ == "__main__":
    unittest.main()
