from __future__ import annotations

import json
import unittest

from src.user_simulator_data.abcd_adapter import (
    ABCD_PROMPT_VERSION,
    build_abcd_cases,
    build_abcd_scenario,
    build_abcd_teacher_messages,
    generate_abcd_one,
    select_abcd_pilot_cases,
)


def sample_conversation() -> dict:
    return {
        "convo_id": 9489,
        "scenario": {
            "personal": {
                "customer_name": "alessandro phoenix",
                "email": "aphoenix939@email.com",
                "member_level": "gold",
                "username": "aphoenix939",
            },
            "order": {
                "order_id": "7916676427",
                "purchase_date": "2019-11-20",
                "products": "[{'brand': 'michael_kors', 'product_type': 'shirt', "
                "'amount': 69, 'image_url': 'images/michael_kors-shirt.jpeg'}]",
            },
            "product": {
                "names": ["michael_kors shirt"],
                "amounts": [69],
            },
            "flow": "product_defect",
            "subflow": "refund_status",
        },
        "original": [
            ["agent", "Good afternoon."],
            ["agent", "How can I help you?"],
            ["customer", "I want to check the status of a refund."],
            ["agent", "May I have your full name?"],
            ["customer", "Alessandro"],
            ["customer", "Phoenix"],
            ["action", "Account 7916676427 was pulled up."],
            ["agent", "Your refund is still processing."],
            ["customer", "How much longer will it take?"],
            ["action", "Refund ETA is 7 days."],
            ["customer", "Thanks, that's all."],
        ],
    }


class FakeDeepSeekClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.request = None

    def complete_json(self, messages, **kwargs):
        self.request = {"messages": messages, **kwargs}
        return {
            "content": json.dumps(self.payload),
            "model": "deepseek-v4-flash",
            "response_id": "pilot-response",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "request_config": {"thinking": False},
        }


class ABCDAdapterTests(unittest.TestCase):
    def test_scenario_exposes_customer_known_goal_and_facts_not_latent_labels(self):
        scenario = build_abcd_scenario(sample_conversation()["scenario"])

        self.assertIn("check the status of a refund", scenario.casefold())
        self.assertIn("how long it will take", scenario.casefold())
        self.assertIn("7916676427", scenario)
        self.assertIn("aphoenix939@email.com", scenario)
        self.assertNotIn("product_defect", scenario)
        self.assertNotIn("refund_status", scenario)
        self.assertNotIn("image_url", scenario)
        self.assertNotIn("images/", scenario)

    def test_case_builder_role_flips_and_merges_customer_fragments(self):
        cases = build_abcd_cases(sample_conversation())

        self.assertEqual(len(cases), 3)
        self.assertEqual(
            [case["reference_response"] for case in cases],
            [
                "I want to check the status of a refund.",
                "Alessandro Phoenix",
                "How much longer will it take?",
            ],
        )
        self.assertEqual(cases[0]["student_messages"][1]["content"], "Hi! How can I help you today?")
        self.assertEqual(
            [message["role"] for message in cases[-1]["student_messages"]],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertTrue(all(case["expected_decision"] == "continue" for case in cases))
        self.assertNotIn("Account 7916676427 was pulled up", json.dumps(cases))
        self.assertNotIn("Refund ETA is 7 days", json.dumps(cases))
        self.assertTrue(
            all(
                "Thanks, that's all." != case["reference_response"]
                for case in cases
            )
        )

    def test_teacher_prompt_uses_reference_only_as_teacher_semantic_anchor(self):
        case = build_abcd_cases(sample_conversation())[1]
        messages = build_abcd_teacher_messages(case)
        rendered = json.dumps(messages, ensure_ascii=False)

        self.assertIn("Alessandro Phoenix", rendered)
        self.assertIn("decision=continue", rendered)
        self.assertIn("must never emit ###STOP###", rendered)
        self.assertNotIn("Account 7916676427 was pulled up", rendered)
        self.assertNotIn("Refund ETA is 7 days", rendered)
        self.assertEqual(case["student_messages"][-1]["role"], "user")
        self.assertNotEqual(
            case["student_messages"][-1]["content"], case["reference_response"]
        )

    def test_generation_produces_current_last_assistant_sft_contract(self):
        case = build_abcd_cases(sample_conversation())[2]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "About how much longer should the refund take?",
                "resolved_goals": ["refund status is processing"],
                "unresolved_goals": ["refund completion time"],
                "evidence": [
                    {
                        "turn_index": 7,
                        "fact": "The agent said the refund is still processing.",
                    }
                ],
            }
        )

        row = generate_abcd_one(client, case)

        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["prompt_version"], ABCD_PROMPT_VERSION)
        self.assertEqual(row["sft_record"]["messages"][-1], {
            "role": "assistant",
            "content": "About how much longer should the refund take?",
        })
        self.assertEqual(
            row["sft_record"]["metadata"]["target_provenance"],
            "deepseek_abcd_pilot",
        )
        self.assertFalse(client.request["thinking"])

    def test_generation_quarantines_novel_identifier_even_when_json_contract_is_valid(self):
        case = build_abcd_cases(sample_conversation())[1]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "My order ID is 1234567890.",
                "resolved_goals": [],
                "unresolved_goals": ["refund status"],
                "evidence": [],
            }
        )

        row = generate_abcd_one(client, case)

        self.assertEqual(row["status"], "quarantined")
        self.assertIn("novel_numeric_fact:1234567890", row["quality_issues"])

    def test_pilot_selection_fails_closed_when_a_reviewed_turn_is_missing(self):
        selected = select_abcd_pilot_cases(
            [sample_conversation()], target_map={9489: {2, 4, 8}}
        )

        self.assertEqual(
            [case["source_turn_indices"][0] for case in selected], [2, 4, 8]
        )
        self.assertTrue(
            all(case["pilot_selection"] == "reviewed_turn_allowlist" for case in selected)
        )
        with self.assertRaisesRegex(ValueError, "reviewed ABCD pilot targets were not built"):
            select_abcd_pilot_cases(
                [sample_conversation()], target_map={9489: {2, 999}}
            )


if __name__ == "__main__":
    unittest.main()
