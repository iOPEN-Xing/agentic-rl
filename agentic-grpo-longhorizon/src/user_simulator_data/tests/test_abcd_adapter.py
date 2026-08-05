from __future__ import annotations

import json
import unittest

from src.user_simulator_data.abcd_adapter import (
    ABCD_PROMPT_VERSION,
    build_abcd_cases,
    build_abcd_scenario,
    build_abcd_teacher_messages,
    canonical_abcd_intent,
    count_abcd_continue_candidates,
    generate_abcd_one,
    select_abcd_pilot_cases,
    validate_abcd_resume_alignment,
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
        "delexed": [
            {
                "speaker": "agent",
                "text": "good afternoon",
                "targets": ["refund_status", "retrieve_utterance", None, [], 0],
            },
            {
                "speaker": "customer",
                "text": "refund status",
                "targets": ["refund_status", None, None, [], -1],
            },
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
        scenario = build_abcd_scenario(
            sample_conversation()["scenario"], convo_id=9489
        )

        self.assertIn("check the status of a refund", scenario.casefold())
        self.assertIn("how long it will take", scenario.casefold())
        self.assertIn("7916676427", scenario)
        self.assertIn("aphoenix939@email.com", scenario)
        self.assertNotIn("product_defect", scenario)
        self.assertNotIn("refund_status", scenario)
        self.assertNotIn("image_url", scenario)
        self.assertNotIn("images/", scenario)

    def test_scenario_fails_closed_for_unreviewed_conversation_in_known_subflow(self):
        with self.assertRaisesRegex(ValueError, "unreviewed ABCD pilot conversation"):
            build_abcd_scenario(
                sample_conversation()["scenario"], convo_id=123456
            )

    def test_canonical_intent_comes_from_consistent_delexed_targets(self):
        conversation = sample_conversation()
        conversation["scenario"]["subflow"] = "timing_4"
        for turn in conversation["delexed"]:
            turn["targets"][0] = "timing"

        self.assertEqual(canonical_abcd_intent(conversation), "timing")

    def test_canonical_intent_rejects_turn_level_label_drift(self):
        conversation = sample_conversation()
        conversation["delexed"][1]["targets"][0] = "refund_update"

        with self.assertRaisesRegex(ValueError, "canonical intent drift"):
            canonical_abcd_intent(conversation)

    def test_full_analysis_counts_only_runtime_reachable_nonterminal_blocks(self):
        counts = count_abcd_continue_candidates(sample_conversation())

        self.assertEqual(counts["candidate_continue_blocks"], 3)
        self.assertEqual(counts["customer_fragments_merged"], 1)
        self.assertEqual(counts["terminal_courtesy_blocks"], 0)
        self.assertTrue(counts["causal_cut_applied"])

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
                "resolved_goals": ["learn the refund status"],
                "unresolved_goals": ["learn the approximate refund completion time"],
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

    def test_generation_quarantines_incomplete_business_goal_partition(self):
        case = build_abcd_cases(sample_conversation())[2]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "How much longer will it take?",
                "resolved_goals": ["refund is processing"],
                "unresolved_goals": ["refund completion time"],
                "evidence": [],
            }
        )

        row = generate_abcd_one(client, case)

        self.assertEqual(row["status"], "quarantined")
        self.assertIn("business_goal_partition_mismatch", row["quality_issues"])

    def test_generation_quarantines_duplicate_business_goal_label(self):
        case = build_abcd_cases(sample_conversation())[2]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "How much longer will it take?",
                "resolved_goals": ["learn the refund status"],
                "unresolved_goals": [
                    "learn the approximate refund completion time",
                    "learn the approximate refund completion time",
                ],
                "evidence": [],
            }
        )

        row = generate_abcd_one(client, case)

        self.assertEqual(row["status"], "quarantined")
        self.assertIn("business_goal_partition_mismatch", row["quality_issues"])

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

    def test_generation_rejects_complete_communication_for_external_continue_case(self):
        case = build_abcd_cases(sample_conversation())[2]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "complete",
                "response": "How much longer will it take?",
                "resolved_goals": [],
                "unresolved_goals": ["refund completion time"],
                "evidence": [],
            }
        )

        row = generate_abcd_one(client, case)

        self.assertEqual(row["status"], "quarantined")
        self.assertIn(
            "continue_with_complete_communication", row["quality_issues"]
        )

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

    def test_resume_alignment_rejects_stale_source_prefix(self):
        case = build_abcd_cases(sample_conversation())[2]
        client = FakeDeepSeekClient(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "How much longer will it take?",
                "resolved_goals": ["learn the refund status"],
                "unresolved_goals": ["learn the approximate refund completion time"],
                "evidence": [],
            }
        )
        row = generate_abcd_one(client, case)
        stale_case = json.loads(json.dumps(case))
        stale_case["student_messages"][-1]["content"] = "A changed Agent prefix."

        with self.assertRaisesRegex(ValueError, "resume source prefix drift"):
            validate_abcd_resume_alignment(
                row, stale_case, expected_model="deepseek-v4-flash"
            )


if __name__ == "__main__":
    unittest.main()
