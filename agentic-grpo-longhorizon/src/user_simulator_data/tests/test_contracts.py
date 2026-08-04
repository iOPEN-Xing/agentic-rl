from __future__ import annotations

import unittest

from src.user_simulator_data.contracts import (
    ContractError,
    TeacherDecision,
    build_sft_record,
    parse_teacher_decision,
    validate_response_constraints,
    validate_observable_grounding_text,
    validate_teacher_decision,
)


class ContractTests(unittest.TestCase):
    def test_parse_teacher_decision_accepts_fenced_json_and_normalizes_response(self):
        raw = """```json
    {
      "decision": "stop",
      "is_over": true,
      "termination_reason": "goal_satisfied",
      "goal_status": "satisfied",
      "communication_status": "complete",
      "response": "  ###STOP###  ",
      "resolved_goals": ["cancel both reservations"],
      "unresolved_goals": [],
      "evidence": [{"turn_index": 8, "fact": "Agent confirmed both cancellations."}]
    }
    ```"""

        decision = parse_teacher_decision(raw)

        self.assertEqual(decision.response, "###STOP###")
        self.assertTrue(decision.is_over)
        self.assertEqual(decision.termination_reason, "goal_satisfied")


    def test_validate_teacher_decision_rejects_inconsistent_stop_contract(self):
        cases = [
        (
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "###STOP###",
                "resolved_goals": [],
                "unresolved_goals": ["total price"],
                "evidence": [],
            },
            "STOP response requires decision=stop",
        ),
        (
            {
                "decision": "stop",
                "is_over": True,
                "termination_reason": "goal_satisfied",
                "goal_status": "satisfied",
                "communication_status": "partial",
                "response": "Thanks",
                "resolved_goals": [],
                "unresolved_goals": ["total price"],
                "evidence": [],
            },
            "terminal response must be exactly ###STOP###",
        ),
        ]
        for payload, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                decision = TeacherDecision.from_mapping(payload)

                with self.assertRaisesRegex(ContractError, expected_error):
                    validate_teacher_decision(decision)


    def test_build_sft_record_keeps_labels_in_metadata_but_plain_text_as_target(self):
        decision = TeacherDecision.from_mapping(
        {
            "decision": "continue",
            "is_over": False,
            "termination_reason": "continue",
            "goal_status": "in_progress",
            "communication_status": "partial",
            "response": "Please also tell me the exact total cost.",
            "resolved_goals": ["two cancellations"],
            "unresolved_goals": ["total cost"],
            "evidence": [{"turn_index": 6, "fact": "Cancellations were confirmed."}],
        }
    )
        source_case = {
        "case_id": "airline-0034-after-cancel",
        "task_id": 34,
        "student_messages": [
            {"role": "system", "content": "simulate the customer"},
            {"role": "user", "content": "Both reservations are cancelled."},
        ],
    }

        record = build_sft_record(source_case, decision, prompt_version="v1")

        self.assertEqual(
            record["messages"][-1],
            {
                "role": "assistant",
                "content": "Please also tell me the exact total cost.",
            },
        )
        self.assertNotIn("goal_status", record["messages"][-1]["content"])
        self.assertEqual(record["metadata"]["goal_status"], "in_progress")
        self.assertEqual(record["metadata"]["unresolved_goals"], ["total cost"])
        self.assertEqual(record["metadata"]["prompt_version"], "v1")


    def test_validation_flags_privileged_entity_leak(self):
        decision = TeacherDecision.from_mapping(
        {
            "decision": "continue",
            "is_over": False,
            "termination_reason": "continue",
            "goal_status": "in_progress",
            "communication_status": "partial",
            "response": "Please use reservation ZXCV12.",
            "resolved_goals": [],
            "unresolved_goals": ["find reservation"],
            "evidence": [],
        }
    )

        issues = validate_teacher_decision(
        decision,
        scenario="I do not remember my reservation id.",
        observable_history=[{"role": "agent", "content": "What is the id?"}],
        privileged_entities={"ZXCV12"},
    )

        self.assertIn("privileged_entity_leak:ZXCV12", issues)

    def test_grounding_rejects_invented_and_example_identifiers(self):
        scenario = "You are mia_li_3668 and do not remember the reservation ID."
        history = [
            {
                "role": "agent",
                "content": "Please provide a six-character code like ABC123.",
            }
        ]

        invented = validate_observable_grounding_text(
            "My reservation ID is MADEUP7.",
            scenario=scenario,
            observable_history=history,
        )
        copied_example = validate_observable_grounding_text(
            "My reservation ID is ABC123.",
            scenario=scenario,
            observable_history=history,
        )
        wrong_type = validate_observable_grounding_text(
            "My reservation ID is mia_li_3668.",
            scenario=scenario,
            observable_history=history,
        )

        self.assertIn("unsupported_reservation_id:MADEUP7", invented)
        self.assertIn(
            "agent_example_copied_as_reservation_id:ABC123", copied_example
        )
        self.assertIn(
            "user_id_mislabeled_as_reservation_id:mia_li_3668", wrong_type
        )

    def test_grounding_allows_observable_identifier_and_missing_id_reply(self):
        scenario = "You are Mia and your reservation is VA5SGQ."
        history = [{"role": "agent", "content": "What is the reservation ID?"}]

        supported = validate_observable_grounding_text(
            "My reservation ID is VA5SGQ.",
            scenario=scenario,
            observable_history=history,
        )
        unavailable = validate_observable_grounding_text(
            "The reservation ID is unavailable. Can you use my profile?",
            scenario=scenario,
            observable_history=history,
        )

        self.assertEqual(supported, [])
        self.assertEqual(unavailable, [])

    def test_grounding_rejects_unsupported_dob_payment_id_and_placeholders(self):
        scenario = "Book a flight for Aarav and use a payment method from the profile."
        history = [{"role": "agent", "content": "Please provide the missing fields."}]

        issues = validate_observable_grounding_text(
            "My DOB is 1990-04-12, my payment ID is 987654321, the date is [date], "
            "and the reservation ID for this trip is ... let me check.",
            scenario=scenario,
            observable_history=history,
        )

        self.assertIn("unsupported_date_of_birth_year:1990", issues)
        self.assertIn("unsupported_payment_id:987654321", issues)
        self.assertIn("template_placeholder_in_response", issues)
        self.assertIn("placeholder_identifier:untyped", issues)

    def test_grounding_checks_airport_code_and_city_aliases(self):
        unsupported = validate_observable_grounding_text(
            "I'm flying to Los Angeles (LAX).",
            scenario="Change my May 17 flight from JFK.",
        )
        supported_alias = validate_observable_grounding_text(
            "I need to leave from Newark.",
            scenario="Change my flight from EWR.",
        )
        person_name = validate_observable_grounding_text(
            "My name is Mia Li.",
            scenario="You are mia_li_3668.",
        )

        self.assertIn("unsupported_airport_or_city:LAX", unsupported)
        self.assertEqual(supported_alias, [])
        self.assertEqual(person_name, [])

    def test_case_constraints_gate_semantics_without_forcing_one_wording(self):
        decision = TeacherDecision.from_mapping(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "none",
                "response": "I need a flight to Seattle in economy with three bags.",
                "resolved_goals": [],
                "unresolved_goals": ["book flight"],
                "evidence": [],
            }
        )

        issues = validate_response_constraints(
            decision,
            {
                "must_include_any_of_each": [["flight", "book"], ["Seattle", "SEA"]],
                "must_not_include": ["economy", "bag"],
                "max_chars": 120,
            },
        )

        self.assertNotIn("missing_required_response_group:0", issues)
        self.assertNotIn("missing_required_response_group:1", issues)
        self.assertIn("forbidden_response_content:0", issues)
        self.assertIn("forbidden_response_content:1", issues)

    def test_build_sft_record_requires_agent_input_before_teacher_target(self):
        decision = TeacherDecision.from_mapping(
            {
                "decision": "continue",
                "is_over": False,
                "termination_reason": "continue",
                "goal_status": "in_progress",
                "communication_status": "partial",
                "response": "Which date?",
                "resolved_goals": [],
                "unresolved_goals": ["date"],
                "evidence": [],
            }
        )
        invalid_case = {
            "case_id": "two-customer-turns",
            "task_id": 0,
            "student_messages": [
                {"role": "system", "content": "simulate"},
                {"role": "assistant", "content": "I need a flight."},
            ],
        }

        with self.assertRaisesRegex(ContractError, "must end with role=user"):
            build_sft_record(invalid_case, decision, prompt_version="v1")


if __name__ == "__main__":
    unittest.main()
