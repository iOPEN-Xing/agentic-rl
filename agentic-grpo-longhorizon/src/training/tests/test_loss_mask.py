from __future__ import annotations

import unittest

from src.training.loss_mask import select_assistant_indices


class LossMaskTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "simulate user"},
            {"role": "user", "content": "How can I help?"},
            {"role": "assistant", "content": "I need a flight."},
            {"role": "user", "content": "Which date?"},
            {"role": "assistant", "content": "May 20."},
        ]

    def test_default_mode_keeps_existing_all_assistant_behavior(self):
        self.assertEqual(
            select_assistant_indices(self.messages, "all_assistant"), [2, 4]
        )

    def test_last_assistant_mode_only_supervises_teacher_target(self):
        self.assertEqual(
            select_assistant_indices(self.messages, "last_assistant"), [4]
        )

    def test_unknown_mode_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "loss_mask_mode"):
            select_assistant_indices(self.messages, "everything")


if __name__ == "__main__":
    unittest.main()
