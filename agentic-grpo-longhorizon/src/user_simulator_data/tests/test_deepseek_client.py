from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.user_simulator_data.deepseek_client import DeepSeekClient


class DeepSeekClientTests(unittest.TestCase):
    def test_requires_runtime_api_key_without_echoing_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                DeepSeekClient.from_environment()

    def test_request_uses_v4_flash_non_thinking_json_contract_with_sampling(self):
        captured = {}

        def fake_transport(url, headers, payload, timeout):
            captured.update(
                url=url, headers=headers, payload=payload, timeout=timeout
            )
            return {
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "continue",
                                    "is_over": False,
                                    "termination_reason": "continue",
                                    "goal_status": "in_progress",
                                    "communication_status": "partial",
                                    "response": "What is the total?",
                                    "resolved_goals": [],
                                    "unresolved_goals": ["total"],
                                    "evidence": [],
                                }
                            ),
                            "reasoning_content": "private reasoning",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

        client = DeepSeekClient(api_key="runtime-secret", transport=fake_transport)
        result = client.complete_json([{"role": "user", "content": "json please"}])

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-v4-flash")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured["payload"])
        self.assertEqual(captured["payload"]["max_tokens"], 1600)
        self.assertEqual(captured["payload"]["temperature"], 0.3)
        self.assertEqual(captured["payload"]["top_p"], 0.9)
        self.assertNotIn("reasoning_content", result)
        self.assertNotIn("runtime-secret", json.dumps(result))
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertEqual(
            result["request_config"],
            {
                "thinking": False,
                "max_tokens": 1600,
                "temperature": 0.3,
                "top_p": 0.9,
                "reasoning_effort": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
