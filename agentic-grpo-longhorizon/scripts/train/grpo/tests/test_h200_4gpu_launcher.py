from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


class H200LauncherContractTest(unittest.TestCase):
    # Each launcher is identified by its file name. ``setUp`` selects whichever
    # entry the dynamic ``glob`` discovers and the test cases verify the matching
    # CONFIG_NAME pair below.
    LAUNCHER_TO_CONFIG = {
        "run_vanilla_h200_4gpu.sh": "vanilla_grpo",
        "run_turn_ppo_h200_4gpu.sh": "turn_level_reward",
        "run_turn_ppo_strict_h200_4gpu.sh": "turn_ppo_strict",
        "run_hybrid_advantage_h200_4gpu.sh": "hybrid_advantage",
        "run_user_simulator_judge_h200_4gpu.sh": "user_simulator_judge",
    }

    def setUp(self) -> None:
        launchers = sorted(SCRIPT_DIR.glob("run_*_h200_4gpu.sh"))
        expected = set(self.LAUNCHER_TO_CONFIG)
        actual = {launcher.name for launcher in launchers}
        self.assertEqual(
            actual,
            expected,
            f"Unexpected launcher set: {actual ^ expected}",
        )
        self.launcher = launchers[0]  # stable order (alphabetical)

    def run_launcher(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({"AGENTIC_RL_DRY_RUN": "1", "AGENTIC_RL_RUN_TAG": "contract-test"})
        env.update(overrides)
        return subprocess.run(
            ["bash", str(self.launcher)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_branch_exposes_one_safe_dry_run_launcher(self) -> None:
        completed = self.run_launcher()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout

        self.assertIn("USER_SIM_CUDA_DEVICES=0", output)
        self.assertIn("REUSE_USER_SIMULATOR=0", output)
        self.assertIn("TRAIN_CUDA_DEVICES=1,2,3", output)
        self.assertIn("TRAIN_GPU_COUNT=3", output)
        self.assertIn("POLICY_MODEL=/data/xjz/model/qwen3-8b", output)
        self.assertIn("USER_SIM_MODEL=/data/xjz/model/qwen3-14b", output)
        self.assertIn("WANDB_ENTITY=jiezhengxing-aaaa", output)
        self.assertIn("WANDB_PROJECT=agentic-grpo-longhorizon", output)
        self.assertIn("data.max_prompt_length=12288", output)
        self.assertIn("data.max_response_length=16384", output)
        self.assertIn("actor_rollout_ref.rollout.max_model_len=32768", output)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=1", output)
        self.assertIn("trainer.n_gpus_per_node=3", output)
        self.assertIn("trainer.resume_mode=auto", output)
        self.assertIn("trainer.default_local_dir=", output)

        expected_config = self.LAUNCHER_TO_CONFIG[self.launcher.name]
        self.assertIn(f"CONFIG_NAME={expected_config}", output)
        if self.launcher.name == "run_hybrid_advantage_h200_4gpu.sh":
            self.assertIn("algorithm.hybrid_advantage.max_scoring_length=32768", output)
        if self.launcher.name == "run_user_simulator_judge_h200_4gpu.sh":
            self.assertIn("algorithm.judge_turn_reward.enabled=true", output)
            self.assertIn("algorithm.judge_turn_reward.turn_weight=0.05", output)
        if self.launcher.name == "run_turn_ppo_strict_h200_4gpu.sh":
            # The strict variant must surface the explicit overrides so reviewers
            # can confirm turn-shaped reward is forced off.
            self.assertIn(
                "actor_rollout_ref.actor.policy_loss.loss_mode=turn_ppo",
                output,
            )
            self.assertIn("algorithm.adv_estimator=turn_gae", output)
            self.assertIn("algorithm.use_kl_in_reward=false", output)
            self.assertIn("algorithm.turn_level_reward.enabled=false", output)

    def test_rejects_duplicate_or_overlapping_gpu_allocations(self) -> None:
        duplicate = self.run_launcher(TRAIN_CUDA_DEVICES="1,1,2")
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("duplicate", duplicate.stderr)

        overlapping = self.run_launcher(TRAIN_CUDA_DEVICES="0,1,2")
        self.assertNotEqual(overlapping.returncode, 0)
        self.assertIn("overlap", overlapping.stderr)

    def test_rejects_context_budget_larger_than_model_context(self) -> None:
        completed = self.run_launcher(
            AGENTIC_RL_MAX_PROMPT_LENGTH="20000",
            AGENTIC_RL_MAX_RESPONSE_LENGTH="20000",
            AGENTIC_RL_POLICY_MAX_MODEL_LEN="32768",
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("context budget", completed.stderr)


if __name__ == "__main__":
    unittest.main()
