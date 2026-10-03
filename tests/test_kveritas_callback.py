"""Unit tests for KVeritasCallback, RewardManager recent rewards, and config overrides."""

import unittest
from unittest.mock import MagicMock, patch
import io
import sys
import types

# Ensure dependencies are available even in minimal environment
for mod in ["torch", "torch.nn", "torch.nn.functional", "trl", "numpy", "typer", "trackio"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

if "datasets" not in sys.modules:
    datasets_mod = types.ModuleType("datasets")
    class DummyDataset(list):
        @classmethod
        def from_dict(cls, data_dict):
            keys = list(data_dict.keys())
            n = len(data_dict[keys[0]]) if keys else 0
            rows = []
            for i in range(n):
                rows.append({k: data_dict[k][i] for k in keys})
            return cls(rows)
    datasets_mod.Dataset = DummyDataset
    sys.modules["datasets"] = datasets_mod

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    class DummyTrainerCallback:
        def __init__(self):
            pass
    transformers.TrainerCallback = DummyTrainerCallback
    sys.modules["transformers"] = transformers

from beni.core.srl.config import MasterConfig, DataConfig, GRPOTrainerConfig, DPOTrainerConfig, SFTTrainerConfig
from beni.core.compute.rewards import RewardManager
from beni.core.srl.grpo.callbacks import KVeritasCallback, emit_kveritas_line


class TestKVeritasCallback(unittest.TestCase):
    def test_emit_kveritas_line(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            emit_kveritas_line("phi", 0.85231, step=10)
            emit_kveritas_line("reward_total", 1.25, step=10)
        output = [line for line in buf.getvalue().strip().split("\n") if line]
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0], "KVERITAS_METRIC name=phi value=0.85231 step=10")
        self.assertEqual(output[1], "KVERITAS_METRIC name=reward_total value=1.25 step=10")

    def test_callback_on_log_rewards(self):
        reward_manager = MagicMock()
        reward_manager.recent_rewards = {
            "reward_total": 2.5,
            "reward_format": 0.5,
            "reward_morph": 1.0,
            "reward_rule": 0.8,
            "reward_lang": 0.2,
        }
        callback = KVeritasCallback(reward_manager=reward_manager)
        state = MagicMock()
        state.global_step = 5

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            callback.on_log(args=None, state=state, control=None, logs={"loss": 0.123})

        output = [line for line in buf.getvalue().strip().split("\n") if line]
        output_metrics = {line.split()[1].split("=")[1]: float(line.split()[2].split("=")[1]) for line in output}
        self.assertIn("reward_total", output_metrics)
        self.assertAlmostEqual(output_metrics["reward_total"], 2.5)
        self.assertAlmostEqual(output_metrics["reward_format"], 0.5)
        self.assertAlmostEqual(output_metrics["reward_morph"], 1.0)
        self.assertAlmostEqual(output_metrics["reward_rule"], 0.8)
        self.assertAlmostEqual(output_metrics["reward_lang"], 0.2)
        self.assertAlmostEqual(output_metrics["loss"], 0.123)

    def test_callback_on_evaluate(self):
        callback = KVeritasCallback(reward_manager=None)
        state = MagicMock()
        state.global_step = 20

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            callback.on_evaluate(args=None, state=state, control=None, metrics={"eval/loss": 0.45, "eval_phi": 0.78})

        output = [line for line in buf.getvalue().strip().split("\n") if line]
        output_metrics = {line.split()[1].split("=")[1]: float(line.split()[2].split("=")[1]) for line in output}
        self.assertIn("eval_loss", output_metrics)
        self.assertAlmostEqual(output_metrics["eval_loss"], 0.45)
        self.assertIn("eval_phi", output_metrics)
        self.assertAlmostEqual(output_metrics["eval_phi"], 0.78)

    def test_reward_manager_tracks_recent_rewards(self):
        rm = RewardManager()
        self.assertEqual(rm.recent_rewards, {})

        # Test reward_format populates recent_rewards['reward_format']
        rm.reward_format(['{"tokens": []}', '{"invalid": true}'])
        self.assertIn("reward_format", rm.recent_rewards)
        # default format_weight is 0.1 in RewardConfig, (0.1 + 0.0)/2 = 0.05
        self.assertAlmostEqual(rm.recent_rewards["reward_format"], 0.05)

        # Test clear() clears recent_rewards
        rm.clear()
        self.assertEqual(rm.recent_rewards, {})

    def test_config_eval_split_and_cli_overrides(self):
        mc = MasterConfig()
        self.assertEqual(mc.data.eval_ratio, 0.1)
        self.assertFalse(mc.experiment.kveritas)

        mc.apply_cli_overrides(kveritas=True, eval_ratio=0.25, eval_steps=50)
        self.assertTrue(mc.experiment.kveritas)
        self.assertEqual(mc.data.eval_ratio, 0.25)
        self.assertEqual(mc.trainer.eval_steps, 50)
        self.assertEqual(mc.trainer.eval_strategy, "steps")

        grpo_dict = mc.trainer.to_dict()
        self.assertEqual(grpo_dict["eval_steps"], 50)
        self.assertEqual(grpo_dict["eval_strategy"], "steps")


    def test_preference_dataset_conversational(self):
        from beni.data.datasets import rank_group_to_preference

        reference = {
            "text": "aw",
            "lang": "bam",
            "tokens": [{"surface": "aw", "stage": 1, "analyses": []}],
        }
        dataset = rank_group_to_preference(
            [{"text": "aw", "lang": "bam", "reference": reference}],
            scheme_prompt=True,
        )
        row = dataset[0]
        # Verify conversational structure required by TRL DPOTrainer
        self.assertIsInstance(row["prompt"], list)
        self.assertIsInstance(row["chosen"], list)
        self.assertIsInstance(row["rejected"], list)
        self.assertEqual(row["chosen"][0]["role"], "assistant")
        self.assertEqual(row["rejected"][0]["role"], "assistant")
        self.assertNotEqual(row["chosen"][0]["content"], "aw")
        self.assertNotEqual(row["chosen"][0]["content"], row["rejected"][0]["content"])


if __name__ == "__main__":
    unittest.main()
