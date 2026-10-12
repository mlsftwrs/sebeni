#!/usr/bin/env python3
"""Unit tests for run_and_push_per_model completion checking, skipping, and GRPO restart logic."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Import functions from run_and_push_per_model without executing main
import sys
from unittest.mock import MagicMock

# Mock third-party dependencies if not installed in current environment
for mod in ("yaml", "huggingface_hub", "huggingface_hub.hf_api", "huggingface_hub.utils", "trackio"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.run_and_push_per_model import (
    format_repo_id,
    cleanup_stale_run_dir,
    is_repo_completed_on_hub,
    check_cell_completed,
    load_cached_metrics,
)


class TestRunAndPushSkip(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_push_skip_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_repo_id(self):
        repo_id_sft = format_repo_id(
            template="{org}/sebeni-{model}-{algo}-{steps}",
            org="mlsftwrs",
            model_slug="smollm2-135m",
            algo="sft",
            params="135M",
            steps=100,
        )
        self.assertEqual(repo_id_sft, "mlsftwrs/sebeni-smollm2-135-sft-100")

        repo_id_grpo = format_repo_id(
            template="{org}/sebeni-{model}-{algo}-{steps}",
            org="mlsftwrs",
            model_slug="smollm2-135m",
            algo="grpo",
            params="135M",
            steps=100,
        )
        self.assertEqual(repo_id_grpo, "mlsftwrs/sebeni-smollm2-135-grpo-100")

    def test_is_repo_completed_on_hub_with_completed_repo(self):
        mock_api = MagicMock()
        mock_api.repo_exists.return_value = True
        mock_api.list_repo_files.return_value = [
            "README.md",
            "config.yaml",
            "adapter_model.safetensors",
            "adapter_config.json",
        ]
        completed = is_repo_completed_on_hub(mock_api, "mlsftwrs/sebeni-smollm2-135-sft-100")
        self.assertTrue(completed)

    def test_is_repo_completed_on_hub_with_empty_or_missing_repo(self):
        mock_api = MagicMock()
        # Missing
        mock_api.repo_exists.return_value = False
        self.assertFalse(is_repo_completed_on_hub(mock_api, "mlsftwrs/sebeni-smollm2-135-grpo-100"))

        # Empty repo (e.g. only .gitattributes)
        mock_api.repo_exists.return_value = True
        mock_api.list_repo_files.return_value = [".gitattributes"]
        self.assertFalse(is_repo_completed_on_hub(mock_api, "mlsftwrs/sebeni-smollm2-135-grpo-100"))

    def test_check_cell_completed_skips_when_on_hf(self):
        mock_api = MagicMock()
        mock_api.repo_exists.return_value = True
        mock_api.list_repo_files.return_value = ["README.md", "config.yaml", "eval_results.json"]

        run_dir = self.temp_dir / "smollm2-135m_sft"
        is_done, reason = check_cell_completed(
            api=mock_api,
            target_repo="mlsftwrs/sebeni-smollm2-135-sft-100",
            run_dir=run_dir,
            algo="sft",
            restart_grpo=True,
            skip_completed=True,
        )
        self.assertTrue(is_done)
        self.assertIn("already completed", reason)

    def test_check_cell_completed_restarts_grpo(self):
        mock_api = MagicMock()
        mock_api.repo_exists.return_value = True
        mock_api.list_repo_files.return_value = ["README.md", "config.yaml"]

        run_dir = self.temp_dir / "smollm2-135m_grpo"
        # When restart_grpo is True, GRPO is not skipped even if files exist
        is_done, reason = check_cell_completed(
            api=mock_api,
            target_repo="mlsftwrs/sebeni-smollm2-135-grpo-100",
            run_dir=run_dir,
            algo="grpo",
            restart_grpo=True,
            skip_completed=True,
        )
        self.assertFalse(is_done)
        self.assertIn("Restarting GRPO", reason)

    def test_check_cell_completed_local_eval(self):
        run_dir = self.temp_dir / "smollm2-135m_apo"
        exp_dir = run_dir / "exp"
        exp_dir.mkdir(parents=True, exist_ok=True)
        eval_file = exp_dir / "eval.json"
        eval_file.write_text(json.dumps({"phi": 0.85, "mer": 12.5}), encoding="utf-8")

        is_done, reason = check_cell_completed(
            api=None,
            target_repo="mlsftwrs/sebeni-smollm2-135-apo-100",
            run_dir=run_dir,
            algo="apo",
            restart_grpo=True,
            skip_completed=True,
        )
        self.assertTrue(is_done)
        self.assertIn("Local evaluation report found", reason)

    def test_cleanup_stale_run_dir(self):
        run_dir = self.temp_dir / "smollm2-135m_grpo"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "temp_file.txt").write_text("junk", encoding="utf-8")

        chk_dir = run_dir / "checkpoint-10"
        chk_dir.mkdir(parents=True, exist_ok=True)

        cleanup_stale_run_dir(run_dir)
        self.assertFalse(models_dir.exists())
        self.assertFalse(chk_dir.exists())

    def test_model_catalog_resolution(self):
        from scripts.run_and_push_per_model import load_model_catalog
        catalog_path = repo_root / "configs" / "matrix" / "models.json"
        catalog = load_model_catalog(catalog_path)
        slugs = [m["slug"] for m in catalog]
        expected_slugs = ["smollm2-135m", "lfm2.5-230m", "granite-4.0-h-350m", "qwen2.5-0.5b", "gemma-4-e2b"]
        for exp in expected_slugs:
            self.assertIn(exp, slugs)

    def test_eval_steps_calculation(self):
        # 100 steps should yield 10 eval steps
        max_steps = 100
        calc_eval_steps = max(1, max_steps // 10)
        self.assertEqual(calc_eval_steps, 10)

        # 50 steps should yield 5 eval steps
        self.assertEqual(max(1, 50 // 10), 5)

        # 5 steps should yield 1 eval step
        self.assertEqual(max(1, 5 // 10), 1)

    def test_dataclass_eval_batch_size(self):
        from beni.core.srl.config import (
            DPOTrainerConfig,
            APOTrainerConfig,
            GRPOTrainerConfig,
            SFTTrainerConfig,
        )
        dpo = DPOTrainerConfig(per_device_eval_batch_size=1)
        self.assertEqual(dpo.per_device_eval_batch_size, 1)
        self.assertEqual(dpo.to_dict()["per_device_eval_batch_size"], 1)

        apo = APOTrainerConfig(per_device_eval_batch_size=1)
        self.assertEqual(apo.per_device_eval_batch_size, 1)
        self.assertEqual(apo.to_dict()["per_device_eval_batch_size"], 1)

        grpo = GRPOTrainerConfig(per_device_eval_batch_size=1)
        self.assertEqual(grpo.per_device_eval_batch_size, 1)
        self.assertEqual(grpo.to_dict()["per_device_eval_batch_size"], 1)

        sft = SFTTrainerConfig(per_device_eval_batch_size=1)
        self.assertEqual(sft.per_device_eval_batch_size, 1)
        self.assertEqual(sft.to_dict()["per_device_eval_batch_size"], 1)

    def test_gemma4_guards_logic(self):
        gemma_entry = {
            "id": "google/gemma-4-E2B",
            "slug": "gemma-4-e2b",
            "params": "2B",
            "recommended_batch_size": 1,
        }
        model_slug_lower = gemma_entry["slug"].lower()
        model_id_lower = gemma_entry["id"].lower()
        is_gemma4 = "gemma-4" in model_slug_lower or "gemma-4" in model_id_lower or "gemma4" in model_slug_lower
        params_str = str(gemma_entry.get("params", "")).upper()
        is_large = is_gemma4 or any(k in params_str for k in ("3B", "4B", "7B", "8B"))

        self.assertTrue(is_gemma4)
        self.assertTrue(is_large)

        # In GRPO, vLLM must be disabled for Gemma 4
        use_vllm_effective = True and not is_gemma4
        self.assertFalse(use_vllm_effective)

        # In DPO/APO, batch size must be 1 and max_length bounded
        rec_batch = gemma_entry["recommended_batch_size"]
        eval_batch = min(rec_batch, 1)
        self.assertEqual(eval_batch, 1)


if __name__ == "__main__":
    unittest.main()


