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


if __name__ == "__main__":
    unittest.main()

