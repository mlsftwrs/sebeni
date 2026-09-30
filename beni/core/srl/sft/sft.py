"""SFT policy-update plugin.

Labels come from the frozen distilled grammar and dictionary. Training starts
from the configured base model and does not read a GRPO, DPO, or APO checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from beni.core.srl.config import trl_config_kwargs
from beni.core.srl.plugin import AlignmentPlugin
from beni.core.srl.grpo.callbacks import TrackioMetricsCallback
from beni.utils import config as cfg


class SebeniSft(AlignmentPlugin):
    """TRL SFTTrainer over DabaX completion labels."""

    name = "sft"

    def extract_labels(self, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build SFT rows from frozen resources. Skips empty analyses."""
        from beni.core.srl.config import SRLGrpoPrompt
        from beni.data.datasets import build_dabax_reference

        prompt = SRLGrpoPrompt(languages=self.config.languages()).prompt()
        accepted: List[Dict[str, str]] = []
        invalid: List[Dict[str, str]] = []
        for row in records or []:
            text = str(row.get("text") or "")
            lang = str(row.get("lang") or row.get("language") or self.config.data.default_lang or "bam")
            if not text:
                continue
            reference = row.get("reference") or build_dabax_reference(text, lang)
            tokens = (reference or {}).get("tokens") if isinstance(reference, dict) else None
            if not reference or not tokens:
                invalid.append({"text": text, "lang": lang})
                continue
            completion = json.dumps(reference, ensure_ascii=False)
            accepted.append(
                {
                    "text": f"{prompt}\n\nUser: {text}\nAssistant: {completion}",
                    "lang": lang,
                    "completion": completion,
                }
            )
        root = Path(self.config.working_dir or cfg.get_workdir().root)
        dataset_path = root / "data" / "sft" / "train.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in accepted)
        dataset_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        report = {
            "n_accepted": len(accepted),
            "n_invalid": len(invalid),
            "dataset_sha256": digest,
            "dataset": str(dataset_path),
            "invalid_sample": invalid[:8],
        }
        report_path = root / "exp" / "sft_label_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if not accepted:
            raise RuntimeError("SFT label extraction produced no rows from the frozen resources.")
        return accepted

    def train(
        self,
        data,
        project_name: Optional[str] = None,
        run_distillation_first: bool = False,
        extra_callbacks: Optional[List] = None,
    ):
        del run_distillation_first
        import inspect

        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        records = [data[i] for i in range(len(data))] if hasattr(data, "column_names") else list(data or [])
        labeled = self.extract_labels(records)
        dataset = Dataset.from_list([{"text": row["text"]} for row in labeled])
        train_dataset = dataset
        eval_dataset = None
        eval_ratio = getattr(self.config.data, "eval_ratio", 0.0) or 0.0
        if eval_ratio > 0.0 and hasattr(dataset, "train_test_split") and len(dataset) > 1:
            split = dataset.train_test_split(test_size=eval_ratio, seed=self.config.sft.seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]

        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        sft_dict = {**self.config.sft.to_dict(), "dataset_text_field": "text"}
        if eval_dataset is not None:
            if "eval_strategy" not in sft_dict or not sft_dict["eval_strategy"]:
                sft_dict["eval_strategy"] = "steps"
            if "eval_steps" not in sft_dict or not sft_dict["eval_steps"]:
                sft_dict["eval_steps"] = max(1, sft_dict.get("logging_steps", 10))

        args = SFTConfig(
            **trl_config_kwargs(
                SFTConfig,
                sft_dict,
                project_name=project_name,
            )
        )
        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if getattr(self.config.experiment, "kveritas", False):
            from beni.core.srl.grpo.callbacks import KVeritasCallback
            callbacks.append(KVeritasCallback(reward_manager=self.reward_manager, config=self.config, eval_dataset=eval_dataset))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": self.model,
            "args": args,
            "train_dataset": train_dataset,
            "callbacks": callbacks,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        if "processing_class" in trainer_sig:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_sig:
            trainer_kwargs["tokenizer"] = self.tokenizer
        if "dataset_text_field" in trainer_sig:
            trainer_kwargs["dataset_text_field"] = "text"
        self.trainer = SFTTrainer(**trainer_kwargs)
        self._wire_pre_update_hooks(self.trainer)
        self._init_trackio(project_name)
        print("Starting Sebeni SFT Training Loop...")
        result = self.trainer.train()
        self.save_model(self.config.sft.output_dir or self.config.trainer.output_dir)
        if self.config.sft.push_to_hub:
            self.push_to_hub(
                repo_id=self.config.sft.hub_model_id,
                token=self.config.sft.hub_token,
                private=self.config.sft.hub_private_repo,
            )
        self._finish_trackio()
        return result
