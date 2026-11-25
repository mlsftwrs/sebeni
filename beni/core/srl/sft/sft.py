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
from beni.core.kveritas import active as kv_active
from beni.utils import config as cfg


class SebeniSft(AlignmentPlugin):
    """TRL SFTTrainer over DabaX completion labels."""

    name = "sft"
    signal_source = "targets"

    def extract_labels(self, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build SFT rows from frozen resources. Skips empty analyses.

        Each row is a ``prompt`` / ``completion`` pair from
        :func:`beni.core.srl.prompting.build_policy_example_parts`: the same chat
        prompt GRPO / DPO train on and ``analyze`` decodes with, so TRL computes
        the loss on the answer only. Targets the SafetyGovernor would refuse
        (invalid JSON, wrong or mixed language) are dropped (``gate_targets``).
        """
        from beni.core.srl.prompting import build_policy_example_parts
        from beni.data.datasets import build_dabax_reference

        languages = self.config.languages()
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
            target = json.dumps(reference, ensure_ascii=False)
            prompt, answer = build_policy_example_parts(text, target, self.tokenizer, languages)
            accepted.append(
                {
                    "prompt": prompt,
                    "completion": answer,
                    "text": prompt + answer,
                    "raw_text": text,
                    "lang": lang,
                    "target": target,
                }
            )
        accepted = self.gate_targets(accepted, completion_key="target")
        accepted = self.drop_overlength(
            accepted,
            prompt_of=lambda row: row["prompt"],
            answers_of=lambda row: [row["completion"]],
            max_length=getattr(self.config.sft, "max_length", None),
        ) if accepted else accepted
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
            "governor": self.target_gate_report,
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
        if self.model is None:
            self.load_models()
        labeled = self.extract_labels(records)
        dataset = Dataset.from_list(
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "raw_text": row.get("raw_text", ""),
                    "lang": row.get("lang", ""),
                }
                for row in labeled
            ]
        )
        train_dataset, eval_dataset = self.split_eval(dataset, seed=self.config.sft.seed)

        project_name = project_name or self.config.project_name
        sft_dict = self.config.sft.to_dict()
        # Hub uploads go through the SafetyGovernor (push_after_train / sebeni push), never
        # through HF Trainer's own checkpoint pushes, which would bypass the gate.
        sft_dict["push_to_hub"] = False
        if eval_dataset is not None:
            if "eval_strategy" not in sft_dict or not sft_dict["eval_strategy"]:
                sft_dict["eval_strategy"] = "steps"
            if "eval_steps" not in sft_dict or not sft_dict["eval_steps"]:
                max_st = sft_dict.get("max_steps", 100)
                sft_dict["eval_steps"] = max(1, max_st // 10) if max_st else max(1, sft_dict.get("logging_steps", 10))

        args = SFTConfig(
            **trl_config_kwargs(
                SFTConfig,
                sft_dict,
                project_name=project_name,
            )
        )
        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if kv_active(self.config):
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
        self.trainer = SFTTrainer(**trainer_kwargs)
        self._wire_pre_update_hooks(self.trainer)
        self._init_trackio(project_name)
        print("Starting Sebeni SFT Training Loop...")
        result = self.trainer.train()
        self.save_model(self.config.sft.output_dir or self.config.trainer.output_dir)
        self.push_after_train(self.config.sft)
        self._finish_trackio()
        return result
