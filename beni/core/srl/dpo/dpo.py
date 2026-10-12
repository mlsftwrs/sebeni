"""DPO policy-update plugin (preference pairs ranked from SAMPG groups)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset

from beni.core.srl.config import trl_config_kwargs
from beni.core.srl.plugin import AlignmentPlugin
from beni.core.srl.grpo.callbacks import TrackioMetricsCallback
from beni.data.datasets import rank_group_to_preference


class SebeniDpo(AlignmentPlugin):
    """TRL DPOTrainer. Preference pairs are built from frozen grammar and dictionary."""

    name = "dpo"

    def _preference_dataset(self, data: Union[List[Dict[str, Any]], Dataset]) -> Dataset:
        if isinstance(data, Dataset):
            if "chosen" in data.column_names and "rejected" in data.column_names:
                return data
            records = [data[i] for i in range(len(data))]
        else:
            records = data
        if records and isinstance(records[0], dict) and "chosen" in records[0]:
            return self.data_loader.ds(records)
        return rank_group_to_preference(
            records, scheme_prompt=True, languages=self.config.languages()
        )

    def train(
        self,
        data,
        project_name: Optional[str] = None,
        run_distillation_first: bool = False,
        extra_callbacks: Optional[List] = None,
    ):
        from trl import DPOConfig, DPOTrainer

        self._preference_records = (
            [data[i] for i in range(len(data))] if isinstance(data, Dataset) else data
        )
        dataset = self._preference_dataset(self._preference_records)
        train_dataset = dataset
        eval_dataset = None
        eval_ratio = getattr(self.config.data, "eval_ratio", 0.0) or 0.0
        max_eval = getattr(self.config.experiment, "max_eval_rows", None)
        if (eval_ratio > 0.0 or (max_eval and max_eval > 0)) and hasattr(dataset, "train_test_split") and len(dataset) > 1:
            n_samples = len(dataset)
            if max_eval is not None and max_eval > 0:
                target_test = min(int(max_eval), max(1, n_samples - 1))
                if n_samples > 2 and target_test >= n_samples:
                    target_test = max(1, n_samples // 2)
                test_size = target_test
            else:
                test_size = eval_ratio

            split = dataset.train_test_split(test_size=test_size, seed=self.config.dpo.seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            if max_eval is not None and max_eval > 0 and len(eval_dataset) > max_eval:
                if hasattr(eval_dataset, "select"):
                    eval_dataset = eval_dataset.select(range(max_eval))
                else:
                    eval_dataset = eval_dataset[:max_eval]

        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        dpo_dict = self.config.dpo.to_dict()
        if eval_dataset is not None:
            if "eval_strategy" not in dpo_dict or not dpo_dict["eval_strategy"]:
                dpo_dict["eval_strategy"] = "steps"
            if "eval_steps" not in dpo_dict or not dpo_dict["eval_steps"]:
                max_st = dpo_dict.get("max_steps", 100)
                dpo_dict["eval_steps"] = max(1, max_st // 10) if max_st else max(1, dpo_dict.get("logging_steps", 10))

        # Memory guard for evaluation: default eval batch size to 1 to avoid OOM with large vocabularies (e.g. Gemma 4)
        if "per_device_eval_batch_size" not in dpo_dict or not dpo_dict["per_device_eval_batch_size"]:
            dpo_dict["per_device_eval_batch_size"] = min(int(dpo_dict.get("per_device_train_batch_size", 1) or 1), 1)
        if "eval_accumulation_steps" not in dpo_dict or not dpo_dict["eval_accumulation_steps"]:
            dpo_dict["eval_accumulation_steps"] = 1

        args = DPOConfig(
            **trl_config_kwargs(
                DPOConfig,
                dpo_dict,
                aliases={"max_prompt_length": "max_length"},
                project_name=project_name,
            )
        )
        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if getattr(self.config.experiment, "kveritas", False):
            from beni.core.srl.grpo.callbacks import KVeritasCallback
            callbacks.append(KVeritasCallback(reward_manager=self.reward_manager, config=self.config, eval_dataset=eval_dataset))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        trainer_kwargs = {
            "model": self.model,
            "ref_model": self.ref_model,
            "args": args,
            "train_dataset": train_dataset,
            "processing_class": self.tokenizer,
            "callbacks": callbacks,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        self.trainer = DPOTrainer(**trainer_kwargs)
        self._wire_pre_update_hooks(self.trainer)
        self._init_trackio(project_name)
        print("Starting Sebeni DPO Training Loop...")
        result = self.trainer.train()
        self.save_model(self.config.dpo.output_dir or self.config.trainer.output_dir)
        if self.config.dpo.push_to_hub:
            self.push_to_hub(
                repo_id=self.config.dpo.hub_model_id,
                token=self.config.dpo.hub_token,
                private=self.config.dpo.hub_private_repo,
            )
        self._finish_trackio()
        return result
