"""DPO policy-update plugin (preference pairs ranked from SAMPG groups)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset

from beni.core.compute.rewards import RewardManager
from beni.core.srl.config import trl_config_kwargs
from beni.core.srl.plugin import AlignmentPlugin
from beni.core.srl.grpo.callbacks import TrackioMetricsCallback
from beni.data.datasets import rank_group_to_preference


class SebeniDpo(AlignmentPlugin):
    """TRL DPOTrainer. Preference pairs are built from frozen grammar and dictionary."""

    name = "dpo"
    signal_source = "targets"

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

    def _rendered_conversation(self, row: Dict[str, Any], answer_key: str) -> str:
        """Prompt plus one answer, rendered with the chat template when the row is conversational."""
        prompt, answer = row.get("prompt"), row.get(answer_key)
        if (
            isinstance(prompt, list)
            and isinstance(answer, list)
            and self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None)
        ):
            return self.tokenizer.apply_chat_template(prompt + answer, tokenize=False)
        return RewardManager.completion_text(prompt) + RewardManager.completion_text(answer)

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
        # SafetyGovernor: the gradient pulls θ towards "chosen", so refuse invalid,
        # wrong-language or mixed-language chosen answers before training.
        rows = [dataset[i] for i in range(len(dataset))]
        kept = self.gate_targets(rows, completion_key="chosen", lang_key="language")
        if not kept:
            raise RuntimeError(f"SafetyGovernor refused every preference pair: {self.target_gate_report}")
        if self.model is None:
            self.load_models()
        kept = self.drop_overlength(
            kept,
            # Count the fully rendered conversation (template suffix included), as TRL tokenizes it.
            prompt_of=lambda row: "",
            answers_of=lambda row: [self._rendered_conversation(row, "chosen"), self._rendered_conversation(row, "rejected")],
            max_length=getattr(self.config.dpo, "max_length", None),
        )
        if len(kept) != len(rows):
            dataset = Dataset.from_list(kept)
        train_dataset, eval_dataset = self.split_eval(dataset, seed=self.config.dpo.seed)

        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        dpo_dict = self.config.dpo.to_dict()
        # Hub uploads go through the SafetyGovernor (push_after_train / sebeni push), never
        # through HF Trainer's own checkpoint pushes, which would bypass the gate.
        dpo_dict["push_to_hub"] = False
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
        self.push_after_train(self.config.dpo)
        self._finish_trackio()
        return result
