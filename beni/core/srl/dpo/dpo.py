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
        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        args = DPOConfig(
            **trl_config_kwargs(
                DPOConfig,
                self.config.dpo.to_dict(),
                aliases={"max_prompt_length": "max_length"},
                project_name=project_name,
            )
        )
        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
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
