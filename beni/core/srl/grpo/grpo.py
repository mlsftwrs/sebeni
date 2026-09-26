from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from beni.core.srl.config import (
    MasterConfig, ModelConfig, DataConfig, GRPOTrainerConfig,
    DistillationConfig, RewardConfig, trl_config_kwargs
)
from beni.core.srl.plugin import AlignmentPlugin
from beni.core.morphotactic.distil.distillation import Distiller
from beni.core.srl.grpo.callbacks import TrackioMetricsCallback


class SebeniGrpo(AlignmentPlugin):
    """GRPO policy-update plugin.

    Grammar and dictionary are frozen before this step. Training does not
    promote new resource checkpoints.
    """

    name = "grpo"

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        model_name: Optional[str] = None,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
        trainer_config: Optional[GRPOTrainerConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        super().__init__(
            config=config,
            model_name=model_name,
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            distillation_config=distillation_config,
            reward_config=reward_config,
        )

    def run_batch_distillation(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """One upstream SAMPG pass per language. Not called from ``train``."""
        if not self.config.distillation.enabled:
            print("Morphotactic distillation is disabled in config.")
            return {}

        from beni.core.srl.algorithm1 import distill_language, group_texts_by_language, records_text_langs

        default_lang = self.config.data.default_lang or "bam"
        lang_groups = group_texts_by_language(
            records_text_langs(sentences, default_lang),
            default_lang=default_lang,
        )
        allowed = set(self.config.languages()) if self.config.data.languages else None

        distillation_results = {}
        for group_code, texts in lang_groups.items():
            if allowed is not None and group_code not in allowed:
                continue
            print(f"Running Morphotactic Distillation for language '{group_code}' ({len(texts)} texts)...")
            try:
                distiller = Distiller(
                    lang_code=group_code,
                    backend=self.config.distillation.selected_backend,
                    model=self.config.distillation.model,
                    working_dir=self.config.distillation.working_dir,
                    vertex=self.config.distillation.vertex,
                    base_url=self.config.distillation.base_url,
                    gguf_path=self.config.distillation.gguf_path,
                    n_ctx=self.config.distillation.n_ctx,
                    max_input_chars=self.config.distillation.max_input_chars,
                )
                decision = distill_language(
                    texts,
                    distiller,
                    self.governor,
                    self.config.distillation.tau,
                    hitl=self.config.distillation.hitl,
                )
                distillation_results[group_code] = {
                    "gram_path": str(distiller.gram_path) if decision.allowed else None,
                    "dict_path": str(distiller.dict_path) if decision.allowed else None,
                    "reason": decision.reason,
                    "allowed": decision.allowed,
                }
            except Exception as exc:
                print(f"Morphotactic distillation for '{group_code}' skipped: {exc}")
                distillation_results[group_code] = {"error": str(exc)}

        return distillation_results

    def train(
        self,
        data: Union[List[Dict[str, Any]], Dataset],
        project_name: Optional[str] = None,
        run_distillation_first: bool = True,
        extra_callbacks: Optional[List] = None,
    ):
        """Execute GRPO against the already frozen grammar and dictionary."""
        del run_distillation_first
        raw_sentences = None
        if isinstance(data, list):
            raw_sentences = data
            dataset = self.format_dataset(raw_sentences)
        else:
            dataset = data

        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        grpo_args = GRPOConfig(
            **trl_config_kwargs(
                GRPOConfig,
                self.config.trainer.to_dict(),
                project_name=project_name,
            )
        )
        reward_funcs = self.reward_manager.get_reward_functions()

        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        trainer_kwargs = {
            "model": self.model,
            "reward_funcs": reward_funcs,
            "args": grpo_args,
            "train_dataset": dataset,
            "callbacks": callbacks,
        }
        if self.tokenizer is not None:
            trainer_kwargs["processing_class"] = self.tokenizer

        self.trainer = GRPOTrainer(**trainer_kwargs)
        if self.ref_model is not None and getattr(self.trainer, "ref_model", None) is None:
            try:
                self.trainer.ref_model = self.ref_model
            except Exception:
                pass

        self._wire_pre_update_hooks(self.trainer)
        self._init_trackio(project_name)

        print("Starting Sebeni GRPO Training Loop...")
        train_result = self.trainer.train()
        self.save_model(self.config.trainer.output_dir)

        if self.config.trainer.push_to_hub:
            self.push_to_hub(
                repo_id=self.config.trainer.hub_model_id,
                token=self.config.trainer.hub_token,
                private=self.config.trainer.hub_private_repo,
            )

        self._finish_trackio()
        return train_result
