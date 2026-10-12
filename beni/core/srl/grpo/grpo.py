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

            split = dataset.train_test_split(test_size=test_size, seed=self.config.trainer.seed)
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
        grpo_dict = self.config.trainer.to_dict()

        model_name_lower = (self.config.model.model_name or "").lower()
        is_gemma4 = "gemma-4" in model_name_lower or "gemma4" in model_name_lower
        if is_gemma4 and grpo_dict.get("use_vllm"):
            print(
                "[GRPO Notice] Disabling vLLM for Gemma 4 architecture as vLLM colocate mode "
                "encounters CUDA kernel / process group incompatibilities with PLE/custom KV cache. "
                "Falling back to standard Hugging Face generation."
            )
            grpo_dict["use_vllm"] = False

        if grpo_dict.get("use_vllm"):
            import torch
            vllm_available = False
            try:
                import vllm  # noqa: F401
                vllm_available = True
            except ImportError:
                vllm_available = False

            if not torch.cuda.is_available() or not vllm_available:
                print(
                    "[GRPO Notice] use_vllm was enabled, but CUDA or vLLM is not installed. "
                    "Falling back to standard Hugging Face generation. "
                    "Install vLLM (pip install vllm) on a CUDA device to enable ultra-fast rollout generation."
                )
                grpo_dict["use_vllm"] = False
            else:
                print(
                    f"[GRPO] vLLM rollout acceleration ENABLED (mode: {grpo_dict.get('vllm_mode', 'colocate')}, "
                    f"vllm_gpu_memory_utilization: {grpo_dict.get('vllm_gpu_memory_utilization', 0.3)})."
                )

        if eval_dataset is not None:
            if "eval_strategy" not in grpo_dict or not grpo_dict["eval_strategy"]:
                grpo_dict["eval_strategy"] = "steps"
            if "eval_steps" not in grpo_dict or not grpo_dict["eval_steps"]:
                max_st = grpo_dict.get("max_steps", 100)
                grpo_dict["eval_steps"] = max(1, max_st // 10) if max_st else max(1, grpo_dict.get("logging_steps", 10))

        if "per_device_eval_batch_size" not in grpo_dict or not grpo_dict["per_device_eval_batch_size"]:
            grpo_dict["per_device_eval_batch_size"] = min(int(grpo_dict.get("per_device_train_batch_size", 1) or 1), 1)

        # Guard: effective batch size must be divisible by num_generations for GRPO group advantage calculation
        num_gen = grpo_dict.get("num_generations", 4) or 4
        per_dev_bs = grpo_dict.get("per_device_train_batch_size", 1) or 1
        grad_accum = grpo_dict.get("gradient_accumulation_steps", 1) or 1
        if (per_dev_bs * grad_accum) % num_gen != 0:
            needed_accum = max(1, (num_gen + per_dev_bs - 1) // per_dev_bs)
            while (per_dev_bs * needed_accum) % num_gen != 0:
                needed_accum += 1
            print(
                f"[GRPO Guard] Adjusted gradient_accumulation_steps from {grad_accum} to {needed_accum} "
                f"so effective generation batch size ({per_dev_bs * needed_accum}) is divisible by num_generations ({num_gen})."
            )
            grpo_dict["gradient_accumulation_steps"] = needed_accum

        grpo_args = GRPOConfig(
            **trl_config_kwargs(
                GRPOConfig,
                grpo_dict,
                project_name=project_name,
            )
        )
        reward_funcs = self.reward_manager.get_reward_functions()

        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if getattr(self.config.experiment, "kveritas", False):
            from beni.core.srl.grpo.callbacks import KVeritasCallback
            callbacks.append(KVeritasCallback(reward_manager=self.reward_manager, config=self.config, eval_dataset=eval_dataset))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        trainer_kwargs = {
            "model": self.model,
            "reward_funcs": reward_funcs,
            "args": grpo_args,
            "train_dataset": train_dataset,
            "callbacks": callbacks,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
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
