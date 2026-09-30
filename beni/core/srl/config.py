from __future__ import annotations

import datetime
import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List, Union, Type

from beni.utils import config as cfg


def trl_config_kwargs(
    config_cls: Type,
    data: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Keep only constructor kwargs accepted by the installed TRL config class.

    Sebeni YAML still exposes 0.x names such as ``max_prompt_length`` and
    ``warmup_ratio``. TRL 1.x dropped those on ``GRPOConfig`` / ``DPOConfig``.
    ``aliases`` maps a dropped Sebeni name onto a still-valid TRL field when
    that field is not already set (DPO ``max_prompt_length`` → ``max_length``).
    When ``report_to`` is ``trackio``, ``project`` is set to ``project_name``
    so Hugging Face's TrackioCallback logs into that project instead of the
    default ``huggingface``.
    """
    params = inspect.signature(config_cls.__init__).parameters
    allowed = set()
    var_keyword = False
    for name, param in params.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword = True
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        allowed.add(name)
    out = dict(data)
    for src, dest in (aliases or {}).items():
        if src not in out:
            continue
        if src not in allowed and dest in allowed:
            out.setdefault(dest, out[src])
        if src not in allowed:
            out.pop(src, None)
    if var_keyword:
        out = dict(out)
    else:
        out = {key: value for key, value in out.items() if key in allowed}
    if str(out.get("report_to") or "").strip().lower() == "trackio" and project_name:
        if var_keyword or "project" in allowed:
            out["project"] = project_name
        if run_name is None:
            run_name = f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if run_name and (var_keyword or "run_name" in allowed):
            out["run_name"] = run_name
    return out


def _overlay_dataclass(instance, data: Dict[str, Any]):
    """Apply a (possibly partial) dict onto an existing dataclass instance."""
    if not data:
        return instance
    valid = {f.name for f in fields(instance)}
    for key, val in data.items():
        if key not in valid:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(val, dict):
            setattr(instance, key, _overlay_dataclass(current, val))
        else:
            setattr(instance, key, val)
    post = getattr(instance, "__post_init__", None)
    if callable(post):
        post()
    return instance


@dataclass
class ModelConfig:
    """Configuration for the base model, reference model, and LoRA/Quantization."""
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    ref_model_name: Optional[str] = None
    flax_model_name: Optional[str] = None
    
    # Quantization (BitsAndBytes)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA / PEFT
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_bias: str = "none"


@dataclass
class DabaXProcessorConfig:
    """Configuration for the DabaX morphological parser baselines."""
    runtime_dir: str = field(default_factory=lambda: str(cfg.get_workdir().runtime))
    data_dir: str = field(default_factory=lambda: str(cfg.DATA_DIR / "baselines"))
    default_tokenizer: str = "default"
    gram_baseline: str = "baseline.gram"
    dict_baseline: str = "baseline.dict"


@dataclass
class DataConfig:
    """Configuration for SebeniGrpo data loading and formatting.

    A run is multilingual when ``languages`` has more than one code, or when
    the source already tags rows with mixed ``lang`` values. ``default_lang``
    is only the fallback for unlabeled rows.
    """
    default_lang: Optional[str] = "bam"
    languages: Optional[List[str]] = None
    source: Optional[Union[str, List[str]]] = None
    scheme: str = "completion"  # completion | preference | online_group
    allow_missing_lang: bool = False
    auto_detect_lang: bool = False
    infer_lang_from_filename: bool = True
    known_langs: Optional[Iterable[str]] = None
    lowercase_lang: bool = True
    encoding: str = "utf-8-sig"
    strip_text: bool = True
    skip_empty: bool = True
    max_text_len: Optional[int] = None
    file_text_key: str = "text"
    file_lang_key: str = "lang"
    hf_config_name: Optional[str] = None
    hf_split: Optional[str] = "train"
    hf_text_key: Optional[str] = None
    hf_lang_key: Optional[str] = None
    hf_streaming: bool = False
    hf_kwargs: Optional[Dict[str, Any]] = None
    clear_on_load: bool = False
    eval_ratio: float = 0.1

    def __post_init__(self) -> None:
        from beni.core.language import parse_lang_codes

        if self.default_lang is not None:
            self.default_lang = self.default_lang.strip().lower()
        if self.languages is not None:
            from beni.core.language import resolve_scope

            parsed = parse_lang_codes(self.languages)
            self.languages = resolve_scope(parsed).languages if parsed else None
        if self.known_langs is not None:
            self.known_langs = {str(c).strip().lower() for c in self.known_langs}
        elif self.languages:
            self.known_langs = set(self.languages)


@dataclass
class GRPOTrainerConfig:
    """Hyperparameters mapping to trl.GRPOConfig / TrainingArguments.

    Override from YAML ``trainer:`` or CLI flags on ``sebeni train``
    (``--lr``, ``--batch-size``, ``--max-steps``, ``--beta``, ...).
    """
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    max_steps: int = 10
    num_train_epochs: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 2
    output_dir: str = field(default_factory=lambda: str(cfg.get_workdir().models))
    max_grad_norm: float = 0.1
    report_to: str = "trackio"
    beta: float = 0.1
    num_generations: int = 4
    num_iterations: int = 1
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    optim: str = "adamw_torch"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0

    # Hugging Face Hub Integration
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "end"
    hub_token: Optional[str] = None
    hub_private_repo: bool = True

    # Hardware / Device
    use_cpu: bool = False
    framework: str = "torch"  # torch | jax

    # Evaluation
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None

    def to_dict(self) -> dict:
        """Export to dictionary for easy unpacking into GRPOConfig."""
        data = {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "output_dir": self.output_dir,
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "beta": self.beta,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_iterations": self.num_iterations,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "optim": self.optim,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "use_cpu": self.use_cpu,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_strategy": self.hub_strategy,
            "hub_token": self.hub_token,
            "hub_private_repo": self.hub_private_repo,
        }
        if self.eval_strategy is not None:
            data["eval_strategy"] = self.eval_strategy
        if self.eval_steps is not None:
            data["eval_steps"] = self.eval_steps
        if self.num_generations:
            data["num_generations"] = self.num_generations
        return data


@dataclass
class DPOTrainerConfig:
    """Hyperparameters mapping to trl.DPOConfig (DPO policy-update plugin)."""
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_length: int = 2048
    max_steps: int = 10
    num_train_epochs: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    output_dir: str = field(default_factory=lambda: str(cfg.get_workdir().models))
    max_grad_norm: float = 0.1
    report_to: str = "trackio"
    beta: float = 0.1
    loss_type: str = "sigmoid"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    use_cpu: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private_repo: bool = True

    # Evaluation
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None

    def to_dict(self) -> dict:
        data = {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "use_cpu": self.use_cpu,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
        }
        if self.eval_strategy is not None:
            data["eval_strategy"] = self.eval_strategy
        if self.eval_steps is not None:
            data["eval_steps"] = self.eval_steps
        return data


@dataclass
class APOTrainerConfig:
    """APO as a TRL DPO loss_type plugin (apo_zero / apo_down)."""
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_length: int = 2048
    max_steps: int = 10
    num_train_epochs: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    output_dir: str = field(default_factory=lambda: str(cfg.get_workdir().models))
    max_grad_norm: float = 0.1
    report_to: str = "trackio"
    beta: float = 0.1
    loss_type: str = "apo_zero"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    use_cpu: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private_repo: bool = True

    # Evaluation
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None

    def to_dict(self) -> dict:
        data = {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "use_cpu": self.use_cpu,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
        }
        if self.eval_strategy is not None:
            data["eval_strategy"] = self.eval_strategy
        if self.eval_steps is not None:
            data["eval_steps"] = self.eval_steps
        return data


@dataclass
class DistillationConfig:
    """Configuration for morphotactic batch distillation using Distiller."""
    enabled: bool = True
    backend: str = "algorithmic"
    provider: Optional[str] = None  # deprecated alias for backend
    model: str = "gemini-2.5-flash"
    working_dir: Optional[str] = None
    batch_size: int = 10
    auto_update_baselines: bool = True
    tau: float = 0.5
    hitl: bool = False
    vertex: Optional[bool] = None
    base_url: Optional[str] = None
    gguf_path: Optional[str] = None
    n_ctx: int = 4096
    max_input_chars: int = 8000

    def __post_init__(self) -> None:
        if self.provider:
            self.backend = self.provider

    @property
    def selected_backend(self) -> str:
        return str(self.provider or self.backend or "algorithmic").lower()


@dataclass
class WordfreqConfig:
    """Raw text inputs for the DabaX frequency-map pipeline."""

    raw_inputs: Optional[Union[str, List[str]]] = None


@dataclass
class RewardConfig:
    """Configuration for reward weights and composition."""
    format_weight: float = 0.1
    morph_weight: float = 0.4
    rule_weight: float = 0.4
    lang_weight: float = 0.1
    enable_format_reward: bool = True
    enable_morph_reward: bool = True
    enable_rule_reward: bool = True
    enable_lang_reward: bool = True
    custom_reward_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class SFTTrainerConfig:
    """Hyperparameters for the SFT arm (TRL ``SFTTrainer``)."""

    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    max_steps: int = 10
    num_train_epochs: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    output_dir: str = field(default_factory=lambda: str(cfg.get_workdir().models))
    max_grad_norm: float = 0.1
    report_to: str = "trackio"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    use_cpu: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private_repo: bool = True

    # Evaluation
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None

    def to_dict(self) -> dict:
        data = {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "use_cpu": self.use_cpu,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
        }
        if self.eval_strategy is not None:
            data["eval_strategy"] = self.eval_strategy
        if self.eval_steps is not None:
            data["eval_steps"] = self.eval_steps
        return data


@dataclass
class ExperimentConfig:
    """Canonical experiment contract: frozen resources, one arm, held-out eval."""

    kveritas: bool = False
    kveritas_seal: bool = False
    freeze_resources: bool = True
    # ``packaged`` reads dataset_300_samples.jsonl. ``raw`` keeps raw/*.txt for non-experiment train.
    dataset: str = "raw"
    scope: Optional[str] = None
    preset: Optional[str] = None
    max_eval_rows: int = 1


@dataclass
class SafetyConfig:
    """SafetyGovernor settings on MasterConfig (default-on)."""
    enabled: bool = True
    require_language: bool = True
    refuse_mixed_language: bool = True
    max_mer: float = 1.0
    require_phi_improve: bool = True
    format_invalid_blocks_update: bool = True
    uncertainty_downweight: bool = True
    require_model_card: bool = True
    require_safety_snapshot: bool = True
    max_kl: Optional[float] = None

    def to_spec(self, tau: float = 0.5, kl_beta: float = 0.1):
        from beni.core.safety.spec import SafetySpec
        return SafetySpec(
            enabled=self.enabled,
            require_language=self.require_language,
            refuse_mixed_language=self.refuse_mixed_language,
            tau=tau,
            require_phi_improve=self.require_phi_improve,
            max_mer=self.max_mer,
            format_invalid_blocks_update=self.format_invalid_blocks_update,
            uncertainty_downweight=self.uncertainty_downweight,
            kl_beta=kl_beta,
            max_kl=self.max_kl,
            require_model_card=self.require_model_card,
            require_safety_snapshot=self.require_safety_snapshot,
        )


@dataclass
class MasterConfig:
    """Aggregated configuration for the entire pipeline."""
    project_name: str = "GRPO-Morphology-Advanced"
    algorithm: str = "grpo"
    working_dir: Optional[str] = None
    
    model: ModelConfig = field(default_factory=ModelConfig)
    processor: DabaXProcessorConfig = field(default_factory=DabaXProcessorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: GRPOTrainerConfig = field(default_factory=GRPOTrainerConfig)
    dpo: DPOTrainerConfig = field(default_factory=DPOTrainerConfig)
    apo: APOTrainerConfig = field(default_factory=APOTrainerConfig)
    sft: SFTTrainerConfig = field(default_factory=SFTTrainerConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    wordfreq: WordfreqConfig = field(default_factory=WordfreqConfig)

    def apply_workdir(self, cli_path: Optional[Union[str, Path]] = None) -> Path:
        """Resolve and activate working_dir; rewrite default output/runtime paths."""
        config_file_dir = getattr(self, "_config_file_dir", None)
        wd = cfg.resolve_working_dir(
            cli_path=cli_path,
            yaml_path=self.working_dir,
            config_file_dir=config_file_dir,
        )
        self.working_dir = str(wd.root)
        if not self.distillation.working_dir:
            self.distillation.working_dir = str(wd.root)
        default_models = str(cfg.DEFAULT_WORKING_DIR / "models")
        if self.trainer.output_dir in ("", default_models) or self.trainer.output_dir == str(cfg.DEFAULT_WORKING_DIR / "models"):
            self.trainer.output_dir = str(wd.models)
        if self.processor.runtime_dir in ("", str(cfg.DEFAULT_WORKING_DIR / "runtime")):
            self.processor.runtime_dir = str(wd.runtime)
        for sibling in (self.dpo, self.apo, self.sft):
            if sibling.output_dir in ("", default_models, str(cfg.DEFAULT_WORKING_DIR / "models")):
                sibling.output_dir = str(wd.models)
        return wd.root

    def languages(self) -> List[str]:
        """Group codes for this run (explicit ``data.languages`` or default_lang)."""
        from beni.core.language import Language

        if self.data.languages:
            return Language.group_codes(self.data.languages)
        if self.data.default_lang:
            return Language.group_codes(self.data.default_lang)
        return ["bam"]

    def apply_cli_overrides(
        self,
        *,
        languages: Optional[List[str]] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        grad_accum: Optional[int] = None,
        max_steps: Optional[int] = None,
        epochs: Optional[float] = None,
        beta: Optional[float] = None,
        num_generations: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        temperature: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        weight_decay: Optional[float] = None,
        seed: Optional[int] = None,
        save_steps: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        optim: Optional[str] = None,
        use_cpu: Optional[bool] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        load_in_4bit: Optional[bool] = None,
        use_peft: Optional[bool] = None,
        bf16: Optional[bool] = None,
        fp16: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
        lr_scheduler_type: Optional[str] = None,
        hitl: Optional[bool] = None,
        kveritas: Optional[bool] = None,
        eval_ratio: Optional[float] = None,
        eval_steps: Optional[int] = None,
    ) -> "MasterConfig":
        """Apply non-None CLI hyperparameter overrides onto trainer / model / data."""
        if languages:
            from beni.core.language import resolve_scope

            resolved = resolve_scope(languages)
            if resolved.languages:
                self.data.languages = list(resolved.languages)
                self.data.known_langs = set(resolved.languages)
                self.experiment.scope = resolved.label
                if self.data.default_lang not in resolved.languages:
                    self.data.default_lang = resolved.languages[0]
        trainer = self.trainer
        if self.algorithm == "dpo":
            trainer = self.dpo
        elif self.algorithm == "apo":
            trainer = self.apo
        elif self.algorithm == "sft":
            trainer = self.sft
        mapping = {
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_steps": max_steps,
            "num_train_epochs": epochs,
            "beta": beta,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "seed": seed,
            "save_steps": save_steps,
            "max_prompt_length": max_prompt_length,
            "optim": optim,
            "use_cpu": use_cpu,
            "bf16": bf16,
            "fp16": fp16,
            "lr_scheduler_type": lr_scheduler_type,
            "eval_steps": eval_steps,
        }
        for attr, value in mapping.items():
            if value is not None and hasattr(trainer, attr):
                setattr(trainer, attr, value)
        if eval_steps is not None and hasattr(trainer, "eval_strategy"):
            trainer.eval_strategy = "steps"
        if epochs is not None and max_steps is None and hasattr(trainer, "max_steps"):
            trainer.max_steps = -1
        if num_generations is not None:
            self.trainer.num_generations = num_generations
        if max_completion_length is not None:
            self.trainer.max_completion_length = max_completion_length
        if temperature is not None:
            self.trainer.temperature = temperature
        if gradient_checkpointing is not None:
            self.trainer.gradient_checkpointing = gradient_checkpointing
        if lora_r is not None:
            self.model.lora_r = lora_r
        if lora_alpha is not None:
            self.model.lora_alpha = lora_alpha
        if lora_dropout is not None:
            self.model.lora_dropout = lora_dropout
        if load_in_4bit is not None:
            self.model.load_in_4bit = load_in_4bit
        if use_peft is not None:
            self.model.use_peft = use_peft
        if hitl is not None:
            self.distillation.hitl = hitl
        if kveritas is not None:
            self.experiment.kveritas = kveritas
        if eval_ratio is not None:
            self.data.eval_ratio = eval_ratio
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MasterConfig:
        """Construct MasterConfig from nested dictionary (partial YAML overlay)."""
        config = cls()
        if not data:
            return config
        top = {f.name for f in fields(config)}
        for key in top:
            if key not in data:
                continue
            val = data[key]
            current = getattr(config, key)
            if is_dataclass(current) and isinstance(val, dict):
                setattr(config, key, _overlay_dataclass(current, val))
            else:
                setattr(config, key, val)
        return config

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
    ) -> MasterConfig:
        """Load YAML/JSON into MasterConfig and activate working_dir."""
        path = Path(path)
        cfg.load_dotenv([Path.cwd() / ".env", path.parent / ".env"])
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".json"}:
            import json
            data = json.loads(text) or {}
        else:
            import yaml
            data = yaml.safe_load(text) or {}
        config = cls.from_dict(data)
        config._config_file_dir = str(path.parent.resolve())
        config.apply_workdir(cli_path=working_dir)
        cfg.load_dotenv([Path(config.working_dir) / ".env"])
        return config


@dataclass
class SRLGrpoPrompt:
    """System prompt for morphological JSON completions.

    Parameters
    ----------
    languages : list of str, optional
        ISO / group codes in this run. The prompt names them so a mixed
        batch can emit the correct per-sentence ``lang``.
    """

    languages: Optional[List[str]] = None

    def prompt(self) -> str:
        from beni.core.language import Language

        codes = Language.group_codes(self.languages) if self.languages else []
        if not codes:
            scope = "extremely low-resource Manding and related languages"
            lang_hint = "iso group code for THIS sentence (e.g. bam, mku, dtm)"
        elif len(codes) == 1:
            ident = Language.from_code(codes[0])
            scope = ident.name or ident.group_code
            lang_hint = f"iso group code (must be {codes[0]})"
        else:
            scope = ", ".join(codes)
            lang_hint = f"iso group code for THIS sentence (one of: {', '.join(codes)})"

        return f"""You are an expert computational linguist specializing in {scope}.
            Your task is to perform deep morphological analysis on the provided sentence.

            ### INSTRUCTIONS
            1. Decompose every token in the sentence into its constituent morphemes.
            2. If a morpheme can be further decomposed into sub-morphemes, represent it as a nested tree using the "morphemes" array.
            3. Assign each token a morphological stage (integer or parser label such as tokenizer / g.disamb).
            4. Set "lang" to the {lang_hint}. Do not mix languages inside one JSON object.
            5. Output ONLY a valid JSON object. Do not include markdown formatting, explanations, or conversational text.

            ### JSON SCHEMA
            {{
            "text": "string (the input sentence)",
            "lang": "string ({lang_hint})",
            "tokens": [
                {{
                "surface": "string (the exact word from the text)",
                "stage": "int or string (daba morphological stage)",
                "analyses": [
                    {{
                    "form": "string (normalized form)",
                    "ps": ["string (part of speech, e.g., 'n', 'v')"],
                    "gloss": "string (optional translation/gloss)",
                    "morphemes": [
                        {{
                        "form": "string (the morpheme)",
                        "ps": ["string (part of speech of morpheme)"],
                        "gloss": "string (optional)",
                        "morphemes": [ ... optional recursive array of sub-morphemes ... ]
                        }}
                    ]
                    }}
                ]
                }}
            ]
            }}
        """
