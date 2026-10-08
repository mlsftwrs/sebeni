"""Shared alignment plugin surface (load / save / push / generate).

GRPO, DPO, and APO implement ``train``; they share this dataclass-style wrapper
around Transformers + PEFT + Hub cards.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from beni.core.compute.rewards import RewardManager
from beni.core.hub.model_card import write_model_card
from beni.core.safety.governor import SafetyGovernor
from beni.core.srl.config import MasterConfig
from beni.core.srl.grpo.callbacks import (
    PreUpdateHookManager,
    distillation_hook,
    format_gradient_mask_hook,
    log_trackio,
)
from beni.data.datasets import SebeniDataLoader as DL

try:
    import trackio
except ImportError:
    trackio = None

# ChatML fallback when the hub tokenizer has no chat_template (e.g. base SmolLM2).
CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def _ensure_chat_template(tokenizer):
    """Set ChatML if ``tokenizer.chat_template`` is missing so TRL can format prompts."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer
    tokenizer.chat_template = CHATML_CHAT_TEMPLATE
    return tokenizer


def _bitsandbytes_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from transformers.utils.import_utils import is_bitsandbytes_available

        return bool(is_bitsandbytes_available())
    except Exception:
        try:
            import bitsandbytes  # noqa: F401

            return True
        except ImportError:
            return False


class AlignmentPlugin:
    """Policy-update plugin base. Subclasses wrap a TRL trainer.

    Parameters
    ----------
    config : MasterConfig
        Aggregated YAML/CLI config. Working dir should already be active.
    """

    name: str = "base"

    def __init__(self, config: Optional[MasterConfig] = None, **kwargs: Any):
        if config is not None:
            self.config = config
        else:
            self.config = MasterConfig()
            model_name = kwargs.get("model_name")
            if model_name:
                self.config.model.model_name = model_name
            for key in ("data_config", "model_config", "trainer_config", "distillation_config", "reward_config"):
                value = kwargs.get(key)
                if value is None:
                    continue
                attr = {
                    "data_config": "data",
                    "model_config": "model",
                    "trainer_config": "trainer",
                    "distillation_config": "distillation",
                    "reward_config": "reward",
                }[key]
                setattr(self.config, attr, value)

        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        spec = self.config.safety.to_spec(
            tau=self.config.distillation.tau,
            kl_beta=self.config.trainer.beta,
        )
        self.governor = SafetyGovernor(spec)
        self.reward_manager = RewardManager(
            reward_config=self.config.reward,
            working_dir=self.config.working_dir,
        )
        self.hook_manager = PreUpdateHookManager(model=None)
        self.data_loader = DL(config=self.config.data)
        self._last_logps = {"model_logps": None, "ref_logps": None, "kl": None}
        self._last_uncertainty = {}

    def register_reward(self, name: str, reward_fn: Callable, weight: float = 1.0) -> None:
        """Register a user-defined custom reward function into the framework."""
        self.reward_manager.register_custom_reward(name, reward_fn, weight)

    def register_hook(self, hook_fn: Callable) -> None:
        """Register a user-defined pre-update hook into the framework."""
        self.hook_manager.register(hook_fn)

    def apply_lora(self, model):
        """Apply LoRA PEFT adapters to the causal LM based on ModelConfig."""
        lora_cfg = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            target_modules=self.config.model.lora_target_modules,
            lora_dropout=self.config.model.lora_dropout,
            bias=self.config.model.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_cfg)

    def load_models(self, model_name: Optional[str] = None, device_map: str = "auto", remote_code: bool = True):
        """Load policy causal LM, optional reference model, and tokenizer."""
        model_name = model_name or self.config.model.model_name
        if self.config.model.load_in_4bit and self.config.model.load_in_8bit:
            raise ValueError("Choose only one quantization mode: load_in_4bit or load_in_8bit.")
        use_cpu = bool(
            getattr(self.config.trainer, "use_cpu", False)
            or getattr(self.config.dpo, "use_cpu", False)
            or getattr(self.config.apo, "use_cpu", False)
        )
        if use_cpu or not torch.cuda.is_available():
            device_map = "cpu"
            if getattr(self.config, "trainer", None):
                self.config.trainer.use_cpu = True
            if getattr(self.config, "dpo", None):
                self.config.dpo.use_cpu = True
            if getattr(self.config, "apo", None):
                self.config.apo.use_cpu = True

        bnb_config = None
        use_4bit = self.config.model.load_in_4bit and device_map != "cpu"
        use_8bit = self.config.model.load_in_8bit and device_map != "cpu"
        if (use_4bit or use_8bit) and not _bitsandbytes_usable():
            quantization_mode = "4-bit" if use_4bit else "8-bit"
            print(
                f"{quantization_mode} quantization requested but bitsandbytes/CUDA is unavailable; "
                "loading full precision. Install bitsandbytes, or set "
                "model.load_in_4bit and model.load_in_8bit to false, or trainer.use_cpu: true."
            )
            use_4bit = False
            use_8bit = False
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            )
        elif use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=remote_code,
        )

        tok_kwargs = {"trust_remote_code": remote_code}
        if "MobileLLM" in model_name:
            tok_kwargs["use_fast"] = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        _ensure_chat_template(self.tokenizer)

        if self.config.model.use_peft:
            self.model = self.apply_lora(self.model)

        if self.config.model.ref_model_name:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.ref_model_name,
                device_map=device_map,
                trust_remote_code=remote_code,
                torch_dtype=torch.float16,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.hook_manager.model = self.model
        self.hook_manager.ref_model = self.ref_model
        return self

    def format_dataset(self, sentences: Optional[List] = None):
        """Format raw records into a Hugging Face Dataset for the active scheme."""
        return self.data_loader.format_to_dataset(sentences)

    def prepare_dataset(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        source_type: str = "auto",
        **kwargs,
    ):
        records = self.data_loader.load(source, source_type=source_type, **kwargs)
        if not records:
            raise ValueError("DataLoader yields 0 records. Check source and source_type arguments provided.")
        return self.data_loader.ds()

    def _wire_pre_update_hooks(self, trainer) -> None:
        """Attach SafetyGovernor + logps capture around optimizer.step / compute_loss."""
        self.hook_manager.model = getattr(trainer, "model", self.model)
        if getattr(trainer, "ref_model", None) is None and self.ref_model is not None:
            try:
                trainer.ref_model = self.ref_model
            except Exception:
                pass
        self.hook_manager.ref_model = getattr(trainer, "ref_model", self.ref_model)

        if format_gradient_mask_hook not in self.hook_manager.hooks:
            self.hook_manager.register(format_gradient_mask_hook)
        if distillation_hook not in self.hook_manager.hooks:
            self.hook_manager.register(distillation_hook)

        if hasattr(trainer, "compute_loss"):
            original_loss = trainer.compute_loss
            callbacks = getattr(
                getattr(trainer, "callback_handler", None), "callbacks", []
            )

            def capturing_loss(model, inputs, *args, **kwargs):
                if isinstance(inputs, dict):
                    for callback in callbacks:
                        process = getattr(callback, "process_inputs", None)
                        if callable(process):
                            process(
                                inputs,
                                step=getattr(
                                    getattr(trainer, "state", None),
                                    "global_step",
                                    None,
                                ),
                            )
                result = original_loss(model, inputs, *args, **kwargs)
                if isinstance(inputs, dict):
                    for key in (
                        "old_per_token_logps",
                        "sampling_per_token_logps",
                        "per_token_logps",
                    ):
                        value = inputs.get(key)
                        if value is not None:
                            self._last_logps["model_logps"] = value
                            break
                    ref_value = inputs.get("ref_per_token_logps")
                    if ref_value is not None:
                        self._last_logps["ref_logps"] = ref_value
                return result

            trainer.compute_loss = capturing_loss

        def install_step_hook(optimizer):
            if optimizer is None or getattr(optimizer, "_sebeni_hooked", False):
                return
            original_step = optimizer.step

            def hooked_step(*args, **kwargs):
                logs = {}
                if (
                    getattr(trainer, "state", None) is not None
                    and trainer.state.log_history
                ):
                    logs = trainer.state.log_history[-1]
                kl = logs.get("kl") or logs.get("objective/kl") or logs.get("train/kl")
                batch_meta = {
                    "format_scores": list(self.reward_manager.format_scores),
                    "model_logps": self._last_logps.get("model_logps"),
                    "ref_logps": self._last_logps.get("ref_logps"),
                    "predicted_stages": list(self.reward_manager.predicted_stages),
                    "kl": kl,
                    "governor": self.governor,
                }
                allowed = self.governor.apply_pre_update(
                    self.hook_manager.model, batch_meta
                )
                self._last_uncertainty = {
                    key: batch_meta[key]
                    for key in ("u_indicator", "u_kl", "uncertainty")
                    if key in batch_meta
                }
                log_trackio(
                    {
                        key: batch_meta[key]
                        for key in ("u_indicator", "u_kl", "uncertainty")
                        if key in batch_meta
                    }
                )
                batch_meta["_governor_applied"] = True
                batch_meta["_governor_allowed"] = allowed
                self.hook_manager(batch_meta)
                self.reward_manager.clear()
                return original_step(*args, **kwargs)

            optimizer.step = hooked_step
            optimizer._sebeni_hooked = True

        install_step_hook(getattr(trainer, "optimizer", None))
        if getattr(trainer, "optimizer", None) is None and hasattr(
            trainer, "create_optimizer"
        ):
            original_create_optimizer = trainer.create_optimizer

            def create_optimizer_with_hooks(*args, **kwargs):
                result = original_create_optimizer(*args, **kwargs)
                install_step_hook(getattr(trainer, "optimizer", None))
                return result

            trainer.create_optimizer = create_optimizer_with_hooks

    def _report_to(self) -> str:
        if self.config.algorithm == "dpo":
            value = getattr(self.config.dpo, "report_to", None)
        elif self.config.algorithm == "apo":
            value = getattr(self.config.apo, "report_to", None)
        else:
            value = getattr(self.config.trainer, "report_to", None)
        if value is None:
            value = self.config.trainer.report_to
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value or "").strip().lower()

    def _trainer_owns_trackio(self) -> bool:
        """Hugging Face TrackioCallback inits/finishes when report_to is trackio."""
        return getattr(self, "trainer", None) is not None and self._report_to() == "trackio"

    def _init_trackio(self, project_name: str) -> None:
        if self._report_to() != "trackio" or trackio is None:
            return
        if self._trainer_owns_trackio():
            return
        try:
            trackio.init(
                project=project_name,
                name=f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": self.config.model.model_name,
                    "algorithm": self.config.algorithm,
                    "distillation_backend": self.config.distillation.selected_backend,
                    "distillation_model": self.config.distillation.model,
                    "working_dir": self.config.working_dir,
                },
            )
        except Exception as exc:
            print(f"Trackio init skipped: {exc}")

    def _finish_trackio(self) -> None:
        if trackio is None or self._trainer_owns_trackio():
            return
        try:
            trackio.finish()
        except Exception:
            pass

    def record_safety_snapshot(self, output_dir: Union[str, Path], **extra: Any):
        """Write SafetySnapshot next to the model card."""
        langs = extra.pop("languages", None) or self.config.languages()
        nested = dict(extra.pop("extra", None) or {})
        nested["languages"] = langs
        extra.setdefault("language", ",".join(langs) if isinstance(langs, (list, tuple)) else langs)
        extra.setdefault("group_code", extra["language"])
        for key, value in self._last_uncertainty.items():
            extra.setdefault(key, value)
        extra["extra"] = nested
        snapshot = self.governor.record_snapshot(
            tau=self.config.distillation.tau,
            algorithm=self.config.algorithm,
            **extra,
        )
        self.governor.write_snapshot(output_dir)
        return snapshot

    def write_card_and_snapshot(self, output_dir: Union[str, Path], **extra: Any) -> Path:
        snapshot = self.record_safety_snapshot(output_dir, **extra)
        return write_model_card(output_dir, snapshot=snapshot, config=self.config)

    def save_model(self, output_dir: Union[str, Path] = None):
        """Save the trained model, tokenizer, model card, and safety snapshot."""
        output_dir = Path(output_dir) if output_dir else Path(self.config.trainer.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        self.write_card_and_snapshot(output_dir)
        return output_dir

    def push_to_hub(self, repo_id: Optional[str] = None, token=None, private: bool = True, **kwargs):
        """Push the trained model to the Hub. Refuses without card + safety snapshot."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before pushing to hub.")
        if not repo_id:
            raise ValueError("repo_id is required to push to hub.")
        output_dir = Path(self.config.trainer.output_dir)
        card = output_dir / "README.md"
        snap = output_dir / SafetyGovernor.SNAPSHOT_NAME
        if not card.exists() or not snap.exists():
            self.write_card_and_snapshot(output_dir)
        if not self.governor.allow_hub_push(output_dir):
            raise PermissionError(
                "Hub push refused: model card and safety snapshot are required "
                f"(gates={self.config.safety.to_spec().fired or self.governor.spec.fired})."
            )
        self.model.push_to_hub(repo_id=repo_id, token=token, private=private, **kwargs)
        self.tokenizer.push_to_hub(repo_id=repo_id, token=token, private=private, **kwargs)

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate text from the policy model given a prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def token_probabilities(self, text: str):
        """Per-token π_θ and π_ref as probabilities. Forward pass may use Torch."""
        import numpy as np

        if self.model is None or self.tokenizer is None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        encoded = self.tokenizer(text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = None
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[-1]) < 2:
            ones = np.asarray([1.0], dtype=float)
            return ones, ones.copy()

        def _token_probs(model) -> np.ndarray:
            with torch.no_grad():
                logits = model(**encoded).logits
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            labels = input_ids[:, 1:]
            gathered = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            return gathered.exp().detach().cpu().float().numpy().reshape(-1)

        self.model.eval()
        pi_theta = _token_probs(self.model)
        pi_ref = None
        if self.ref_model is not None:
            self.ref_model.eval()
            pi_ref = _token_probs(self.ref_model)
        else:
            disable = getattr(self.model, "disable_adapter", None)
            if callable(disable):
                try:
                    with disable():
                        pi_ref = _token_probs(self.model)
                except Exception:
                    pi_ref = None
        if pi_ref is None:
            pi_ref = np.array(pi_theta, copy=True)
        return pi_theta, pi_ref

    geneate = generate

    def train(self, data, project_name: Optional[str] = None, **kwargs):
        raise NotImplementedError
