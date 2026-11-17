"""Shared alignment plugin surface (load / save / push / generate).

GRPO, DPO, and APO implement ``train``; they share this dataclass-style wrapper
around Transformers + PEFT + Hub cards.
"""

from __future__ import annotations

import datetime
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("sebeni.srl.plugin")

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


_TRAINER_BLOCKS = ("trainer", "sft", "dpo", "apo")

# Written next to saved weights: the language list the training prompt named.
POLICY_META = "sebeni_policy.json"


def write_policy_meta(output_dir: Union[str, Path], languages: Sequence[str]) -> Path:
    import json

    path = Path(output_dir) / POLICY_META
    path.write_text(json.dumps({"prompt_languages": list(languages)}, indent=2), encoding="utf-8")
    return path


def _read_policy_languages(weights_dir: Path) -> Optional[List[str]]:
    import json

    path = Path(weights_dir) / POLICY_META
    if not path.is_file():
        return None
    try:
        langs = json.loads(path.read_text(encoding="utf-8")).get("prompt_languages")
    except (OSError, ValueError, AttributeError):
        return None
    return [str(x) for x in langs] if isinstance(langs, list) and langs else None


def _active_block(config):
    """Trainer block of the active arm (``trainer`` for grpo; ``sft`` / ``dpo`` / ``apo`` otherwise)."""
    algorithm = str(getattr(config, "algorithm", "grpo") or "grpo").lower()
    block = getattr(config, algorithm, None) if algorithm in ("sft", "dpo", "apo") else None
    return block if block is not None else getattr(config, "trainer", None)


def _resolve_device(config) -> str:
    """``"cpu"`` when the active arm asks for it or CUDA is missing, else ``"auto"``."""
    if bool(getattr(_active_block(config), "use_cpu", False)):
        return "cpu"
    return "auto" if torch.cuda.is_available() else "cpu"


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
    # What the governor's per-item signals describe: the policy's own completions
    # ("samples", GRPO) or the training targets the gradient pulls towards
    # ("targets": SFT labels, DPO/APO chosen answers).
    signal_source: str = "samples"

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
        self._last_logps = {"model_logps": None, "ref_logps": None, "kl": None, "log_ratio": None}
        self._step_logps: List[Dict[str, Any]] = []
        self.target_gate_report: Optional[Dict[str, int]] = None
        self._last_uncertainty = {}
        self.loaded_weights: Optional[str] = None
        # Local directory of the loaded weights (a Hub id resolves to its snapshot).
        self.weights_dir: Optional[str] = None
        # Optimizer steps the SafetyGovernor refused (θ unchanged for those steps).
        self.blocked_steps = 0
        self._consecutive_blocked = 0
        # Languages named in the training system prompt (read from POLICY_META next to loaded weights).
        self.prompt_languages: Optional[List[str]] = None

    def register_reward(self, name: str, reward_fn: Callable, weight: float = 1.0) -> None:
        """Register a user-defined custom reward function into the framework."""
        self.reward_manager.register_custom_reward(name, reward_fn, weight)

    def register_hook(self, hook_fn: Callable) -> None:
        """Register a user-defined pre-update hook into the framework."""
        self.hook_manager.register(hook_fn)

    def apply_lora(self, model):
        """Apply LoRA PEFT adapters to the causal LM based on ModelConfig."""
        # Discover any custom module classes (e.g. LiquidAI LFM)
        # Exclude standard torch modules and bitsandbytes modules which PEFT supports natively
        custom_classes = set()
        for _, module in model.named_modules():
            cls = module.__class__
            module_module = getattr(cls, "__module__", "")
            if "bitsandbytes" in module_module:
                continue
            if cls in (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d):
                continue
            if hasattr(module, "weight") and hasattr(module, "in_features") and hasattr(module, "out_features"):
                custom_classes.add(cls)

        # Collect supported linear/embedding modules vs unsupported wrapper modules
        supported_exact_modules = []
        unsupported_matching_modules = []
        base_targets = self.config.model.lora_target_modules
        if isinstance(base_targets, (list, tuple, set)):
            target_set = set(base_targets)
            for name, module in model.named_modules():
                cls = module.__class__
                mod_name = getattr(cls, "__module__", "")
                is_supported = (
                    isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d))
                    or "bitsandbytes" in mod_name
                    or cls in custom_classes
                )
                for t in target_set:
                    if name == t or name.endswith(f".{t}"):
                        if is_supported:
                            supported_exact_modules.append(name)
                        else:
                            unsupported_matching_modules.append((name, cls.__name__))

        # If unsupported modules match targets (e.g. Gemma4ClippableLinear in Gemma 4 vision/audio encoders),
        # scope target_modules strictly to supported modules so PEFT doesn't crash on non-linear wrappers.
        effective_target_modules = self.config.model.lora_target_modules
        if unsupported_matching_modules and supported_exact_modules:
            logger.info(
                "Detected %d unsupported module(s) (%s) matching LoRA targets. "
                "Scoping target_modules to %d supported linear layers.",
                len(unsupported_matching_modules),
                unsupported_matching_modules[0][1],
                len(supported_exact_modules),
            )
            effective_target_modules = supported_exact_modules

        lora_kwargs = {
            "r": self.config.model.lora_r,
            "lora_alpha": self.config.model.lora_alpha,
            "target_modules": effective_target_modules,
            "lora_dropout": self.config.model.lora_dropout,
            "bias": self.config.model.lora_bias,
            "task_type": TaskType.CAUSAL_LM,
        }
        # Only pass custom_module_classes if supported by the installed PEFT version
        peft_supports_custom = "custom_module_classes" in inspect.signature(LoraConfig.__init__).parameters
        if custom_classes and peft_supports_custom:
            lora_kwargs["custom_module_classes"] = list(custom_classes)

        try:
            lora_cfg = LoraConfig(**lora_kwargs)
            return get_peft_model(model, lora_cfg)
        except Exception as exc:
            logger.warning("LoRA injection failed with error: %s. Attempting fallback...", exc)

            # Fallback 1: Filter target_modules to only standard supported linear/quantized modules
            if supported_exact_modules:
                try:
                    fallback_kwargs = dict(lora_kwargs)
                    fallback_kwargs["target_modules"] = supported_exact_modules
                    fallback_kwargs.pop("custom_module_classes", None)
                    return get_peft_model(model, LoraConfig(**fallback_kwargs))
                except Exception as exc2:
                    logger.warning("Fallback targeting standard linear modules failed: %s", exc2)

            # Fallback 2: Full parameter fine-tuning fallback (only floating-point & complex tensors can require gradients)
            logger.warning("Model does not support PEFT LoRA adapter wrapping. Falling back to fine-tuning mode.")
            for param in model.parameters():
                if param.is_floating_point() or param.is_complex():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            return model

    def load_models(
        self,
        model_name: Optional[str] = None,
        device_map: str = "auto",
        remote_code: bool = True,
        apply_peft: Optional[bool] = None,
        tokenizer_name: Optional[str] = None,
    ):
        """Load policy causal LM, optional reference model, and tokenizer.

        ``apply_peft=False`` skips the fresh LoRA wrap (used before loading a
        trained adapter). ``tokenizer_name`` loads the tokenizer from another
        path, e.g. the training output directory.
        """
        model_name = model_name or self.config.model.model_name
        if self.config.model.load_in_4bit and self.config.model.load_in_8bit:
            raise ValueError("Choose only one quantization mode: load_in_4bit or load_in_8bit.")
        if _resolve_device(self.config) == "cpu":
            device_map = "cpu"
            for block in _TRAINER_BLOCKS:
                if getattr(self.config, block, None) is not None:
                    getattr(self.config, block).use_cpu = True

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
        use_vllm_active = bool(getattr(getattr(self.config, "trainer", None), "use_vllm", False))
        if use_vllm_active and use_4bit:
            print(
                "[GRPO Notice] Disabling 4-bit quantization because vLLM rollout acceleration requires "
                "native unquantized weights (bf16/fp16). Loading in native precision."
            )
            use_4bit = False
            self.config.model.load_in_4bit = False

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            )
        elif use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        torch_dtype = None
        if not bnb_config and device_map != "cpu":
            trainer_cfg = _active_block(self.config)
            if getattr(trainer_cfg, "bf16", False):
                torch_dtype = torch.bfloat16
            elif getattr(trainer_cfg, "fp16", False):
                torch_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=remote_code,
        )

        tok_kwargs = {"trust_remote_code": remote_code}
        if "MobileLLM" in model_name:
            tok_kwargs["use_fast"] = False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, **tok_kwargs)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        _ensure_chat_template(self.tokenizer)

        if self.config.model.use_peft if apply_peft is None else apply_peft:
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

    def split_eval(self, dataset, seed: Optional[int] = None):
        """Carve the in-training hold-out from ``dataset``.

        Uses ``data.eval_rows`` (a row count; 0 disables the split) when set,
        otherwise ``data.eval_ratio``. ``experiment.max_eval_rows`` is not read here: it
        only caps the held-out ``test.json`` scored by ``run_eval``.
        Returns ``(train_dataset, eval_dataset_or_None)``.
        """
        n_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        if not hasattr(dataset, "train_test_split") or n_samples < 2:
            return dataset, None
        eval_rows = getattr(self.config.data, "eval_rows", None)
        eval_ratio = float(getattr(self.config.data, "eval_ratio", 0.0) or 0.0)
        if eval_rows is not None:
            if int(eval_rows) <= 0:
                return dataset, None  # eval_rows: 0 turns the in-training eval off
            test_size = min(int(eval_rows), n_samples - 1)
        elif eval_ratio > 0.0:
            if eval_ratio >= 1.0:
                raise ValueError(f"data.eval_ratio must be below 1.0, got {eval_ratio}")
            test_size = eval_ratio
        else:
            return dataset, None
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split["train"], split["test"]

    def _governor_wants_logps(self) -> bool:
        spec = self.governor.spec
        return bool(spec.enabled and (spec.uncertainty_downweight or spec.max_kl is not None))

    def gate_targets(self, rows: List[Dict[str, Any]], completion_key: str, lang_key: str = "lang"):
        """Drop training rows whose target the SafetyGovernor would refuse.

        Applies the policy-update gates to each row's target once, before
        training: format validity (``format_invalid_blocks_update``), declared
        language vs the row's language (``require_language``) and mixing
        (``refuse_mixed_language``). Fires ``format_invalid_target`` /
        ``language_mismatch_target`` / ``mixed_language_target``. Returns the
        kept rows; counts go to ``self.target_gate_report``.
        """
        spec = self.governor.spec
        kept: List[Dict[str, Any]] = []
        dropped = {"format_invalid_target": 0, "language_mismatch_target": 0, "mixed_language_target": 0}
        for row in rows:
            target = row.get(completion_key)
            reason = None
            if spec.enabled:
                ok, mixed = RewardManager.language_identity(target, row.get(lang_key) or "")
                if spec.format_invalid_blocks_update and not RewardManager.target_valid(target):
                    reason = "format_invalid_target"
                elif spec.refuse_mixed_language and mixed:
                    reason = "mixed_language_target"
                elif spec.require_language and ok is False:
                    reason = "language_mismatch_target"
            if reason is None:
                kept.append(row)
            else:
                dropped[reason] += 1
                self.governor._fire(reason)
        self.target_gate_report = {"n_kept": len(kept), "n_dropped": sum(dropped.values()), **dropped}
        if self.target_gate_report["n_dropped"]:
            logger.warning("SafetyGovernor dropped training targets: %s", self.target_gate_report)
        return kept

    def drop_overlength(
        self,
        rows: List[Dict[str, Any]],
        prompt_of: Callable[[Dict[str, Any]], str],
        answers_of: Callable[[Dict[str, Any]], List[str]],
        max_length: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Drop rows whose prompt + longest answer exceeds ``max_length`` tokens.

        TRL truncates such sequences, which cuts the JSON target mid-object and
        trains the policy on invalid output. Counts go to
        ``target_gate_report["overlength_target"]``.
        """
        report = self.target_gate_report if self.target_gate_report is not None else {"n_kept": len(rows), "n_dropped": 0}
        if not max_length or self.tokenizer is None:
            report.setdefault("overlength_target", 0)
            self.target_gate_report = report
            return rows
        kept: List[Dict[str, Any]] = []
        longest = 0
        for row in rows:
            n_prompt = len(self.tokenizer(prompt_of(row))["input_ids"])
            n_answer = max(
                (len(self.tokenizer(a, add_special_tokens=False)["input_ids"]) for a in answers_of(row)),
                default=0,
            )
            longest = max(longest, n_prompt + n_answer)
            if n_prompt + n_answer + 1 <= int(max_length):  # +1 for an appended eos
                kept.append(row)
        dropped = len(rows) - len(kept)
        report["overlength_target"] = dropped
        report["max_length"] = int(max_length)
        report["n_kept"] = len(kept)
        report["n_dropped"] = int(report.get("n_dropped", 0)) + dropped
        self.target_gate_report = report
        if dropped:
            self.governor._fire("overlength_target")
            logger.warning(
                "Dropped %d/%d training rows longer than max_length=%d tokens (longest %d); "
                "raise max_length or use shorter (sentence-level) rows.",
                dropped, len(rows), int(max_length), longest,
            )
        if rows and not kept:
            raise RuntimeError(
                f"Every training row is longer than max_length={int(max_length)} tokens (longest {longest}); "
                "TRL would truncate the JSON targets. Raise max_length or use shorter rows."
            )
        return kept

    def _decode_targets(self, inputs: Dict[str, Any]) -> List[str]:
        """Decode the target tokens of a batch: DPO ``completion_mask`` (chosen half) or SFT ``labels``."""
        input_ids = inputs.get("input_ids")
        if input_ids is None or self.tokenizer is None:
            return []
        if inputs.get("completion_mask") is not None:
            mask = inputs["completion_mask"].bool()
            half = input_ids.shape[0] // 2  # DPO batches are [chosen..., rejected...]
            input_ids, mask = input_ids[:half], mask[:half]
        elif inputs.get("labels") is not None:
            mask = inputs["labels"] != -100
        else:
            return []
        return [
            self.tokenizer.decode(ids[row_mask].tolist(), skip_special_tokens=True)
            for ids, row_mask in zip(input_ids, mask)
        ]

    def _targets_loss(self, trainer, original_loss, model, inputs, *args, **kwargs):
        """compute_loss for target-driven arms: feed the governor, then return the loss.

        Records format / language signals for the batch's targets and the
        policy-vs-reference log-ratio (SFT: per target token, with the adapter
        disabled or the ref model; DPO/APO: from TRL's ``rewards/chosen`` and
        ``rewards/rejected``).
        """
        training = bool(getattr(model, "training", False))
        if training:
            try:
                self.reward_manager.observe_targets(self._decode_targets(inputs), allowed=self.config.languages())
            except Exception as exc:  # signals must never break training
                logger.debug("Target observation failed: %s", exc)
        want_logps = training and self._governor_wants_logps()
        sft_like = inputs.get("labels") is not None and inputs.get("completion_mask") is None
        if not (want_logps and sft_like):
            lengths = (1.0, 1.0)
            mask = inputs.get("completion_mask")
            if want_logps and mask is not None:
                counts = mask[:, 1:].sum(dim=1).float()
                half = counts.shape[0] // 2
                lengths = (float(counts[:half].mean().item()), float(counts[half:].mean().item()))
            result = original_loss(model, inputs, *args, **kwargs)
            if want_logps:
                self._capture_dpo_log_ratio(trainer, *lengths)
            return result

        caller_wants_outputs = bool(kwargs.pop("return_outputs", False))
        # TRL's SFT compute_loss may pop "labels" (custom compute_loss_func); keep our own references.
        saved = {k: inputs[k] for k in ("input_ids", "attention_mask", "labels") if inputs.get(k) is not None}
        loss, outputs = original_loss(model, inputs, *args, return_outputs=True, **kwargs)
        try:
            self._capture_sft_logps(model, saved, outputs)
        except Exception as exc:
            if not getattr(self, "_warned_logps", False):
                self._warned_logps = True
                logger.warning("SFT log-prob capture failed; U and the KL bound get no log-ratio: %r", exc)
        return (loss, outputs) if caller_wants_outputs else loss

    @staticmethod
    def _selected_logps(logits, labels, mask, chunk: int = 256):
        """log p(label) at ``mask`` positions only, in row chunks (no full-vocab fp32 copy)."""
        rows = logits[mask]
        targets = labels[mask]
        out = []
        for i in range(0, rows.shape[0], chunk):
            block = rows[i : i + chunk].float()
            out.append(block.gather(-1, targets[i : i + chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(block, dim=-1))
        return torch.cat(out) if out else rows.new_zeros(0, dtype=torch.float32)

    def _target_logps(self, module, forward, labels):
        """Per-target-token log-probs from ``module``; only the target suffix's logits are computed when supported."""
        length = labels.shape[1]
        positions = (labels != -100).float().argmax(dim=1)
        keep = int(length - int(positions.min().item()) + 1)
        try:
            logits = module(**forward, logits_to_keep=keep).logits
        except TypeError:
            logits = module(**forward).logits
        keep = logits.shape[1]
        shifted = labels[:, length - keep + 1 :]
        mask = shifted != -100
        return self._selected_logps(logits[:, :-1, :], shifted.clamp(min=0), mask)

    def _capture_sft_logps(self, model, inputs, outputs) -> None:
        labels = inputs["labels"]
        if not bool((labels[:, 1:] != -100).any()):
            return
        with torch.no_grad():
            forward = {k: inputs[k] for k in ("input_ids", "attention_mask") if inputs.get(k) is not None}
            base = getattr(model, "module", model)
            logits = getattr(outputs, "logits", None)
            if logits is not None and logits.shape[1] == labels.shape[1]:
                shifted = labels[:, 1:]
                policy = self._selected_logps(logits[:, :-1, :].detach(), shifted.clamp(min=0), shifted != -100)
            else:
                # TRL's SFT loss (chunked_nll) skips materialising logits; one no-grad policy forward.
                policy = self._target_logps(base, forward, labels)
            ref = None
            if self.ref_model is not None:
                ref = self._target_logps(self.ref_model, forward, labels)
            elif hasattr(base, "disable_adapter"):
                with base.disable_adapter():
                    ref = self._target_logps(base, forward, labels)
        if ref is None or ref.shape != policy.shape:
            return
        entry = {
            "model_logps": policy,
            "ref_logps": ref,
            "kl": float((policy - ref).mean().item()),
            "log_ratio": None,
        }
        self._record_microbatch_logps(entry)

    def _record_microbatch_logps(self, entry: Dict[str, Any]) -> None:
        """Record one microbatch's log-ratio entry so gradient accumulation aggregates all microbatches."""
        self._last_logps = dict(entry)
        self._step_logps.append(dict(entry))

    def _aggregated_step_logps(self) -> Dict[str, Any]:
        """Aggregate log-ratios collected across microbatches since the last optimizer step."""
        if not self._step_logps:
            return dict(self._last_logps)
        if len(self._step_logps) == 1:
            return dict(self._step_logps[0])

        def _combine_tensors(items):
            if not items:
                return None
            if all(torch.is_tensor(x) for x in items):
                return torch.cat([x.detach().reshape(-1) for x in items])
            return items[-1]

        kls = [float(e["kl"]) for e in self._step_logps if e.get("kl") is not None]
        lrs = [float(e["log_ratio"]) for e in self._step_logps if e.get("log_ratio") is not None]
        return {
            "model_logps": _combine_tensors([e.get("model_logps") for e in self._step_logps if e.get("model_logps") is not None]),
            "ref_logps": _combine_tensors([e.get("ref_logps") for e in self._step_logps if e.get("ref_logps") is not None]),
            "kl": (sum(kls) / len(kls)) if kls else None,
            "log_ratio": (sum(lrs) / len(lrs)) if lrs else None,
        }

    def _capture_dpo_log_ratio(self, trainer, n_chosen: float = 1.0, n_rejected: float = 1.0) -> None:
        """Per-token |log π_θ/π_ref| for the chosen and rejected answers, from TRL's rewards.

        TRL's ``rewards/*`` are β · log-ratio summed over each answer, averaged over
        the batch; dividing by the mean answer length puts them on the same
        per-token scale as SFT. Absolute values keep chosen-up / rejected-down
        drift from cancelling.
        """
        metrics = (getattr(trainer, "_metrics", None) or {}).get("train") or {}
        chosen, rejected = metrics.get("rewards/chosen"), metrics.get("rewards/rejected")
        beta = float(getattr(trainer, "beta", 0.0) or 0.0)
        if not chosen or not rejected or beta <= 0:
            return
        per_token_c = abs(float(chosen[-1])) / (beta * max(float(n_chosen), 1.0))
        per_token_r = abs(float(rejected[-1])) / (beta * max(float(n_rejected), 1.0))
        drift = (per_token_c + per_token_r) / 2.0
        self._record_microbatch_logps({"model_logps": None, "ref_logps": None, "log_ratio": drift, "kl": drift})

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
                if self.signal_source == "targets" and isinstance(inputs, dict):
                    return self._targets_loss(trainer, original_loss, model, inputs, *args, **kwargs)
                result = original_loss(model, inputs, *args, **kwargs)
                if isinstance(inputs, dict):
                    model_val = None
                    for key in (
                        "old_per_token_logps",
                        "sampling_per_token_logps",
                        "per_token_logps",
                    ):
                        value = inputs.get(key)
                        if value is not None:
                            model_val = value
                            break
                    ref_value = inputs.get("ref_per_token_logps")
                    if model_val is not None or ref_value is not None:
                        self._record_microbatch_logps(
                            {
                                "model_logps": model_val,
                                "ref_logps": ref_value,
                                "kl": None,
                                "log_ratio": None,
                            }
                        )
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
                step_logps = self._aggregated_step_logps()
                kl = logs.get("kl") or logs.get("objective/kl") or logs.get("train/kl")
                if kl is None:
                    kl = step_logps.get("kl")
                batch_meta = {
                    "signal_source": self.signal_source,
                    "lang_ok": list(self.reward_manager.lang_ok),
                    "mixed_language": list(self.reward_manager.mixed_language),
                    "format_scores": list(self.reward_manager.format_scores),
                    "model_logps": step_logps.get("model_logps"),
                    "ref_logps": step_logps.get("ref_logps"),
                    "predicted_stages": list(self.reward_manager.predicted_stages),
                    "kl": kl,
                    "log_ratio": step_logps.get("log_ratio"),
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
                # Signals are per step: never let a stale log-ratio reach the next step's gates.
                self._step_logps.clear()
                self._last_logps = {"model_logps": None, "ref_logps": None, "kl": None, "log_ratio": None}
                if not allowed:
                    # Zeroed gradients are not enough: AdamW momentum and decoupled
                    # weight decay would still move θ. Skip the step entirely.
                    self.blocked_steps += 1
                    self._consecutive_blocked += 1
                    if self._consecutive_blocked in (10, 50) or self._consecutive_blocked % 100 == 0:
                        print(
                            f"[SafetyGovernor] Warning: {self._consecutive_blocked} consecutive optimizer steps "
                            f"refused (gates fired: {list(self.governor.spec.fired)}); θ is not changing. "
                            "Check the data, max_length, safety.max_kl, or the arm's warm start."
                        )
                    return None
                self._consecutive_blocked = 0
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
        nested.setdefault("blocked_steps", self.blocked_steps)
        if self.target_gate_report is not None:
            nested.setdefault("target_gate", dict(self.target_gate_report))
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
        write_policy_meta(output_dir, self.config.languages())
        self.write_card_and_snapshot(output_dir)
        # The policy in memory now matches these files; push and eval refer to them.
        self.weights_dir = str(output_dir)
        print(
            f"[SafetyGovernor] {self.name}: signals={self.signal_source} blocked_steps={self.blocked_steps} "
            f"gates_fired={list(self.governor.spec.fired)} last_uncertainty={self._last_uncertainty or None}"
        )
        return output_dir

    def push_to_hub(
        self,
        repo_id: Optional[str] = None,
        token=None,
        private: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        write_card: bool = False,
        **kwargs,
    ):
        """Upload the checked weights directory to the Hub. Refuses without a card and a matching eval snapshot.

        ``output_dir`` is the directory of the weights being pushed (default:
        the loaded weights, else the trainer ``output_dir``). Nothing is
        written implicitly: run ``sebeni eval`` first (it writes
        ``safety_snapshot.json`` there). ``write_card=True`` re-renders
        ``README.md`` from that snapshot.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before pushing to hub.")
        if not repo_id:
            raise ValueError("repo_id is required to push to hub.")
        directory = Path(output_dir or self.weights_dir or self.config.trainer.output_dir)
        snap = directory / SafetyGovernor.SNAPSHOT_NAME
        if write_card:
            if not snap.is_file():
                raise PermissionError(
                    f"Hub push refused: no {SafetyGovernor.SNAPSHOT_NAME} in {directory}. "
                    "Run `sebeni eval` first; --write-card only renders the card from it."
                )
            from beni.core.safety.governor import SafetySnapshot

            write_model_card(directory, snapshot=SafetySnapshot.load(snap), config=self.config)
        if not self.governor.allow_hub_push(directory):
            raise PermissionError(
                f"Hub push refused for {directory} (gates={self.governor.spec.fired}). "
                "Run `sebeni eval` first (writes safety_snapshot.json with eval metrics), "
                "and pass --write-card if README.md is missing."
            )
        # Upload exactly the directory the gate checked (weights, tokenizer, card,
        # snapshot), not whatever model happens to be in memory.
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(directory),
            repo_id=repo_id,
            ignore_patterns=["checkpoint-*", "checkpoint-*/**", "runs/**", "*.pt", "*.pth"],
            **kwargs,
        )

    def push_after_train(self, block) -> None:
        """Honour ``<arm>.push_to_hub: true`` at the end of training.

        Training writes a snapshot without eval metrics, so the Hub gate
        normally refuses here; that is reported, not raised, so a finished
        training run is not lost.
        """
        if not getattr(block, "push_to_hub", False):
            return
        try:
            self.push_to_hub(
                repo_id=getattr(block, "hub_model_id", None),
                token=getattr(block, "hub_token", None),
                private=getattr(block, "hub_private_repo", True),
                output_dir=getattr(block, "output_dir", None),
            )
        except PermissionError as exc:
            print(f"[SafetyGovernor] Skipped push_to_hub after training: {exc}")

    def load_aligned_weights(self, weights_path_or_repo: Union[str, Path]):
        """Load trained weights for evaluation, generation, or push.

        ``weights_path_or_repo`` is a local directory or a Hub repo id holding
        either a PEFT adapter (``adapter_config.json``) or a full checkpoint.
        An adapter is attached to a freshly loaded base model (no new LoRA).
        Raises ``RuntimeError`` when nothing loadable is found.
        """
        label = str(weights_path_or_repo)
        target = Path(label).expanduser()
        if not target.exists():
            # Hub repo id: resolve to the local snapshot (uses the cache, works with HF_HUB_OFFLINE).
            try:
                from huggingface_hub import snapshot_download

                target = Path(snapshot_download(label))
            except Exception as exc:
                raise RuntimeError(
                    f"Weights not found: {label} is neither a local directory nor a reachable Hub repo ({exc})"
                ) from exc
        if not target.is_dir():
            raise RuntimeError(f"Weights path is not a directory: {target}")
        files = {p.name for p in target.iterdir()}
        target_str = str(target)
        tokenizer_source = target_str if "tokenizer_config.json" in files else None

        if "adapter_config.json" in files:
            from peft import PeftModel

            self.load_models(apply_peft=False, tokenizer_name=tokenizer_source)
            try:
                self.model = PeftModel.from_pretrained(self.model, target_str, is_trainable=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to load the PEFT adapter from {target_str}: {exc}") from exc
        elif "config.json" in files:
            try:
                self.load_models(model_name=target_str, apply_peft=False, tokenizer_name=tokenizer_source)
            except Exception as exc:
                raise RuntimeError(f"Failed to load the model checkpoint from {target_str}: {exc}") from exc
        else:
            raise RuntimeError(
                f"No adapter_config.json or config.json in {target_str}; nothing to load."
            )
        self.hook_manager.model = self.model
        self.model.eval()
        self.loaded_weights = label if not Path(label).expanduser().exists() else target_str
        self.weights_dir = target_str
        self.prompt_languages = _read_policy_languages(target)
        return self.model

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate from a raw prompt string, returning only the new tokens.

        ``temperature <= 0`` decodes greedily. Use :meth:`analyze` to decode a
        sentence with the training prompt.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        self.model.eval()
        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = "cpu"
        add_special = kwargs.pop("add_special_tokens", True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special).to(device)
        input_len = inputs["input_ids"].shape[-1]

        if temperature and temperature > 0.0:
            gen_kwargs = {"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p}
        else:
            gen_kwargs = {"do_sample": False}
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs.update(kwargs)
        if max_length is not None:
            gen_kwargs["max_length"] = max_length
        else:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        # Strip prompt tokens so completion contains only generated content
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def analyze(
        self,
        text: str,
        languages: Optional[Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Decode the morphological JSON for one sentence, greedily, with the training prompt.

        ``max_new_tokens`` defaults to ``experiment.eval_max_new_tokens``.
        """
        from beni.core.srl.prompting import build_policy_prompt, end_of_turn_ids, strip_end_of_turn

        if self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")
        prompt = build_policy_prompt(
            text, self.tokenizer, languages or self.prompt_languages or self.config.languages()
        )
        limit = max_new_tokens or int(getattr(self.config.experiment, "eval_max_new_tokens", 512) or 512)
        extra = {}
        stop_ids = end_of_turn_ids(self.tokenizer)
        if stop_ids:
            extra["eos_token_id"] = stop_ids
        # A rendered chat template already carries any BOS; TRL GRPO/DPO tokenize it the same way.
        extra["add_special_tokens"] = not getattr(self.tokenizer, "chat_template", None)
        completion = self.generate(prompt, max_new_tokens=limit, temperature=0.0, **extra)
        return strip_end_of_turn(completion, self.tokenizer)

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
