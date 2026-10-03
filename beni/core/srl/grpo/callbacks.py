# Prehook and Training Callbacks
import torch.nn.functional as F
from typing import List, Dict, Any, Callable, Optional
from transformers import TrainerCallback

try:
    import trackio
except ImportError:
    trackio = None


def log_trackio(metrics: dict, step=None) -> None:
    """Write numeric metrics to the current Trackio run (``trackio.log``)."""
    if trackio is None:
        return
    payload = {
        key: value
        for key, value in (metrics or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not payload:
        return
    try:
        trackio.log(payload, step=step)
    except Exception:
        pass


class BatchMetadata:
    def __init__(self):
        self.format_scores = []
        self.total_reward = 0.0
        self.model_logps = None
        self.ref_logps = None
        self.predicted_stages = []
        
    def clear(self):
        self.format_scores.clear()
        self.total_reward = 0.0
        self.model_logps = None
        self.ref_logps = None
        self.predicted_stages.clear()

metadata = BatchMetadata()


class PreUpdateHookManager:
    """Manages execution of pre-optimizer-step hooks on model gradients and batch metadata."""
    def __init__(self, model=None, ref_model=None):
        self.model = model
        self.ref_model = ref_model
        self.hooks: List[Callable] = []

    def register(self, hook_fn: Callable) -> None:
        self.hooks.append(hook_fn)

    def __call__(self, batch_metadata: Dict[str, Any]) -> None:
        for hook in self.hooks:
            try:
                hook(self.model, self.ref_model, batch_metadata)
            except Exception as e:
                print(f"[PreUpdateHookManager] Exception in hook {hook.__name__}: {e}")


def format_gradient_mask_hook(model, ref_model, batch_meta: Dict[str, Any]) -> None:
    """Zero out gradients if format scores in batch are 0 (no valid JSON).

    Delegates to SafetyGovernor when present; otherwise applies the legacy mask.
    """
    gov = batch_meta.get("governor")
    if gov is not None:
        if batch_meta.get("_governor_applied"):
            return
        gov.apply_pre_update(model, batch_meta)
        batch_meta["_governor_applied"] = True
        return
    format_scores = batch_meta.get("format_scores", [])
    if not format_scores:
        return
    avg_format = sum(format_scores) / len(format_scores)
    if avg_format == 0.0:
        print("[Hook] Format Masking: Zeroing gradients (No valid JSON in batch).")
        if model is not None:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()


def distillation_hook(model, ref_model, batch_meta: Dict[str, Any]) -> None:
    """Scale gradients by inverse KL between policy and reference log-probs.

    This is reward-distrust / KL scaling, **not** SAMPG's
    per-batch Distiller. When a SafetyGovernor is attached it owns this path.
    """
    if batch_meta.get("_governor_applied"):
        return
    gov = batch_meta.get("governor")
    if gov is not None:
        gov.apply_pre_update(model, batch_meta)
        batch_meta["_governor_applied"] = True
        return
    model_logps = batch_meta.get("model_logps")
    ref_logps = batch_meta.get("ref_logps")
    if model_logps is None or ref_logps is None:
        return
    kl_div = F.kl_div(model_logps, ref_logps, log_target=True, reduction="batchmean")
    scale = 1.0 / (1.0 + kl_div.item())
    if model is not None:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)


class SelfAwareCallback(TrainerCallback):
    """Kept so older imports resolve. Resource promotion now happens upstream.

    This callback does not propose or write grammar and dictionary files.
    """

    def __init__(self, config, governor=None, distiller=None, on_promote=None):
        super().__init__()
        self.config = config
        self.governor = governor
        self.distiller = distiller
        self.on_promote = on_promote
        self._distillers = {}
        self.tau = float(getattr(config.distillation, "tau", 0.5))
        self.last_decision = None
        self._last_batch_key = None

    def _distiller_for(self, lang: Optional[str]):
        from beni.core.morphotactic.distil.distillation import Distiller
        from beni.core.language import Language

        code = lang or self.config.data.default_lang or "bam"
        group = Language.from_code(code).group_code
        cached = self._distillers.get(group)
        if cached is not None:
            return cached
        injected = self.distiller
        if injected is not None:
            injected_lang = getattr(injected, "lang", None)
            if not isinstance(injected_lang, str) or injected_lang == group:
                self._distillers[group] = injected
                return injected
        distiller = Distiller(
            lang_code=group,
            backend=self.config.distillation.selected_backend,
            model=self.config.distillation.model,
            working_dir=self.config.distillation.working_dir or self.config.working_dir,
            vertex=self.config.distillation.vertex,
            base_url=self.config.distillation.base_url,
            gguf_path=self.config.distillation.gguf_path,
            n_ctx=self.config.distillation.n_ctx,
            max_input_chars=self.config.distillation.max_input_chars,
        )
        self._distillers[group] = distiller
        return distiller

    def process_inputs(self, inputs, step=None):
        """No-op. Distillation is not part of the policy step."""
        del inputs, step
        return None

    def on_train_batch_begin(self, args, state, control, **kwargs):
        self.process_inputs(kwargs.get("inputs"), step=getattr(state, "global_step", None))
        return control


class TrackioMetricsCallback(TrainerCallback):
    """Log Sebeni extras (U, reward totals) into the Trainer Trackio run."""
    def __init__(self, reward_manager=None):
        super().__init__()
        self.reward_manager = reward_manager

    def on_log(self, args, state, control, logs=None, **kwargs):
        extra = {}
        if self.reward_manager is not None:
            total = getattr(self.reward_manager, "total_reward", 0.0) or 0.0
            if total:
                extra["sebeni/reward"] = float(total)
        log_trackio(extra, step=getattr(state, "global_step", None))


def emit_kveritas_line(name: str, value: float, step: int = 0) -> None:
    """Print standard K-Veritas protocol metric line."""
    if value is not None:
        try:
            val = float(value)
            print(f"KVERITAS_METRIC name={name} value={val:.6g} step={int(step)}", flush=True)
        except (ValueError, TypeError):
            pass


class KVeritasCallback(TrainerCallback):
    """Callback to stream training step rewards and eval metrics to K-Veritas stdout."""

    def __init__(self, reward_manager=None, config=None, eval_dataset=None):
        super().__init__()
        self.reward_manager = reward_manager
        self.config = config
        self.eval_dataset = eval_dataset

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = getattr(state, "global_step", 0)
        # Log reward metrics from reward_manager if available
        if self.reward_manager is not None:
            recent = getattr(self.reward_manager, "recent_rewards", {})
            for key in ["reward_total", "reward_format", "reward_morph", "reward_rule", "reward_lang"]:
                if key in recent:
                    emit_kveritas_line(key, recent[key], step=step)
            # If reward_total not in recent, fallback to total_reward
            if "reward_total" not in recent:
                total = getattr(self.reward_manager, "total_reward", 0.0) or 0.0
                if total:
                    emit_kveritas_line("reward_total", total, step=step)

        # Also forward any logs metrics matching reward or loss
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)) and ("reward" in k or "loss" in k):
                    metric_name = k.replace("/", "_")
                    if not metric_name.startswith("reward_") and not metric_name.endswith("_loss"):
                        emit_kveritas_line(metric_name, float(v), step=step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = getattr(state, "global_step", 0)
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_name = k.replace("/", "_")
                    emit_kveritas_line(clean_name, float(v), step=step)

        # Compute phi, mer, mcs, uwec on eval_dataset if present
        if self.eval_dataset and len(self.eval_dataset) > 0 and self.config is not None:
            try:
                from beni.core.pipeline import run_eval
                eval_records = []
                for row in self.eval_dataset:
                    text_val = ""
                    lang_val = ""
                    if isinstance(row, dict) or hasattr(row, "get"):
                        text_val = row.get("text") or ""
                        if not text_val:
                            p = row.get("prompt")
                            if isinstance(p, list):
                                for msg in reversed(p):
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        text_val = msg.get("content", "")
                                        break
                                if not text_val and p and isinstance(p[-1], dict):
                                    text_val = p[-1].get("content", "")
                            elif isinstance(p, str):
                                text_val = p
                        lang_val = row.get("lang") or row.get("language") or ""
                    eval_records.append({"text": text_val, "lang": lang_val})

                if eval_records:
                    report = run_eval(self.config, eval_records)
                    for metric_key in ["phi", "mer", "mcs", "uwec"]:
                        val = report.get(metric_key)
                        if val is not None:
                            emit_kveritas_line(metric_key, float(val), step=step)
                    by_lang = report.get("by_language") or {}
                    for lang, lang_metrics in by_lang.items():
                        for metric_key in ["phi", "mer", "mcs", "uwec"]:
                            lval = (lang_metrics or {}).get(metric_key)
                            if lval is not None:
                                emit_kveritas_line(f"{metric_key}_{lang}", float(lval), step=step)
            except Exception:
                pass