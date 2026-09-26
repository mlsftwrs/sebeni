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