from beni.core.srl.grpo.callbacks import (
    PreUpdateHookManager,
    TrackioMetricsCallback,
    SelfAwareCallback,
    format_gradient_mask_hook,
    distillation_hook,
)

__all__ = [
    "SebeniGrpo",
    "PreUpdateHookManager",
    "TrackioMetricsCallback",
    "SelfAwareCallback",
    "format_gradient_mask_hook",
    "distillation_hook",
]


def __getattr__(name: str):
    if name == "SebeniGrpo":
        from beni.core.srl.grpo.grpo import SebeniGrpo

        return SebeniGrpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
