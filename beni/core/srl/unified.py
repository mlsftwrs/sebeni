"""Unified SAMPG driver wrapping policy-update plugins.

``SRLTrainer`` trains one policy arm against frozen grammar and dictionary.
Resource distillation runs once, upstream, via :mod:`beni.core.pipeline`.

Plugins only replace the policy-update step. Register a new method
with :func:`register_algorithm`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from beni.core.safety.governor import SafetyGovernor
from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg


class AlignmentAlgorithm(Protocol):
    """Policy-update plugin: load models, train, save, push."""

    name: str

    def load_models(self, *args, **kwargs):
        ...

    def train(self, data, **kwargs):
        ...

    def save_model(self, output_dir=None):
        ...

    def push_to_hub(self, repo_id=None, **kwargs):
        ...


class DataSource(Protocol):
    """SAMPG samples o ~ π_θ(·|B). Offline pairs use scheme=preference."""

    def ds(self, *args, **kwargs):
        ...


class RewardFunction(Protocol):
    """π_rf: R_morph + R_format (JSON), plus optional R_rule / R_lang."""

    def get_reward_functions(self) -> list:
        ...


_ALGORITHM_REGISTRY: Dict[str, type] = {}


def register_algorithm(name: str, cls: type) -> None:
    """Register a policy-update plugin (one class + config, not a new outer loop)."""
    _ALGORITHM_REGISTRY[str(name).lower()] = cls


def get_algorithm(name: str) -> type:
    key = str(name or "grpo").lower()
    if key not in _ALGORITHM_REGISTRY:
        _load_builtins()
    if key not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm {name!r}. Registered: {sorted(_ALGORITHM_REGISTRY)}"
        )
    return _ALGORITHM_REGISTRY[key]


def _load_builtins() -> None:
    if "grpo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.grpo.grpo import SebeniGrpo

        register_algorithm("grpo", SebeniGrpo)
    if "dpo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.dpo.dpo import SebeniDpo

        register_algorithm("dpo", SebeniDpo)
    if "apo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.apo.apo import SebeniApo

        register_algorithm("apo", SebeniApo)
    if "sft" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.sft.sft import SebeniSft

        register_algorithm("sft", SebeniSft)


class SRLTrainer:
    """SAMPG driver. Not an alias of ``SebeniGrpo``.

    Parameters
    ----------
    config : MasterConfig
        Loaded from YAML via ``MasterConfig.from_yaml``.
    """

    def __init__(self, config: Optional[MasterConfig] = None, **kwargs: Any):
        self.config = config or MasterConfig()
        if kwargs.get("model_name"):
            self.config.model.model_name = kwargs["model_name"]
        spec = self.config.safety.to_spec(
            tau=self.config.distillation.tau,
            kl_beta=self.config.trainer.beta,
        )
        self.governor = SafetyGovernor(spec)
        framework = str(getattr(self.config.trainer, "framework", "torch")).lower()
        if framework == "jax" and str(self.config.algorithm).lower() == "sft":
            raise ValueError(
                "SFT is torch-only. Set trainer.framework: torch for algorithm: sft."
            )
        if framework == "jax":
            from beni.core.srl.jax import JaxPolicyPlugin

            plugin_cls = JaxPolicyPlugin
        else:
            plugin_cls = get_algorithm(self.config.algorithm)
        self.plugin = plugin_cls(self.config)
        self.plugin.governor = self.governor

    def __getattr__(self, name: str):
        return getattr(self.plugin, name)

    def _records(self, data) -> List[Dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, list):
            return data
        try:
            return [{"text": row.get("text", ""), "lang": row.get("language") or row.get("lang")} for row in data]
        except TypeError:
            return []

    def train(self, data=None, project_name: Optional[str] = None, **kwargs):
        """Train the selected arm. Distill first only when resources are not frozen."""
        if self.config.working_dir:
            cfg.set_working_dir(self.config.working_dir)
        records = self._records(data)
        from beni.core.pipeline import resources_frozen, run_distill

        if records and not resources_frozen(self.config):
            run_distill(self.config, records)
        return self.plugin.train(
            data,
            project_name=project_name,
            run_distillation_first=False,
            extra_callbacks=kwargs.get("extra_callbacks"),
        )
