"""SafetyGovernor — consult before SAMPG promote, policy step, and Hub push."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import torch
except ImportError:
    torch = None

from beni.core.safety.spec import SafetySpec

logger = logging.getLogger(__name__)


@dataclass
class PromoteDecision:
    """Result of a SAMPG grammar/dictionary promote gate."""

    allowed: bool
    reason: str
    phi: float = 0.0
    phi_prime: float = 0.0
    parseable: bool = False
    first_create: bool = False


@dataclass
class SafetySnapshot:
    """Release-integrity record written next to the model card."""

    phi: Optional[float] = None
    tau: float = 0.5
    checkpoint_id: Optional[str] = None
    mer: Optional[float] = None
    mcs: Optional[float] = None
    format_validity: Optional[float] = None
    r_lang: Optional[float] = None
    u_indicator: Optional[float] = None
    u_kl: Optional[float] = None
    uncertainty: Optional[float] = None
    algorithm: Optional[str] = None
    language: Optional[str] = None
    group_code: Optional[str] = None
    gates_fired: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data

    def write(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SafetySnapshot":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in SafetySnapshot.__dataclass_fields__.values()}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["extra"] = extra
        return cls(**kwargs)


class SafetyGovernor:
    """Hard gates for promote, policy update, and Hub export.

    Parameters
    ----------
    spec : SafetySpec
        Default-on safety contract. Built from ``MasterConfig.safety`` plus τ/β.
    """

    SNAPSHOT_NAME = "safety_snapshot.json"
    CARD_NAME = "README.md"

    def __init__(self, spec: Optional[SafetySpec] = None):
        self.spec = spec or SafetySpec()
        self.snapshot: Optional[SafetySnapshot] = None

    def _fire(self, gate: str) -> None:
        if gate not in self.spec.fired:
            self.spec.fired.append(gate)
        logger.info("SafetyGovernor fired gate: %s", gate)

    def allow_promote(
        self,
        phi: float,
        phi_prime: float,
        *,
        parseable: bool = False,
        first_create: bool = False,
        language_ok: bool = True,
        mer: Optional[float] = None,
    ) -> PromoteDecision:
        """Decide whether Distiller may write ``baseline_vN``.

        Scratch first-create uses the parseable-file gate. Every later checkpoint
        uses SAMPG: promote iff Φ′ > Φ, plus language/integrity gates.
        """
        if not self.spec.enabled:
            return PromoteDecision(True, "safety_disabled", phi, phi_prime, parseable, first_create)

        if self.spec.require_language and not language_ok:
            self._fire("language_mismatch")
            return PromoteDecision(False, "language_mismatch", phi, phi_prime, parseable, first_create)

        if first_create:
            if not parseable:
                self._fire("unparseable_bootstrap")
                return PromoteDecision(False, "unparseable_bootstrap", phi, phi_prime, parseable, True)
            return PromoteDecision(True, "first_create", phi, phi_prime, parseable, True)

        if not parseable:
            self._fire("unparseable_candidate")
            return PromoteDecision(False, "unparseable_candidate", phi, phi_prime, False, False)

        if self.spec.require_phi_improve and not (phi_prime > phi):
            self._fire("phi_not_improved")
            return PromoteDecision(False, "phi_not_improved", phi, phi_prime, parseable, False)

        if mer is not None and mer > self.spec.max_mer:
            self._fire("mer_exceeded")
            return PromoteDecision(False, "mer_exceeded", phi, phi_prime, parseable, False)

        return PromoteDecision(True, "phi_improved", phi, phi_prime, parseable, False)

    def allow_policy_update(self, batch_meta: Dict[str, Any]) -> bool:
        """Return False when the batch must not update θ.

        ``batch_meta["signal_source"]`` says what the per-item signals describe:

        - ``"samples"`` (GRPO): the policy's own completions. Bad completions are
          pushed down by their reward, so the batch is refused only when **no**
          completion is good (all format-invalid, none in the right language,
          or every language-declaring completion mixes languages).
        - ``"targets"`` (SFT labels, DPO/APO chosen answers): the gradient pulls
          θ towards them, so **any** invalid, wrong-language or mixed target
          refuses the batch.

        Per-item signals: ``format_scores`` (0.0 = invalid), ``lang_ok``
        (True / False / None = no language declared), ``mixed_language``.
        A legacy boolean ``language_ok`` is still honoured. ``kl`` is checked
        against ``max_kl``.
        """
        if not self.spec.enabled:
            return True
        strict = batch_meta.get("signal_source") == "targets"

        def _refuse(flags: List[bool]) -> bool:
            return bool(flags) and (any(flags) if strict else all(flags))

        format_scores = batch_meta.get("format_scores") or []
        if self.spec.format_invalid_blocks_update and _refuse([float(s) == 0.0 for s in format_scores]):
            self._fire("format_invalid_batch")
            return False

        if self.spec.require_language:
            known = [ok for ok in (batch_meta.get("lang_ok") or []) if ok is not None]
            if batch_meta.get("language_ok") is False or _refuse([not ok for ok in known]):
                self._fire("language_mismatch_batch")
                return False

        if self.spec.refuse_mixed_language:
            declared = [
                bool(mixed)
                for mixed, ok in zip(batch_meta.get("mixed_language") or [], batch_meta.get("lang_ok") or [])
                if ok is not None
            ]
            if _refuse(declared):
                self._fire("mixed_language_batch")
                return False

        kl = batch_meta.get("kl")
        if self.spec.max_kl is not None and kl is not None and float(kl) > self.spec.max_kl:
            self._fire("kl_bound")
            return False

        return True

    def apply_pre_update(self, model, batch_meta: Dict[str, Any]) -> bool:
        """Zero or scale gradients according to format / uncertainty / KL gates.

        Returns
        -------
        bool
            True if the optimizer step may proceed.
        """
        allowed = self.allow_policy_update(batch_meta)
        if not allowed:
            if model is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            return False

        if not self.spec.enabled or model is None:
            return True

        if self.spec.uncertainty_downweight:
            scale = self._uncertainty_scale(batch_meta)
            if scale != 1.0:
                self._fire("uncertainty_downweight")
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale)
        return True

    def _uncertainty_scale(self, batch_meta: Dict[str, Any]) -> float:
        u = batch_meta.get("uncertainty")
        if u is None:
            model_logps = batch_meta.get("model_logps")
            ref_logps = batch_meta.get("ref_logps")
            log_ratio = batch_meta.get("log_ratio")
            if model_logps is not None and ref_logps is not None:
                u_kl = _mean_log_ratio(model_logps, ref_logps, self.spec.kl_beta)
            elif log_ratio is not None:
                # Pre-aggregated mean log(π_θ / π_ref), e.g. from DPO's logged rewards.
                u_kl = self.spec.kl_beta * float(log_ratio)
            else:
                return 1.0
            indicator = _mean_stage_indicator(batch_meta.get("predicted_stages"))
            u = indicator + u_kl
            batch_meta["u_indicator"] = indicator
            batch_meta["u_kl"] = u_kl
            batch_meta["uncertainty"] = u
        try:
            return 1.0 / (1.0 + max(float(u), 0.0))
        except (TypeError, ValueError):
            return 1.0

    def record_snapshot(self, **kwargs: Any) -> SafetySnapshot:
        """Store a safety eval snapshot for the model card / Hub gate."""
        kwargs.setdefault("tau", self.spec.tau)
        kwargs.setdefault("gates_fired", list(self.spec.fired))
        self.snapshot = SafetySnapshot(**kwargs)
        return self.snapshot

    def write_snapshot(self, directory: Union[str, Path]) -> Path:
        if self.snapshot is None:
            self.record_snapshot()
        return self.snapshot.write(Path(directory) / self.SNAPSHOT_NAME)

    def allow_hub_push(self, output_dir: Union[str, Path]) -> bool:
        """Refuse Hub upload without a model card and an eval safety snapshot on disk.

        Both files must be in ``output_dir`` (the weights being pushed). With
        ``require_eval_metrics`` the snapshot must carry Φ and MER or MCS,
        i.e. it was written by ``sebeni eval`` / ``run_eval`` for these weights.
        """
        output_dir = Path(output_dir)
        if not self.spec.enabled:
            return True
        card = output_dir / self.CARD_NAME
        snap = output_dir / self.SNAPSHOT_NAME
        if self.spec.require_model_card and not card.exists():
            self._fire("missing_model_card")
            return False
        if self.spec.require_safety_snapshot and not snap.exists():
            self._fire("missing_safety_snapshot")
            return False
        if self.spec.require_eval_metrics:
            try:
                data = json.loads(snap.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if data.get("phi") is None or (data.get("mer") is None and data.get("mcs") is None):
                self._fire("missing_eval_metrics")
                return False
            # The eval must have scored exactly these weight files.
            recorded = data.get("weights_sha256")
            if not recorded or recorded != weights_fingerprint(output_dir):
                self._fire("snapshot_weights_mismatch")
                return False
        return True


WEIGHT_PATTERNS = ("adapter_model*.safetensors", "adapter_model*.bin", "model*.safetensors", "pytorch_model*.bin")


def weights_fingerprint(directory: Union[str, Path]) -> Optional[str]:
    """sha256 over the weight files in ``directory`` (names and bytes); None when there are none."""
    import hashlib

    root = Path(directory)
    if not root.is_dir():
        return None
    files = sorted({p for pattern in WEIGHT_PATTERNS for p in root.glob(pattern) if p.is_file()})
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _mean_log_ratio(model_logps, ref_logps, beta: float) -> float:
    """β-weighted mean log(π_θ / π_ref) used as a distrust term."""
    try:
        if torch is not None:
            m = model_logps if torch.is_tensor(model_logps) else torch.as_tensor(model_logps)
            r = ref_logps if torch.is_tensor(ref_logps) else torch.as_tensor(ref_logps)
            ratio = (m.detach().float() - r.detach().float()).mean().item()
            return beta * ratio
        import numpy as np
        m = np.asarray(model_logps, dtype=float)
        r = np.asarray(ref_logps, dtype=float)
        return beta * float((m - r).mean())
    except Exception:
        return 0.0


def _mean_stage_indicator(predicted_stages) -> float:
    """Mean I(stage != -1) over morphological tokens, separate from LM tokens."""
    if predicted_stages is None:
        return 0.0
    flat = []
    for stages in predicted_stages:
        values = stages if isinstance(stages, (list, tuple)) else [stages]
        for stage in values:
            try:
                flat.append(1.0 if int(stage) != -1 else 0.0)
            except (TypeError, ValueError):
                flat.append(1.0)
    return sum(flat) / len(flat) if flat else 0.0
