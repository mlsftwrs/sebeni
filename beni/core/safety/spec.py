"""SafetySpec — default-on linguistic integrity and reward-distrust contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SafetySpec:
    """Hard gates wrapping SAMPG promote, policy update, and Hub export.

    Encodes ELRL morphological integrity plus reward distrust:
    noisy ``π_rf`` must not update ``θ`` unchecked.

    Parameters
    ----------
    enabled : bool
        Master switch. When False, gates log but do not block.
    require_language : bool
        Completions must declare a language identity matching **that row**.
    refuse_mixed_language : bool
        Refuse updates driven by completions that declare more than one
        language (several ``lang`` fields, or one listing several codes).
        Mixed-language **datasets** (one language per row) are allowed;
        mixing languages inside a single completion is not.
    tau : float
        SAMPG morphological integrity threshold Φ.
    require_phi_improve : bool
        Distiller may promote ``{G, D}`` only when Φ′ > Φ (after the first-create gate).
    max_mer : float
        Upper bound on morpheme error rate vs a reference when present.
    format_invalid_blocks_update : bool
        Format-invalid batches cannot update the policy (SAMPG ``R_format``).
    uncertainty_downweight : bool
        Scale gradients by uncertainty ``U`` (indicator + β log π_θ/π_ref).
    kl_beta : float
        KL-to-ref coefficient (TRL ``beta``).
    max_kl : float, optional
        If set, block the update when mean KL exceeds this bound.
    require_model_card : bool
        Hub push requires a generated README model card.
    require_safety_snapshot : bool
        Hub push requires a recorded eval snapshot (Φ, τ, checkpoint id).
    """

    enabled: bool = True
    require_language: bool = True
    refuse_mixed_language: bool = True
    tau: float = 0.5
    require_phi_improve: bool = True
    max_mer: float = 1.0
    format_invalid_blocks_update: bool = True
    uncertainty_downweight: bool = True
    kl_beta: float = 0.1
    max_kl: Optional[float] = None
    require_model_card: bool = True
    require_safety_snapshot: bool = True
    fired: List[str] = field(default_factory=list)

    def reset_fired(self) -> None:
        self.fired.clear()
