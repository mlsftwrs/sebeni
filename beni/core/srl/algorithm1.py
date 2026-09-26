"""SAMPG — self-aware morphotactic generation (per-batch Φ / τ / Distiller)."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, Sequence

from beni.core.safety.governor import PromoteDecision, SafetyGovernor

logger = logging.getLogger(__name__)


def batch_user_texts(inputs: Any) -> list[str]:
    """Extract user-side texts from a TRL / Hugging Face trainer batch."""
    return [text for text, _lang in batch_text_langs(inputs)]


def records_text_langs(
    records: Sequence[Any],
    default_lang: str = "bam",
) -> list[tuple[str, str]]:
    """``(text, lang)`` pairs from loader / train records."""
    pairs: list[tuple[str, str]] = []
    for rec in records or []:
        if isinstance(rec, dict):
            text = rec.get("text") or rec.get("prompt") or ""
            lang = rec.get("lang") or rec.get("language") or default_lang
        else:
            text = getattr(rec, "text", "") or ""
            lang = getattr(rec, "lang", None) or default_lang
        if text:
            pairs.append((str(text), str(lang or default_lang)))
    return pairs


def batch_text_langs(inputs: Any, default_lang: str = "bam") -> list[tuple[str, str]]:
    """Extract (text, language) pairs, aligned with a ``language`` column when present.

    Mixed-language batches are the normal case: each row keeps its own code so
    SAMPG can score Φ and Distill against that language's G, D.
    """
    if inputs is None:
        return []
    if isinstance(inputs, (list, tuple)):
        pairs: list[tuple[str, str]] = []
        for item in inputs:
            pairs.extend(batch_text_langs(item, default_lang=default_lang))
        return pairs
    if not isinstance(inputs, dict):
        return []

    texts: list[str] = []
    prompt = inputs.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        texts.append(prompt.strip())
    elif isinstance(prompt, (list, tuple)):
        for item in prompt:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
            elif isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if role == "user" and content:
                    texts.append(str(content).strip())
            elif isinstance(item, (list, tuple)):
                texts.extend(t for t, _ in batch_text_langs({"prompt": item}, default_lang))

    for key in ("text", "texts"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, (list, tuple)):
            texts.extend(str(v).strip() for v in value if v)

    langs_raw = inputs.get("language", inputs.get("lang"))
    langs: list[str]
    if isinstance(langs_raw, str) and langs_raw.strip():
        langs = [langs_raw.strip()]
    elif isinstance(langs_raw, (list, tuple)):
        langs = [str(v).strip() if v else default_lang for v in langs_raw]
    else:
        langs = []

    if not texts:
        return []
    if len(langs) == 1 and len(texts) > 1:
        langs = langs * len(texts)
    while len(langs) < len(texts):
        langs.append(default_lang)
    return list(zip(texts, langs[: len(texts)]))


def group_texts_by_language(
    pairs: Sequence[tuple[str, str]],
    default_lang: str = "bam",
) -> Dict[str, list[str]]:
    """Bucket texts by Sebeni group code (MKU not MLQ)."""
    from beni.core.language import Language

    grouped: Dict[str, list[str]] = defaultdict(list)
    for text, lang in pairs:
        if not text:
            continue
        grouped[Language.from_code(lang or default_lang).group_code].append(text)
    return dict(grouped)


def maybe_distill_languages(
    pairs: Sequence[tuple[str, str]],
    distiller_for: Callable[[str], Any],
    governor: SafetyGovernor,
    tau: float,
    default_lang: str = "bam",
) -> Dict[str, PromoteDecision]:
    """Run SAMPG Φ / Distiller **per language** in a mixed batch.

    Each group uses its own Distiller checkpoint ``{G, D}``. Φ < τ triggers
    Distiller only for that language.
    """
    decisions: Dict[str, PromoteDecision] = {}
    for group, texts in group_texts_by_language(pairs, default_lang=default_lang).items():
        decisions[group] = maybe_distill_batch(texts, distiller_for(group), governor, tau)
    return decisions


def confirm_hitl(gram_text: str, dict_text: str, enabled: bool) -> bool:
    """Human-in-the-loop gate for the initial Distiller write.

    Skipped when ``enabled`` is False, or when stdin is not a TTY (CI).
    """
    if not enabled:
        return True
    if not sys.stdin.isatty():
        logger.info("HITL requested but stdin is not a TTY; continuing.")
        return True
    print("\n--- Proposed grammar (head) ---")
    print((gram_text or "")[:800])
    print("\n--- Proposed dictionary (head) ---")
    print((dict_text or "")[:800])
    answer = input("\nPromote this baseline? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def distill_language(
    texts: Sequence[str],
    distiller: Any,
    governor: SafetyGovernor,
    tau: float,
    *,
    hitl: bool = False,
) -> PromoteDecision:
    """Run SAMPG once on a language's full train split, then stop.

    Φ ← DabaX(texts, G, D). If Φ < τ, Distiller proposes candidates. A scratch
    bootstrap may be written when it parses. Every later checkpoint, including
    the first update of a packaged baseline, is written only when Φ′ > Φ.
    """
    return _promote_if_below_tau(texts, distiller, governor, tau, hitl=hitl)


def maybe_distill_batch(
    texts: Sequence[str],
    distiller: Any,
    governor: SafetyGovernor,
    tau: float,
) -> PromoteDecision:
    """Run SAMPG Φ vs τ on a batch.

    Φ ← DabaX(B, G, D). If Φ < τ, Distiller proposes candidates; promote iff
    Φ′ > Φ (or the scratch first-create parse gate) and SafetyGovernor agrees.

    Parameters
    ----------
    texts : sequence of str
        Batch strings (the same data the policy step will train on).
    distiller : Distiller
        Distiller with ``phi_on_texts`` / ``propose`` / ``write_checkpoint``.
    governor : SafetyGovernor
        Promote gate (language, parse, Φ′ > Φ).
    tau : float
        Morphological integrity threshold (default 0.5).

    Returns
    -------
    PromoteDecision
        ``reason="phi_above_tau"`` means Distiller was skipped.
    """
    return _promote_if_below_tau(texts, distiller, governor, tau, hitl=False)


def _promote_if_below_tau(
    texts: Sequence[str],
    distiller: Any,
    governor: SafetyGovernor,
    tau: float,
    *,
    hitl: bool = False,
) -> PromoteDecision:
    texts = [t for t in texts if t]
    if not texts:
        return PromoteDecision(False, "empty_batch", 0.0, 0.0)

    phi = float(distiller.phi_on_texts(texts))
    if phi >= tau:
        logger.info("SAMPG: Φ=%.4f ≥ τ=%.4f; skip Distiller", phi, tau)
        return PromoteDecision(False, "phi_above_tau", phi, phi, True, False)

    proposal = distiller.propose(texts, current_phi=phi)
    if proposal is None:
        scratch = bool(getattr(distiller, "is_scratch", lambda: False)())
        return PromoteDecision(False, "no_proposal", phi, phi, False, scratch)

    if hitl and not confirm_hitl(proposal.gram_text, proposal.dict_text, True):
        return PromoteDecision(
            False, "hitl_rejected", proposal.phi, proposal.phi_prime, proposal.parseable, proposal.first_create
        )

    decision = governor.allow_promote(
        proposal.phi,
        proposal.phi_prime,
        parseable=proposal.parseable,
        first_create=proposal.first_create,
        language_ok=proposal.language_ok,
        mer=proposal.mer,
    )
    if decision.allowed:
        distiller.write_checkpoint(proposal.gram_text, proposal.dict_text)
        logger.info(
            "SAMPG: promoted Φ=%.4f → Φ′=%.4f (%s)",
            proposal.phi,
            proposal.phi_prime,
            decision.reason,
        )
    else:
        logger.info("SAMPG: keep G,D (%s)", decision.reason)
    return decision
