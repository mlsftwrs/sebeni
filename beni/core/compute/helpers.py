
import numpy as np
from typing import Any, Collection, FrozenSet, List, Optional, Tuple, Union
from beni.core import Token


def array_module(xp: Optional[Any] = None) -> Any:
    """Return the array module for a reduction. Defaults to NumPy."""
    return np if xp is None else xp


def levenshtein_matrix(ref: List[str], hyp: List[str], xp: Optional[Any] = None) -> Any:
    """Build a Levenshtein matrix on ``numpy`` or ``jax.numpy``."""
    xp = array_module(xp)
    m, n = len(ref), len(hyp)
    rows = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        rows[i][0] = i
    for j in range(n + 1):
        rows[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
    return xp.asarray(rows, dtype=xp.int32)


def mer_edit_counts(ref_forms, hyp_forms, xp: Optional[Any] = None) -> Tuple[Any, Any]:
    """Return ``(S_m + D_m + I_m, N_m)`` as array-module scalars."""
    xp = array_module(xp)
    ref = [str(x) for x in list(ref_forms or [])]
    hyp = [str(x) for x in list(hyp_forms or [])]
    m, n = len(ref), len(hyp)
    if m == 0 and n == 0:
        return xp.asarray(0, dtype=xp.int32), xp.asarray(0, dtype=xp.int32)
    ld = levenshtein_matrix(ref, hyp, xp=xp)
    s_m, d_m, i_m = _traceback_ld_ops(ld, ref, hyp)
    ops = xp.asarray(s_m + d_m + i_m, dtype=xp.int32)
    n_ref = xp.asarray(m, dtype=xp.int32)
    return ops, n_ref


def mer_micro(ref_forms, hyp_forms, xp: Optional[Any] = None) -> Any:
    """Morpheme error rate ``(S_m + D_m + I_m) / N_m``."""
    xp = array_module(xp)
    ops, n_ref = mer_edit_counts(ref_forms, hyp_forms, xp=xp)
    n_val = int(n_ref)
    if n_val == 0:
        return xp.asarray(0.0 if int(ops) == 0 else float("inf"), dtype=float)
    return xp.asarray(ops, dtype=float) / xp.asarray(n_ref, dtype=float)


def mcs_mismatch(pred_stages, ref_stages, xp: Optional[Any] = None) -> Any:
    """Stage mismatch rate. Unpaired tokens count as mismatches."""
    counts = mcs_mismatch_counts(pred_stages, ref_stages, xp=xp)
    n_val = int(counts[1])
    if n_val == 0:
        return array_module(xp).asarray(0.0, dtype=float)
    return counts[0] / counts[1]


def mcs_mismatch_counts(pred_stages, ref_stages, xp: Optional[Any] = None) -> Tuple[Any, Any]:
    """Return ``(mismatches, n_tokens)`` using max(pred, ref) length."""
    matches, n = mcs_match_counts(pred_stages, ref_stages, xp=xp)
    return n - matches, n


def mcs_match_counts(pred_stages, ref_stages, xp: Optional[Any] = None) -> Tuple[Any, Any]:
    """Return ``(matches, n_tokens)`` for the morphosyntactic consistency score.

    MCS = matches / n, with n = max(pred, ref). Unpaired tokens are not matches.
    """
    xp = array_module(xp)
    pred = [str(s) for s in list(pred_stages or [])]
    ref = [str(s) for s in list(ref_stages or [])]
    n = max(len(pred), len(ref))
    if n == 0:
        return xp.asarray(0.0, dtype=float), xp.asarray(0.0, dtype=float)
    matches = 0
    for i in range(min(len(pred), len(ref))):
        if pred[i] == ref[i]:
            matches += 1
    return xp.asarray(matches, dtype=float), xp.asarray(n, dtype=float)


def _traceback_ld_ops(ld, ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    """Trace a Levenshtein matrix to substitution / deletion / insertion counts."""
    grid = np.asarray(ld).tolist()
    m, n = len(ref), len(hyp)
    i, j = m, n
    s_m, d_m, i_m = 0, 0, 0
    while i > 0 or j > 0:
        if i == 0 or j == 0:
            if i == 0:
                i_m += 1
            else:
                d_m += 1
            break
        cost = 0 if ref[i - 1] == hyp[j - 1] else 1
        if grid[i][j] == grid[i - 1][j - 1] + cost:
            if cost == 1:
                s_m += 1
            i -= 1
            j -= 1
        elif grid[i][j] == grid[i - 1][j] + 1:
            d_m += 1
            i -= 1
        else:
            i_m += 1
            j -= 1
    return s_m, d_m, i_m


def to_morpheme_list(source: Union[Token, List[str]]) ->List[str]:
    """ Normalize input to list of morpheme strings """
    if isinstance(source, Token):
        return source.best_morphemes()
    return list(source)

def is_empr_token(token: Any) -> bool:
    """Return True when any analysis on ``token`` carries the ``EMPR`` borrowing gloss."""
    for analysis in getattr(token, "analyses", None) or []:
        if str(getattr(analysis, "gloss", "") or "").strip().upper() == "EMPR":
            return True
        for morph in getattr(analysis, "morphemes", None) or []:
            if str(getattr(morph, "gloss", "") or "").strip().upper() == "EMPR":
                return True
    return False


def _resolve_empr_stages(
    empr_stages: Optional[Collection[int]] = None,
    lang: Optional[str] = None,
) -> FrozenSet[int]:
    if empr_stages is not None:
        return frozenset(int(s) for s in empr_stages)
    if lang:
        from beni.core.morphotactic.dabax import empr_stages_for_lang

        return empr_stages_for_lang(lang)
    return frozenset({6})


def stage_to_phi(
    stage: Union[int, str],
    empr: float = 0.5,
    eps: float = 1e-8,
    *,
    empr_stages: Optional[Collection[int]] = None,
    lang: Optional[str] = None,
    token: Any = None,
) -> float:
    """Map morphological stage (or EMPR borrowing token) to its Φ contribution."""
    if isinstance(stage, str):
        try:
            stage = int(stage)
        except (ValueError, TypeError):
            if token is not None and is_empr_token(token):
                return float(empr)
            return 1.0

    if stage < 0:
        return 0.0
    if token is not None and is_empr_token(token):
        return float(empr)
    stages = _resolve_empr_stages(empr_stages, lang)
    if stage in stages:
        return float(empr)
    if stages and stage > max(stages):
        return float(eps)
    return 1.0


def recognized_word(
    stage: Union[int, str],
    *,
    empr_stages: Optional[Collection[int]] = None,
    lang: Optional[str] = None,
    token: Any = None,
) -> bool:
    """Check if a word is recognized by the dictionary/grammar (not unanalyzed or EMPR)."""
    if token is not None and is_empr_token(token):
        return False

    if isinstance(stage, str):
        try:
            stage = int(stage)
        except (ValueError, TypeError):
            return True

    if stage < 0:
        return False
    stages = _resolve_empr_stages(empr_stages, lang)
    return stage not in stages and (not stages or stage <= max(stages))
