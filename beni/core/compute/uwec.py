"""UWEC: post-training evaluation cost. Not a training gradient coefficient.

U_A(w_i, o) = I(Stage_pred(w_i) != -1) + beta * |log((pi_theta + eps) / (pi_ref + eps))|

UWEC is the token mean of that cost. Reductions run on NumPy or jax.numpy.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def array_module(xp: Optional[Any] = None) -> Any:
    """Return the array module for a reduction. Defaults to NumPy."""
    return np if xp is None else xp


def _as_float(xp: Any, values) -> Any:
    if values is None:
        return xp.asarray([], dtype=float)
    return xp.asarray(values, dtype=float)


def _as_stages(xp: Any, stages) -> Any:
    if stages is None:
        return xp.asarray([], dtype=float)
    out = []
    for stage in list(stages):
        try:
            out.append(float(int(stage)))
        except (TypeError, ValueError):
            out.append(0.0)
    return xp.asarray(out, dtype=float)


def _align_probs(xp: Any, probs, n: int) -> Any:
    """Align LM token probabilities to ``n`` morphological tokens.

    Same-length arrays pass through. Otherwise each morphological token gets
    the mean sequence probability so UWEC stays a token mean over stages.
    """
    arr = _as_float(xp, probs)
    if n <= 0:
        return arr
    size = int(getattr(arr, "size", len(arr)))
    if size == n:
        return arr
    if size == 0:
        return xp.ones((n,), dtype=float)
    mean = xp.mean(arr)
    return mean * xp.ones((n,), dtype=float)


def _as_probabilities(xp: Any, probs, n: int, *, log_input: bool = False) -> Any:
    """Align token scores to ``n`` morphological tokens and return probabilities.

    Negative values are log-probabilities. Non-finite scores fall back to 1
    so the log-ratio term stays defined and UWEC remains a finite token mean.
    """
    arr = _align_probs(xp, probs, n)
    if n <= 0:
        return arr
    host = np.asarray(arr)
    if not np.all(np.isfinite(host)):
        host = np.where(np.isfinite(host), host, 1.0)
        arr = xp.asarray(host, dtype=float)
    # Log-probabilities are <= 0. Exponentiate those; leave real probabilities alone.
    if host.size and (log_input or (float(np.min(host)) < 0.0 and float(np.max(host)) <= 0.0)):
        arr = xp.exp(arr)
    return xp.clip(arr, 0.0, 1.0)


def uwec(
    stages,
    pi_theta,
    pi_ref,
    *,
    beta: float = 0.1,
    eps: float = 1e-8,
    xp: Optional[Any] = None,
    log_input: bool = False,
) -> Tuple[Any, Any]:
    """Per-token UWEC and the token-mean scalar.

    Parameters
    ----------
    stages :
        Predicted morphological stages. Non ``-1`` contributes indicator 1.
    pi_theta, pi_ref :
        Policy and reference token probabilities. If ``log_input`` is true,
        values are treated as log-probabilities and exponentiated first.
    xp :
        ``numpy`` or ``jax.numpy``. Defaults to NumPy.
    """
    xp = array_module(xp)
    stage_arr = _as_stages(xp, stages)
    n = int(getattr(stage_arr, "size", len(stage_arr)))
    if n == 0:
        empty = xp.asarray([], dtype=float)
        return empty, xp.asarray(0.0, dtype=float)
    theta = _as_probabilities(xp, pi_theta, n, log_input=log_input)
    ref = _as_probabilities(xp, pi_ref, n, log_input=log_input)
    indicator = xp.where(stage_arr != -1.0, 1.0, 0.0)
    log_ratio = xp.log((theta + eps) / (ref + eps))
    per_token = indicator + float(beta) * xp.abs(log_ratio)
    return per_token, xp.mean(per_token)
