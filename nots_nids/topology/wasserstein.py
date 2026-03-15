"""
topology/wasserstein.py
=======================
Wasserstein distance between persistence diagrams using persim, with timing.
"""

import functools
import logging
import time
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel distance for empty diagrams
_SENTINEL_DISTANCE = 999.0


# ── Timing decorator ────────────────────────────────────────────────────────

def _timed(func):
    """Decorator that logs wall-clock time for a function call."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.debug("%s took %.4f s", func.__name__, dt)
        return result

    return wrapper


# ── Core distance computation ────────────────────────────────────────────────

@_timed
def wasserstein_distance(
    dgm1: Dict[str, np.ndarray],
    dgm2: Dict[str, np.ndarray],
) -> float:
    """Compute the combined Wasserstein distance between two persistence diagrams.

    The total distance is ``W_H0 + W_H1``.

    Parameters
    ----------
    dgm1, dgm2 : dict
        Persistence diagrams with keys ``'dgm_0'`` and ``'dgm_1'``.

    Returns
    -------
    W_total : float
        ``W_H0 + W_H1``.  Returns ``_SENTINEL_DISTANCE`` if either diagram
        is entirely empty for both dimensions.

    Notes
    -----
    Uses ``persim.wasserstein`` which computes a fixed distance variant
    (sum of matched L2 costs via the Hungarian algorithm).
    """
    from persim import wasserstein as persim_wasserstein  # type: ignore

    W_total = 0.0
    any_valid = False

    for k in (0, 1):
        key = f"dgm_{k}"
        d1 = dgm1.get(key, np.empty((0, 2)))
        d2 = dgm2.get(key, np.empty((0, 2)))

        if d1.size == 0 and d2.size == 0:
            continue  # Both empty → distance 0 for this dimension

        if d1.size == 0 or d2.size == 0:
            # One empty, one not → large distance
            W_total += _SENTINEL_DISTANCE
            any_valid = True
            continue

        try:
            w = persim_wasserstein(d1, d2)
            W_total += w
            any_valid = True
        except Exception as e:
            logger.warning("Wasserstein computation failed for H%d: %s", k, e)
            W_total += _SENTINEL_DISTANCE
            any_valid = True

    if not any_valid:
        return _SENTINEL_DISTANCE

    return float(W_total)


# ── Trajectory computation ───────────────────────────────────────────────────

def compute_wasserstein_trajectory(
    window_diagrams: List[Dict[str, np.ndarray]],
    D_norm: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute Wasserstein distances between each live window diagram and D_norm.

    Parameters
    ----------
    window_diagrams : list[dict]
        List of persistence diagrams for live windows.
    D_norm : dict
        Baseline (normal) persistence diagram.

    Returns
    -------
    W_values : np.ndarray, shape (n_windows,)
        Array of Wasserstein distances.
    """
    W_values = np.zeros(len(window_diagrams), dtype=np.float64)
    for i, dgm in enumerate(window_diagrams):
        W_values[i] = wasserstein_distance(dgm, D_norm)
    logger.info(
        "Wasserstein trajectory: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        W_values.mean(),
        W_values.std(),
        W_values.min(),
        W_values.max(),
    )
    return W_values
