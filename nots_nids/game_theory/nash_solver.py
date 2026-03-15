"""
game_theory/nash_solver.py
==========================
Linear Programming formulation of the minimax Nash equilibrium over feature
subsets.  Uses CVXPY for the optimisation.

The Nash equilibrium S* gives the defender (NIDS) a randomised feature-subset
sampling strategy that guarantees a minimum detection payoff ε_min regardless
of attacker behaviour.
"""

import logging
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def solve_nash_equilibrium(
    feature_importance_matrix: np.ndarray,
    n_subsets: int,
) -> Dict[str, Any]:
    """Solve the zero-sum game LP for the defender's Nash-optimal strategy.

    The payoff matrix ``U`` has shape ``(n_subsets, n_attacker_responses)``.
    Entry ``U[i, j]`` is the detection payoff (mean Wasserstein distance) when
    the defender uses feature subset *i* and the attacker plays response *j*.

    LP formulation (CVXPY)::

        maximise    v
        subject to  ∑_i  p[i] · U[i, j]  ≥  v   ∀ j
                    ∑_i  p[i]  =  1
                    p  ≥  0

    Parameters
    ----------
    feature_importance_matrix : np.ndarray, shape (n_subsets, n_responses)
        Payoff matrix.
    n_subsets : int
        Number of feature subsets (rows of the payoff matrix).

    Returns
    -------
    result : dict
        ``{'S_star': np.ndarray,         # Nash distribution over subsets
            'epsilon_min': float,        # Guaranteed detection floor
            'subset_indices': None}``    # Placeholder (set by caller)
    """
    U = np.asarray(feature_importance_matrix, dtype=np.float64)
    n_sub, n_resp = U.shape
    assert n_sub == n_subsets, (
        f"Payoff matrix rows ({n_sub}) != n_subsets ({n_subsets})"
    )

    # ── Define LP ────────────────────────────────────────────────────────
    p = cp.Variable(n_sub, nonneg=True)
    v = cp.Variable()

    constraints = [cp.sum(p) == 1]
    for j in range(n_resp):
        constraints.append(U[:, j] @ p >= v)

    objective = cp.Maximize(v)
    problem = cp.Problem(objective, constraints)

    # ── Solve ────────────────────────────────────────────────────────────
    try:
        problem.solve(solver=cp.ECOS)
    except cp.SolverError:
        logger.warning("ECOS failed, falling back to SCS solver")
        problem.solve(solver=cp.SCS)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        logger.error("Nash LP solve status: %s — returning uniform distribution",
                      problem.status)
        S_star = np.ones(n_sub) / n_sub
        epsilon_min = float(U.min())
    else:
        S_star = np.array(p.value).flatten()
        # Clamp negatives from numerical noise and re-normalise
        S_star = np.maximum(S_star, 0.0)
        S_star /= S_star.sum()
        epsilon_min = float(v.value)

    logger.info("Nash equilibrium solved: ε_min = %.6f", epsilon_min)
    logger.info("S* (top-5 subsets): %s", np.argsort(S_star)[-5:][::-1])
    logger.info("S* weights (top-5): %s", np.sort(S_star)[-5:][::-1])

    return {
        "S_star": S_star,
        "epsilon_min": epsilon_min,
        "subset_indices": None,  # Caller sets this
    }


def compute_feature_payoff_matrix(
    val_windows: List[Dict],
    subset_indices: List[List[int]],
    detector_fn,
    n_eval: int = 20,
) -> np.ndarray:
    """Compute the feature payoff matrix U from validation windows.

    For each feature subset and each validation attack window, compute the
    Wasserstein distance as the detection payoff.

    Parameters
    ----------
    val_windows : list[dict]
        Validation windows (preferably attack windows only).
    subset_indices : list[list[int]]
        Feature indices for each subset.
    detector_fn : callable
        ``detector_fn(points, subset) -> float``  Wasserstein distance.
    n_eval : int
        Number of validation windows to evaluate per subset.

    Returns
    -------
    U : np.ndarray, shape (n_subsets, n_eval)
    """
    n_subsets = len(subset_indices)
    n_eval = min(n_eval, len(val_windows))
    U = np.zeros((n_subsets, n_eval), dtype=np.float64)

    for i, subset in enumerate(subset_indices):
        for j in range(n_eval):
            pts = val_windows[j]["points"][:, subset]
            W = detector_fn(pts, subset)
            U[i, j] = W

    logger.info("Payoff matrix computed: shape=%s, mean=%.4f", U.shape, U.mean())
    return U
