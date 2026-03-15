"""
topology/ripser_filtration.py
=============================
Vietoris-Rips filtration via Ripser and Betti number computation.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum point cloud size for meaningful persistence
_MIN_POINTS = 5
_INF_EPSILON = 1e-3  # Small offset to replace inf deaths


def compute_persistence_diagram(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    max_edge: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Compute the Vietoris-Rips persistence diagram of a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (N, d)
        Points in R^d.
    max_dim : int
        Maximum homological dimension (default 1 → H0 and H1).
    max_edge : float
        Maximum filtration radius.

    Returns
    -------
    diagram : dict
        ``{'dgm_0': array of (birth, death), 'dgm_1': array of (birth, death)}``
        Infinite deaths are replaced with ``max_finite_death + epsilon``.
    """
    from ripser import ripser  # lazy import for optional dependency

    n_points = point_cloud.shape[0]
    empty_dgm: Dict[str, np.ndarray] = {
        f"dgm_{k}": np.empty((0, 2), dtype=np.float64) for k in range(max_dim + 1)
    }

    if n_points < _MIN_POINTS:
        logger.warning(
            "Point cloud has only %d points (< %d). Returning empty diagrams.",
            n_points,
            _MIN_POINTS,
        )
        return empty_dgm

    result = ripser(
        point_cloud.astype(np.float64),
        maxdim=max_dim,
        thresh=max_edge,
    )

    diagram: Dict[str, np.ndarray] = {}
    for k in range(max_dim + 1):
        dgm = result["dgms"][k].copy()
        if dgm.size == 0:
            diagram[f"dgm_{k}"] = np.empty((0, 2), dtype=np.float64)
            continue

        # Replace inf deaths with max finite death + epsilon
        finite_mask = np.isfinite(dgm[:, 1])
        if finite_mask.any():
            max_finite = dgm[finite_mask, 1].max()
        else:
            max_finite = max_edge
        inf_mask = ~finite_mask
        dgm[inf_mask, 1] = max_finite + _INF_EPSILON

        diagram[f"dgm_{k}"] = dgm

    n_h0 = len(diagram.get("dgm_0", []))
    n_h1 = len(diagram.get("dgm_1", []))
    logger.debug("Persistence diagram: H0=%d features, H1=%d features", n_h0, n_h1)
    return diagram


def compute_betti_numbers(
    diagram: Dict[str, np.ndarray],
    epsilon: float,
) -> Tuple[int, int]:
    """Compute Betti numbers β₀ and β₁ at filtration scale ε.

    Parameters
    ----------
    diagram : dict
        As returned by ``compute_persistence_diagram``.
    epsilon : float
        Filtration scale at which to count alive features.

    Returns
    -------
    beta_0 : int
        Number of connected components alive at ε.
    beta_1 : int
        Number of 1-dimensional loops alive at ε.
    """
    def _count_alive(dgm: np.ndarray, eps: float) -> int:
        if dgm.size == 0:
            return 0
        alive = (dgm[:, 0] <= eps) & (dgm[:, 1] > eps)
        return int(alive.sum())

    dgm_0 = diagram.get("dgm_0", np.empty((0, 2)))
    dgm_1 = diagram.get("dgm_1", np.empty((0, 2)))

    beta_0 = _count_alive(dgm_0, epsilon)
    beta_1 = _count_alive(dgm_1, epsilon)

    return beta_0, beta_1
