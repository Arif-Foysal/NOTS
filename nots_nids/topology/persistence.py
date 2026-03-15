"""
topology/persistence.py
=======================
Persistence diagram utilities: distance helpers, diagram statistics, and
persistence landscape summaries.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def diagram_stats(diagram: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary statistics for a persistence diagram.

    Returns
    -------
    stats : dict
        Keys: ``n_h0``, ``n_h1``, ``max_persistence_h0``, ``max_persistence_h1``,
        ``mean_persistence_h0``, ``mean_persistence_h1``, ``total_persistence``.
    """
    stats: Dict[str, float] = {}
    total_persistence = 0.0

    for k in (0, 1):
        key = f"dgm_{k}"
        dgm = diagram.get(key, np.empty((0, 2)))
        n = len(dgm)
        stats[f"n_h{k}"] = float(n)

        if n > 0:
            pers = dgm[:, 1] - dgm[:, 0]
            stats[f"max_persistence_h{k}"] = float(pers.max())
            stats[f"mean_persistence_h{k}"] = float(pers.mean())
            total_persistence += float(pers.sum())
        else:
            stats[f"max_persistence_h{k}"] = 0.0
            stats[f"mean_persistence_h{k}"] = 0.0

    stats["total_persistence"] = total_persistence
    return stats


def persistence_vector(
    diagram: Dict[str, np.ndarray],
    max_features: int = 50,
) -> np.ndarray:
    """Create a fixed-length feature vector from a persistence diagram.

    The vector concatenates sorted persistence values (death - birth)
    for H0 and H1, zero-padded to ``max_features`` each.

    Parameters
    ----------
    diagram : dict
    max_features : int
        Maximum number of persistence values per dimension.

    Returns
    -------
    vec : np.ndarray, shape (2 * max_features,)
    """
    parts = []
    for k in (0, 1):
        dgm = diagram.get(f"dgm_{k}", np.empty((0, 2)))
        if dgm.size > 0:
            pers = np.sort(dgm[:, 1] - dgm[:, 0])[::-1]  # descending
            pers = pers[:max_features]
        else:
            pers = np.array([], dtype=np.float64)

        # Zero-pad
        padded = np.zeros(max_features, dtype=np.float64)
        padded[: len(pers)] = pers
        parts.append(padded)

    return np.concatenate(parts)


def merge_diagrams(diagrams: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Merge multiple persistence diagrams by concatenating their points.

    Parameters
    ----------
    diagrams : list[dict]
        Each dict has ``'dgm_0'`` and ``'dgm_1'``.

    Returns
    -------
    merged : dict
        Merged diagram with all points from all input diagrams.
    """
    merged: Dict[str, np.ndarray] = {}
    for k in (0, 1):
        key = f"dgm_{k}"
        parts = [d[key] for d in diagrams if key in d and d[key].size > 0]
        if parts:
            merged[key] = np.vstack(parts)
        else:
            merged[key] = np.empty((0, 2), dtype=np.float64)
    return merged


def frechet_mean_diagram(
    diagrams: List[Dict[str, np.ndarray]],
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Approximate the Fréchet mean of persistence diagrams.

    Practical approximation: merge all (birth, death) points, cluster with
    k-means, return cluster centres as the mean diagram.

    Parameters
    ----------
    diagrams : list[dict]
    n_clusters : int or None
        Number of representative features.  If None, uses the median count
        of features across input diagrams.
    random_state : int

    Returns
    -------
    mean_diagram : dict
    """
    from sklearn.cluster import KMeans

    mean_diagram: Dict[str, np.ndarray] = {}

    for k in (0, 1):
        key = f"dgm_{k}"
        all_points = [d[key] for d in diagrams if key in d and d[key].size > 0]
        if not all_points:
            mean_diagram[key] = np.empty((0, 2), dtype=np.float64)
            continue

        merged = np.vstack(all_points)
        if n_clusters is None:
            # Use median feature count
            counts = [len(d[key]) for d in diagrams if key in d and d[key].size > 0]
            nc = max(1, int(np.median(counts)))
        else:
            nc = n_clusters

        nc = min(nc, len(merged))  # can't have more clusters than points

        if nc < 1:
            mean_diagram[key] = np.empty((0, 2), dtype=np.float64)
            continue

        km = KMeans(n_clusters=nc, random_state=random_state, n_init=10)
        km.fit(merged)
        centres = km.cluster_centers_

        # Ensure birth <= death
        centres[:, 1] = np.maximum(centres[:, 0], centres[:, 1])

        mean_diagram[key] = centres
        logger.debug("Fréchet mean H%d: %d → %d centres", k, len(merged), nc)

    return mean_diagram
