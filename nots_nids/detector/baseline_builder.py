"""
detector/baseline_builder.py
=============================
Build the reference persistence diagram D_norm from benign training windows.

Uses the unified Projector interface (PCA, Random Projection, or UMAP).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from topology.ripser_filtration import compute_persistence_diagram
from topology.persistence import frechet_mean_diagram

logger = logging.getLogger(__name__)


def build_baseline(
    benign_windows: List[Dict],
    projector,
    n_baseline_windows: int = 50,
    max_dim: int = 1,
    max_edge: float = 2.0,
    save_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Build the baseline persistence diagram D_norm from benign traffic.

    Parameters
    ----------
    benign_windows : list[dict]
        Windows with ``'points'`` key (point-cloud arrays).  Only benign windows
        should be passed here.
    projector : Projector or UMAPReducer
        A *fitted* projector (any object with a ``transform()`` method).
    n_baseline_windows : int
        Number of benign windows to use (taken from the start).
    max_dim : int
        Maximum persistence dimension.
    max_edge : float
        Maximum filtration radius.
    save_dir : str or None
        If given, save D_norm arrays here as ``.npy`` files.

    Returns
    -------
    D_norm : dict
        ``{'dgm_0': np.ndarray, 'dgm_1': np.ndarray}``
    """
    n_use = min(n_baseline_windows, len(benign_windows))
    logger.info("Building baseline from %d benign windows", n_use)

    diagrams: List[Dict[str, np.ndarray]] = []
    for i in range(n_use):
        pts_raw = benign_windows[i]["points"]
        pts_low = projector.transform(pts_raw)
        dgm = compute_persistence_diagram(pts_low, max_dim=max_dim, max_edge=max_edge)
        diagrams.append(dgm)
        if (i + 1) % 10 == 0:
            logger.info("  Processed %d / %d baseline windows", i + 1, n_use)

    # Compute Fréchet mean
    D_norm = frechet_mean_diagram(diagrams)

    # Log stats
    for k in (0, 1):
        key = f"dgm_{k}"
        n_pts = len(D_norm.get(key, []))
        logger.info("D_norm H%d: %d representative features", k, n_pts)

    # Optionally save
    if save_dir is not None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        for k in (0, 1):
            key = f"dgm_{k}"
            np.save(str(p / f"D_norm_{key}.npy"), D_norm.get(key, np.empty((0, 2))))
        logger.info("D_norm saved to %s", save_dir)

    return D_norm


def load_baseline(save_dir: str) -> Dict[str, np.ndarray]:
    """Load a previously saved baseline from disk.

    Parameters
    ----------
    save_dir : str
        Directory containing ``D_norm_dgm_0.npy`` and ``D_norm_dgm_1.npy``.

    Returns
    -------
    D_norm : dict
    """
    p = Path(save_dir)
    D_norm: Dict[str, np.ndarray] = {}
    for k in (0, 1):
        key = f"dgm_{k}"
        path = p / f"D_norm_{key}.npy"
        if path.exists():
            D_norm[key] = np.load(str(path))
        else:
            D_norm[key] = np.empty((0, 2), dtype=np.float64)
    logger.info("Loaded baseline from %s", save_dir)
    return D_norm
