"""
preprocessing/windowing.py
==========================
Sliding-window transform: sequence of flows → list of point clouds in R^d.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


def create_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "is_attack",
    attack_type_col: str = "label_int",
    window_size: int = 500,
    step_size: Optional[int] = None,
) -> List[Dict]:
    """Convert a DataFrame of flows into a list of point-cloud windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *feature_cols*, *label_col* (binary 0/1), and optionally
        *attack_type_col* (integer-encoded multiclass label).
    feature_cols : list[str]
        Numeric feature column names.
    label_col : str
        Binary label column (0 = benign, 1 = attack).
    attack_type_col : str
        Multiclass integer label column.
    window_size : int
        Number of flows per point cloud.
    step_size : int or None
        Stride between windows.  Defaults to ``window_size // 2`` (50 % overlap).

    Returns
    -------
    windows : list[dict]
        Each dict has:
        - ``'points'``: np.ndarray of shape ``(window_size, n_features)``
        - ``'label'``: ``'attack'`` if ANY flow is an attack, else ``'benign'``
        - ``'attack_type'``: majority attack type code in the window (int)
        - ``'attack_frac'``: fraction of flows that are attacks
    """
    if step_size is None:
        step_size = window_size // 2

    n_rows = len(df)
    if n_rows < window_size:
        logger.warning(
            "DataFrame has only %d rows, less than window_size=%d. "
            "Returning empty window list.",
            n_rows,
            window_size,
        )
        return []

    # Pre-extract arrays for speed
    features = df[feature_cols].values.astype(np.float32)
    labels_binary = df[label_col].values.astype(np.int8)

    has_attack_type = attack_type_col in df.columns
    if has_attack_type:
        attack_types = df[attack_type_col].values.astype(np.int32)
    else:
        attack_types = labels_binary.astype(np.int32)

    windows: List[Dict] = []
    start = 0
    while start + window_size <= n_rows:
        end = start + window_size
        pts = features[start:end]
        lbls = labels_binary[start:end]
        atypes = attack_types[start:end]

        # Window label: attack if ANY flow is attack
        is_attack = int(lbls.sum() > 0)
        label_str = "attack" if is_attack else "benign"

        # Majority attack type (for multiclass evaluation)
        # Among attack flows only; if all benign → type 0
        if is_attack:
            attack_only = atypes[lbls == 1]
            cnt = Counter(attack_only)
            majority_type = cnt.most_common(1)[0][0]
        else:
            majority_type = 0

        attack_frac = float(lbls.sum()) / window_size

        windows.append({
            "points": pts,
            "label": label_str,
            "attack_type": int(majority_type),
            "attack_frac": attack_frac,
        })

        start += step_size

    logger.info(
        "Created %d windows (size=%d, step=%d): %d benign, %d attack",
        len(windows),
        window_size,
        step_size,
        sum(1 for w in windows if w["label"] == "benign"),
        sum(1 for w in windows if w["label"] == "attack"),
    )
    return windows
