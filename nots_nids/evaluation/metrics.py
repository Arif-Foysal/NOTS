"""
evaluation/metrics.py
=====================
Detection rate, FPR, F1, precision, recall — per attack class and overall.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DR = TP / (TP + FN) for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (1 = attack).
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    dr : float
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN).

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 = benign).
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    fpr : float
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: np.ndarray,
    label_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Compute DR, FPR, F1, Precision, Recall per unique attack type.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth (1 = attack).
    y_pred : array-like
        Binary predictions.
    attack_types : array-like
        Integer-coded attack-type labels (0 = benign, 1+ = attack classes).
    label_map : dict or None
        ``{string_label: int}`` mapping for human-readable display.

    Returns
    -------
    df : pd.DataFrame
        Rows = attack types, columns = DR, FPR, F1, Precision, Recall, Support.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    attack_types = np.asarray(attack_types, dtype=int)

    # Reverse label map for display
    inv_map: Dict[int, str] = {}
    if label_map:
        inv_map = {v: k for k, v in label_map.items()}

    unique_types = sorted(np.unique(attack_types))
    rows = []

    for atype in unique_types:
        mask = attack_types == atype
        yt = y_true[mask]
        yp = y_pred[mask]

        name = inv_map.get(atype, str(atype))

        dr = compute_detection_rate(yt, yp) if yt.sum() > 0 else np.nan
        fpr = compute_fpr(yt, yp) if (yt == 0).sum() > 0 else np.nan
        f1 = f1_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)

        rows.append({
            "attack_type": name,
            "DR": dr,
            "FPR": fpr,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "Support": int(mask.sum()),
        })

    df = pd.DataFrame(rows).set_index("attack_type")
    return df


def compute_full_metrics(
    results_list: List[Dict[str, Any]],
    windows: List[Dict],
    label_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive metrics from detector outputs and ground truth.

    Parameters
    ----------
    results_list : list[dict]
        Output from ``NOTSDetector.detect_batch``.
    windows : list[dict]
        Corresponding windows with ``'label'`` and ``'attack_type'`` keys.
    label_map : dict or None

    Returns
    -------
    metrics : dict
        ``{'overall': {...}, 'per_class': pd.DataFrame, 'W_values': np.ndarray}``
    """
    y_true = np.array([1 if w["label"] == "attack" else 0 for w in windows])
    y_pred = np.array([1 if r["alert"] else 0 for r in results_list])
    attack_types = np.array([w["attack_type"] for w in windows])
    W_values = np.array([r["W"] for r in results_list])

    overall = {
        "DR": compute_detection_rate(y_true, y_pred),
        "FPR": compute_fpr(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "n_windows": len(windows),
        "n_attacks": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "mean_W": float(W_values.mean()),
        "std_W": float(W_values.std()),
    }

    per_class = compute_per_class_metrics(y_true, y_pred, attack_types, label_map)

    logger.info("Overall: DR=%.4f, FPR=%.4f, F1=%.4f", overall["DR"], overall["FPR"], overall["F1"])
    logger.info("Per-class:\n%s", per_class.to_string())

    return {
        "overall": overall,
        "per_class": per_class,
        "W_values": W_values,
        "y_true": y_true,
        "y_pred": y_pred,
    }
