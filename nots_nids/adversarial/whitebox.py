"""
adversarial/whitebox.py
========================
White-box FGSM / PGD-style L-∞ adversarial attack.

Since the Wasserstein distance is not easily differentiable w.r.t. flow
features, we use numerical gradient estimation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


def compute_numerical_gradient(
    window: Dict,
    detector,
    feature_mask: List[int],
    h: float = 1e-4,
) -> np.ndarray:
    """Numerically estimate gradient of W w.r.t. mean flow features.

    For each feature i in feature_mask::

        grad[i] ≈ (W(x + h·eᵢ) − W(x − h·eᵢ)) / (2h)

    We perturb the *mean* flow in the window (representative point).

    Parameters
    ----------
    window : dict
        Window with ``'points'`` key.
    detector : NOTSDetector
        Fitted NOTS detector.
    feature_mask : list[int]
        Feature indices to compute gradient for.
    h : float
        Finite-difference step size.

    Returns
    -------
    grad : np.ndarray, shape (n_features,)
        Gradient vector (zero for features not in mask).
    """
    points = window["points"].copy()
    n_features = points.shape[1]
    grad = np.zeros(n_features, dtype=np.float64)

    # Baseline W
    base_result = detector.detect({"points": points, "label": window.get("label", "attack")})
    W_base = base_result["W"]

    for i in feature_mask:
        # Forward perturbation
        pts_plus = points.copy()
        pts_plus[:, i] += h
        w_plus = detector.detect({"points": pts_plus, "label": window.get("label", "attack")})

        # Backward perturbation
        pts_minus = points.copy()
        pts_minus[:, i] -= h
        w_minus = detector.detect({"points": pts_minus, "label": window.get("label", "attack")})

        grad[i] = (w_plus["W"] - w_minus["W"]) / (2 * h)

    return grad


def fgsm_attack(
    window: Dict,
    delta_max: float,
    detector,
    n_steps: int = 10,
    step_size: Optional[float] = None,
    h: float = 1e-4,
    perturbable_features: Optional[List[int]] = None,
) -> Tuple[Dict, List[float]]:
    """Iterative FGSM (PGD-style) L-∞ attack on a window.

    At each step::

        x ← clip(x + step_size · sign(grad), x_orig − δ_max, x_orig + δ_max)

    Also clips to [0, 1] since features are MinMax-scaled.

    Parameters
    ----------
    window : dict
        Original window.
    delta_max : float
        Maximum L-∞ perturbation budget.
    detector : NOTSDetector
        Fitted detector (the target to evade).
    n_steps : int
        Number of PGD steps.
    step_size : float or None
        Per-step size.  Defaults to ``delta_max / n_steps``.
    h : float
        Numerical gradient step.
    perturbable_features : list[int] or None
        Feature indices the attacker can perturb.  None = all features.

    Returns
    -------
    perturbed_window : dict
        Window with perturbed points.
    W_trajectory : list[float]
        Wasserstein distance at each step (for convergence analysis).
    """
    if step_size is None:
        step_size = delta_max / max(1, n_steps)

    points_orig = window["points"].copy().astype(np.float64)
    points = points_orig.copy()
    n_features = points.shape[1]
    feature_mask = perturbable_features if perturbable_features is not None else list(range(n_features))

    W_trajectory: List[float] = []

    for step in range(n_steps):
        current_window = {
            "points": points.copy(),
            "label": window.get("label", "attack"),
            "attack_type": window.get("attack_type", 1),
        }

        # Compute gradient (direction that INCREASES W — attacker wants to DECREASE)
        grad = compute_numerical_gradient(current_window, detector, feature_mask, h=h)

        # FGSM step: move in direction that DECREASES W (negative sign)
        perturbation = -step_size * np.sign(grad)

        # Apply to all flows in window
        points += perturbation[np.newaxis, :]

        # Clip to L-∞ ball around original
        points = np.clip(points, points_orig - delta_max, points_orig + delta_max)

        # Clip to [0, 1] (MinMax-scaled feature range)
        points = np.clip(points, 0.0, 1.0)

        # Record W
        result = detector.detect({
            "points": points.copy(),
            "label": window.get("label", "attack"),
            "attack_type": window.get("attack_type", 1),
        })
        W_trajectory.append(result["W"])

    perturbed_window = {
        "points": points,
        "label": window.get("label", "attack"),
        "attack_type": window.get("attack_type", 1),
        "attack_frac": window.get("attack_frac", 1.0),
    }

    return perturbed_window, W_trajectory


def run_whitebox_sweep(
    test_windows: List[Dict],
    detector,
    delta_values: List[float],
    n_steps: int = 10,
    max_windows: int = 200,
    perturbable_features: Optional[List[int]] = None,
) -> Dict[float, Dict[str, Any]]:
    """Run white-box adversarial sweep across δ values.

    Parameters
    ----------
    test_windows : list[dict]
    detector : NOTSDetector
    delta_values : list[float]
    n_steps : int
    max_windows : int
        Maximum attack windows to process per δ.
    perturbable_features : list[int] or None
        Feature indices the attacker can perturb.  None = all features.

    Returns
    -------
    results : dict
        ``{delta: {'DR': float, 'FPR': float, 'W_values': np.ndarray}}``
    """
    attack_windows = [w for w in test_windows if w["label"] == "attack"]
    benign_windows = [w for w in test_windows if w["label"] == "benign"]

    # Limit for computational efficiency
    attack_windows = attack_windows[:max_windows]
    benign_windows = benign_windows[:max_windows]

    results: Dict[float, Dict[str, Any]] = {}

    for delta in delta_values:
        logger.info("White-box sweep: δ = %.4f", delta)

        # Perturb attack windows
        perturbed = []
        for w in attack_windows:
            pw, _ = fgsm_attack(w, delta, detector, n_steps=n_steps,
                                perturbable_features=perturbable_features)
            perturbed.append(pw)

        # Run detection on perturbed attack + unperturbed benign
        all_windows = perturbed + benign_windows
        all_labels = [1] * len(perturbed) + [0] * len(benign_windows)

        det_results = detector.detect_batch(all_windows)
        y_pred = np.array([1 if r["alert"] else 0 for r in det_results])
        y_true = np.array(all_labels)
        W_values = np.array([r["W"] for r in det_results])

        results[delta] = {
            "DR": compute_detection_rate(y_true, y_pred),
            "FPR": compute_fpr(y_true, y_pred),
            "W_values": W_values,
            "n_attack": len(perturbed),
            "n_benign": len(benign_windows),
        }

        logger.info(
            "  δ=%.4f: DR=%.4f, FPR=%.4f, mean_W=%.4f",
            delta,
            results[delta]["DR"],
            results[delta]["FPR"],
            W_values.mean(),
        )

    return results
