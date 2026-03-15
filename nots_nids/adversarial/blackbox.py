"""
adversarial/blackbox.py
========================
Black-box transfer attack using a surrogate Random Forest.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


def train_surrogate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest surrogate on the same training data.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    random_state : int

    Returns
    -------
    surrogate : RandomForestClassifier
    """
    logger.info("Training surrogate RF on %s", X_train.shape)
    surrogate = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    surrogate.fit(X_train, y_train)
    logger.info("Surrogate trained. Feature importances: top-5 = %s",
                np.argsort(surrogate.feature_importances_)[-5:][::-1])
    return surrogate


def craft_blackbox_adversarial(
    X_attack: np.ndarray,
    surrogate_model: RandomForestClassifier,
    delta_max: float = 0.10,
    perturbable_features: Optional[List[int]] = None,
) -> np.ndarray:
    """Craft adversarial perturbations using surrogate feature importances.

    Perturbs features with highest importance in the direction that reduces
    the surrogate's attack probability.

    Parameters
    ----------
    X_attack : np.ndarray, shape (n_samples, n_features)
        Original attack samples.
    surrogate_model : RandomForestClassifier
    delta_max : float
        L-∞ perturbation budget.
    perturbable_features : list[int] or None
        Feature indices the attacker can perturb.  None = all features.

    Returns
    -------
    X_perturbed : np.ndarray
        Perturbed attack samples.
    """
    importances = surrogate_model.feature_importances_
    n_features = X_attack.shape[1]
    allowed = perturbable_features if perturbable_features is not None else list(range(n_features))

    # Perturb top-k most important features (restricted to allowed set)
    allowed_importances = [(idx, importances[idx]) for idx in allowed]
    allowed_importances.sort(key=lambda x: x[1], reverse=True)
    k = max(1, len(allowed) // 4)  # Top 25% of allowed features
    top_features = [idx for idx, _ in allowed_importances[:k]]

    X_perturbed = X_attack.copy()

    for feat_idx in top_features:
        # Direction: for each sample, move feature toward the mean of
        # benign predictions (heuristic: subtract delta if above median,
        # add if below)
        median_val = np.median(X_attack[:, feat_idx])
        direction = np.where(X_attack[:, feat_idx] > median_val, -1.0, 1.0)
        perturbation = delta_max * direction * importances[feat_idx]
        X_perturbed[:, feat_idx] += perturbation

    # Clip to [0, 1]
    X_perturbed = np.clip(X_perturbed, 0.0, 1.0)

    # Verify L-∞ constraint
    actual_delta = np.abs(X_perturbed - X_attack).max()
    logger.info("Black-box perturbation: max L-∞ = %.6f (budget = %.4f)",
                actual_delta, delta_max)

    return X_perturbed


def run_blackbox_experiment(
    test_windows: List[Dict],
    detector,
    surrogate_model: RandomForestClassifier,
    delta_max: float = 0.10,
    max_windows: int = 200,
    perturbable_features: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run the black-box transfer attack experiment.

    Parameters
    ----------
    test_windows : list[dict]
    detector : NOTSDetector
    surrogate_model : RandomForestClassifier
    delta_max : float
    max_windows : int
    perturbable_features : list[int] or None
        Feature indices the attacker can perturb.  None = all features.

    Returns
    -------
    results : dict
        ``{'DR': float, 'FPR': float, 'n_attack': int, 'n_benign': int}``
    """
    attack_windows = [w for w in test_windows if w["label"] == "attack"][:max_windows]
    benign_windows = [w for w in test_windows if w["label"] == "benign"][:max_windows]

    logger.info("Black-box experiment: %d attack, %d benign windows",
                len(attack_windows), len(benign_windows))

    # Perturb attack windows
    perturbed_windows = []
    for w in attack_windows:
        X_perturbed = craft_blackbox_adversarial(
            w["points"], surrogate_model, delta_max=delta_max,
            perturbable_features=perturbable_features,
        )
        pw = {
            "points": X_perturbed,
            "label": "attack",
            "attack_type": w.get("attack_type", 1),
            "attack_frac": w.get("attack_frac", 1.0),
        }
        perturbed_windows.append(pw)

    # Detect
    all_windows = perturbed_windows + benign_windows
    all_labels = [1] * len(perturbed_windows) + [0] * len(benign_windows)

    det_results = detector.detect_batch(all_windows)
    y_pred = np.array([1 if r["alert"] else 0 for r in det_results])
    y_true = np.array(all_labels)

    results = {
        "DR": compute_detection_rate(y_true, y_pred),
        "FPR": compute_fpr(y_true, y_pred),
        "n_attack": len(perturbed_windows),
        "n_benign": len(benign_windows),
    }

    logger.info(
        "Black-box: DR=%.4f, FPR=%.4f",
        results["DR"],
        results["FPR"],
    )
    return results
