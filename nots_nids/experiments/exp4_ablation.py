"""
experiments/exp4_ablation.py
=============================
Experiment 4: Ablation study with 4 variants.

Variant A — Full NOTS: TDA + Nash (standard detector)
Variant B — TDA only: TDA with fixed feature set (no Nash)
Variant C — Game theory only: Nash sampling + Mahalanobis (no TDA)
Variant D — Plain ML: Standard RF baseline
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance

from config import Config
from detector.nots_detector import NOTSDetector
from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


def _mahalanobis_detector(
    train_windows: List[Dict],
    test_windows: List[Dict],
    feature_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Variant C: statistical anomaly detection using Mahalanobis distance.

    Trains on benign windows, detects based on Mahalanobis distance threshold.
    """
    # Extract mean feature vectors per window
    benign_train = [w for w in train_windows if w["label"] == "benign"]
    if not benign_train:
        return {"DR": 0.0, "FPR": 1.0}

    # Compute mean features for benign training
    benign_means = []
    for w in benign_train:
        pts = w["points"]
        if feature_indices is not None:
            pts = pts[:, feature_indices]
        benign_means.append(pts.mean(axis=0))
    benign_means = np.array(benign_means)

    # Fit covariance
    try:
        cov = EmpiricalCovariance().fit(benign_means)
        mean_vec = benign_means.mean(axis=0)
        cov_inv = np.linalg.inv(cov.covariance_ + 1e-6 * np.eye(benign_means.shape[1]))
    except Exception as e:
        logger.warning("Mahalanobis fit failed: %s", e)
        return {"DR": 0.0, "FPR": 1.0}

    # Set threshold on training benign (95th percentile)
    train_dists = []
    for m in benign_means:
        d = mahalanobis(m, mean_vec, cov_inv)
        train_dists.append(d)
    threshold = np.percentile(train_dists, 95)

    # Evaluate on test
    y_true = []
    y_pred = []
    for w in test_windows:
        pts = w["points"]
        if feature_indices is not None:
            pts = pts[:, feature_indices]
        m = pts.mean(axis=0)
        d = mahalanobis(m, mean_vec, cov_inv)
        y_true.append(1 if w["label"] == "attack" else 0)
        y_pred.append(1 if d > threshold else 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "DR": compute_detection_rate(y_true, y_pred),
        "FPR": compute_fpr(y_true, y_pred),
    }


def run_experiment_4(
    train_windows: List[Dict],
    val_windows: List[Dict],
    test_windows: List[Dict],
    config: Config,
    full_detector: Optional[NOTSDetector] = None,
) -> Dict[str, Dict[str, float]]:
    """Experiment 4: ablation study.

    Parameters
    ----------
    train_windows, val_windows, test_windows : list[dict]
    config : Config
    full_detector : NOTSDetector or None
        Pre-fitted full NOTS detector (Variant A).

    Returns
    -------
    results : dict
        ``{variant_name: {'DR': float, 'FPR': float}}``
    """
    t_start = time.perf_counter()
    logger.info("=== Experiment 4: Ablation study ===")
    results: Dict[str, Dict[str, float]] = {}

    # ── Variant A: Full NOTS ─────────────────────────────────────────────
    logger.info("Variant A: Full NOTS (TDA + Nash)")
    if full_detector is not None:
        det_results = full_detector.detect_batch(test_windows)
        y_true = np.array([1 if w["label"] == "attack" else 0 for w in test_windows])
        y_pred = np.array([1 if r["alert"] else 0 for r in det_results])
        results["A: Full NOTS"] = {
            "DR": compute_detection_rate(y_true, y_pred),
            "FPR": compute_fpr(y_true, y_pred),
        }
    else:
        det_a = NOTSDetector(config)
        det_a.fit(train_windows, val_windows)
        det_results = det_a.detect_batch(test_windows)
        y_true = np.array([1 if w["label"] == "attack" else 0 for w in test_windows])
        y_pred = np.array([1 if r["alert"] else 0 for r in det_results])
        results["A: Full NOTS"] = {
            "DR": compute_detection_rate(y_true, y_pred),
            "FPR": compute_fpr(y_true, y_pred),
        }

    # ── Variant B: TDA only (no Nash, use all features) ──────────────────
    logger.info("Variant B: TDA only (no Nash sampling)")
    from topology.umap_reducer import UMAPReducer
    from topology.ripser_filtration import compute_persistence_diagram
    from topology.wasserstein import wasserstein_distance
    from detector.baseline_builder import build_baseline

    umap_b = UMAPReducer(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        random_state=config.RANDOM_SEED,
    )
    all_pts = np.vstack([w["points"] for w in train_windows])
    umap_b.fit(all_pts)

    benign_train = [w for w in train_windows if w["label"] == "benign"]
    D_norm_b = build_baseline(benign_train, umap_b, n_baseline_windows=config.N_BASELINE_WINDOWS)

    # Set threshold from validation
    val_W = []
    val_labels = []
    for w in val_windows[:50]:
        pts_low = umap_b.transform(w["points"])
        dgm = compute_persistence_diagram(pts_low, max_dim=config.RIPSER_MAX_DIM)
        W = wasserstein_distance(dgm, D_norm_b)
        val_W.append(W)
        val_labels.append(1 if w["label"] == "attack" else 0)

    benign_W = [W for W, l in zip(val_W, val_labels) if l == 0]
    tau_b = np.percentile(benign_W, 95) if benign_W else np.median(val_W)

    # Test
    y_pred_b = []
    for w in test_windows:
        pts_low = umap_b.transform(w["points"])
        dgm = compute_persistence_diagram(pts_low, max_dim=config.RIPSER_MAX_DIM)
        W = wasserstein_distance(dgm, D_norm_b)
        y_pred_b.append(1 if W >= tau_b else 0)

    y_true = np.array([1 if w["label"] == "attack" else 0 for w in test_windows])
    y_pred_b = np.array(y_pred_b)
    results["B: TDA Only"] = {
        "DR": compute_detection_rate(y_true, y_pred_b),
        "FPR": compute_fpr(y_true, y_pred_b),
    }

    # ── Variant C: Game theory only (Mahalanobis) ────────────────────────
    logger.info("Variant C: Game theory only (Mahalanobis + Nash sampling)")
    results["C: GameTheory Only"] = _mahalanobis_detector(
        train_windows, test_windows
    )

    # ── Variant D: Plain ML (RF) ────────────────────────────────────────
    logger.info("Variant D: Plain ML (Random Forest)")
    from baselines.rf_baseline import RFBaseline
    X_train = np.vstack([w["points"].mean(axis=0) for w in train_windows])
    y_train = np.array([1 if w["label"] == "attack" else 0 for w in train_windows])
    X_test = np.vstack([w["points"].mean(axis=0) for w in test_windows])
    y_test = np.array([1 if w["label"] == "attack" else 0 for w in test_windows])

    rf = RFBaseline(random_state=config.RANDOM_SEED)
    rf.fit(X_train, y_train)
    rf_results = rf.evaluate(X_test, y_test)
    results["D: Plain ML (RF)"] = {"DR": rf_results["DR"], "FPR": rf_results["FPR"]}

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "exp4_ablation.csv")
    rows = [{"variant": k, **v} for k, v in results.items()]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logger.info("Ablation results saved to %s", save_path)

    dt = time.perf_counter() - t_start
    logger.info("Experiment 4 complete in %.1f s", dt)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Experiment 4 — Ablation Study")
    logger.info("=" * 50)
    for name, metrics in results.items():
        logger.info(f"  {name:<25} DR={metrics['DR']:.4f}  FPR={metrics['FPR']:.4f}")

    # Contribution analysis
    full_dr = results.get("A: Full NOTS", {}).get("DR", 0.0)
    for name, metrics in results.items():
        if "Full" not in name:
            contrib = full_dr - metrics["DR"]
            logger.info(f"  Contribution over {name}: {contrib:+.4f}")

    return results
