"""
experiments/exp1_baseline.py
=============================
Experiment 1: Baseline detection with no adversarial perturbation.

Runs NOTS and all baseline detectors on the same test data, computes metrics
per attack class and overall.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from evaluation.metrics import (
    compute_detection_rate,
    compute_fpr,
    compute_full_metrics,
    compute_per_class_metrics,
)

logger = logging.getLogger(__name__)


def run_experiment_1(
    detector,
    test_windows: List[Dict],
    baselines: Dict[str, Any],
    config: Config,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    label_map: Optional[Dict[str, int]] = None,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Experiment 1: baseline detection performance.

    Parameters
    ----------
    detector : NOTSDetector
    test_windows : list[dict]
    baselines : dict
        ``{name: baseline_object}`` — each must have ``evaluate(X, y)``.
    config : Config
    X_test : np.ndarray or None
        Flow-level test features (for baselines that work on flows).
    y_test : np.ndarray or None
        Flow-level test labels.
    label_map : dict or None
    checkpoint_dir : str or None

    Returns
    -------
    results : dict
        ``{'NOTS': {...}, 'RF': {...}, ...}``
    """
    t_start = time.perf_counter()
    results: Dict[str, Any] = {}

    # ── NOTS ─────────────────────────────────────────────────────────────
    logger.info("=== Experiment 1: Running NOTS detector ===")
    nots_results = detector.detect_batch(test_windows)
    nots_metrics = compute_full_metrics(nots_results, test_windows, label_map)
    results["NOTS"] = nots_metrics

    # Checkpoint
    if checkpoint_dir:
        _save_checkpoint(results, checkpoint_dir, "exp1")

    # ── Baselines ────────────────────────────────────────────────────────
    if X_test is not None and y_test is not None:
        for name, baseline in baselines.items():
            logger.info("=== Experiment 1: Running %s baseline ===", name)
            try:
                bl_metrics = baseline.evaluate(X_test, y_test)
                results[name] = bl_metrics
            except Exception as e:
                logger.error("Baseline %s failed: %s", name, e)
                results[name] = {"DR": 0.0, "FPR": 1.0, "F1": 0.0, "error": str(e)}

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "exp1_results.csv")
    _results_to_csv(results, save_path)

    dt = time.perf_counter() - t_start
    logger.info("Experiment 1 complete in %.1f s", dt)

    # Print summary
    _print_summary(results)

    return results


def _save_checkpoint(results: Dict, dirpath: str, prefix: str) -> None:
    """Save intermediate results checkpoint."""
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"{prefix}_checkpoint.csv")
    _results_to_csv(results, path)
    logger.info("Checkpoint saved: %s", path)


def _results_to_csv(results: Dict, path: str) -> None:
    """Save results dict as flat CSV."""
    rows = []
    for method, data in results.items():
        if isinstance(data, dict):
            overall = data.get("overall", data)
            row = {"method": method}
            for k, v in overall.items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = v
            rows.append(row)
    if rows:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Results saved to %s", path)


def _print_summary(results: Dict) -> None:
    """Print a summary table to console."""
    header = f"{'Method':<15} {'DR':>8} {'FPR':>8} {'F1':>8}"
    logger.info("\n" + "=" * 42)
    logger.info("Experiment 1 Summary")
    logger.info("=" * 42)
    logger.info(header)
    logger.info("-" * 42)
    for method, data in results.items():
        overall = data.get("overall", data) if isinstance(data, dict) else data
        dr = overall.get("DR", 0.0)
        fpr = overall.get("FPR", 0.0)
        f1 = overall.get("F1", 0.0)
        logger.info(f"{method:<15} {dr:>8.4f} {fpr:>8.4f} {f1:>8.4f}")
    logger.info("=" * 42)
