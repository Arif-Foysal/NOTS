"""
experiments/exp2_whitebox.py
=============================
Experiment 2: White-box adversarial attack sweep.

Includes the critical ε_min theorem validation.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adversarial.whitebox import run_whitebox_sweep
from config import Config
from evaluation.metrics import compute_detection_rate, compute_fpr
from evaluation.theorem_validator import validate_epsilon_min_bound

logger = logging.getLogger(__name__)


def run_experiment_2(
    detector,
    test_windows: List[Dict],
    config: Config,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Experiment 2: white-box adversarial attack sweep.

    Steps
    -----
    1. For each δ in ``config.DELTA_VALUES``, run FGSM on attack windows.
    2. Run NOTS on perturbed windows, record DR, FPR, mean W.
    3. **THEOREM VALIDATION**: assert empirical DR ≥ ε_min.
    4. Save results.

    Parameters
    ----------
    detector : NOTSDetector
    test_windows : list[dict]
    config : Config
    checkpoint_dir : str or None

    Returns
    -------
    results : dict
        ``{'sweep': {delta: metrics}, 'theorem_validation': {...}}``
    """
    t_start = time.perf_counter()

    logger.info("=== Experiment 2: White-box adversarial sweep ===")
    logger.info("δ values: %s", config.DELTA_VALUES)
    logger.info("ε_min (from Nash solver): %.6f", detector.epsilon_min)

    # ── Run sweep ────────────────────────────────────────────────────────
    sweep_results = run_whitebox_sweep(
        test_windows=test_windows,
        detector=detector,
        delta_values=config.DELTA_VALUES,
        n_steps=config.FGSM_N_STEPS,
    )

    # ── Theorem validation — CRITICAL INTEGRITY CHECK ────────────────────
    logger.info("=== THEOREM VALIDATION ===")
    theorem_result = validate_epsilon_min_bound(sweep_results, detector.epsilon_min)

    # Additional per-delta assertion with CRITICAL logging
    for delta, metrics in sweep_results.items():
        empirical_dr = metrics["DR"]
        if empirical_dr < detector.epsilon_min - 0.02:
            logging.critical(
                "THEOREM BOUND VIOLATED at delta=%.4f: "
                "empirical DR=%.4f < epsilon_min=%.4f. "
                "Review proof or implementation.",
                delta,
                empirical_dr,
                detector.epsilon_min,
            )

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "exp2_results.csv")
    rows = []
    for delta, metrics in sorted(sweep_results.items()):
        rows.append({
            "delta": delta,
            "DR": metrics["DR"],
            "FPR": metrics["FPR"],
            "mean_W": float(metrics["W_values"].mean()),
            "epsilon_min": detector.epsilon_min,
            "bound_holds": metrics["DR"] >= detector.epsilon_min - 0.02,
        })
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logger.info("Results saved to %s", save_path)

    dt = time.perf_counter() - t_start
    logger.info("Experiment 2 complete in %.1f s", dt)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2 Summary — White-box Attack")
    logger.info("=" * 60)
    logger.info(f"{'δ':>8} {'DR':>8} {'FPR':>8} {'mean_W':>10} {'Bound?':>8}")
    logger.info("-" * 60)
    for row in rows:
        logger.info(
            f"{row['delta']:>8.4f} {row['DR']:>8.4f} {row['FPR']:>8.4f} "
            f"{row['mean_W']:>10.4f} {'✓' if row['bound_holds'] else '✗':>8}"
        )
    logger.info("=" * 60)
    logger.info("Theorem validation: %s", theorem_result["summary"])

    return {
        "sweep": sweep_results,
        "theorem_validation": theorem_result,
    }
