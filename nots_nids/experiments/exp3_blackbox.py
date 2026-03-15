"""
experiments/exp3_blackbox.py
=============================
Experiment 3: Black-box transfer attack.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adversarial.blackbox import (
    craft_blackbox_adversarial,
    run_blackbox_experiment,
    train_surrogate_model,
)
from config import Config

logger = logging.getLogger(__name__)


def run_experiment_3(
    detector,
    train_windows: List[Dict],
    test_windows: List[Dict],
    config: Config,
) -> Dict[str, Any]:
    """Experiment 3: black-box transfer attack.

    Steps
    -----
    1. Extract flow-level features from training windows.
    2. Train surrogate RF model.
    3. Craft adversarial flows using surrogate at δ_max = 0.10.
    4. Package into windows, run NOTS detection.
    5. Compare DR to Exp 1 baseline.

    Parameters
    ----------
    detector : NOTSDetector
    train_windows : list[dict]
    test_windows : list[dict]
    config : Config

    Returns
    -------
    results : dict
    """
    t_start = time.perf_counter()
    logger.info("=== Experiment 3: Black-box transfer attack ===")

    # ── Step 1: Extract flow-level features ─────────────────────────────
    logger.info("Step 1: Extracting flow-level features from training windows")
    X_train_flows = np.vstack([w["points"] for w in train_windows])
    y_train_flows = np.concatenate([
        np.full(len(w["points"]), 1 if w["label"] == "attack" else 0)
        for w in train_windows
    ])

    # ── Step 2: Train surrogate ─────────────────────────────────────────
    logger.info("Step 2: Training surrogate RF model")
    surrogate = train_surrogate_model(
        X_train_flows, y_train_flows, random_state=config.RANDOM_SEED
    )

    # ── Steps 3–4: Craft and detect ─────────────────────────────────────
    logger.info("Step 3-4: Crafting adversarial flows and running detection")
    delta_max = max(config.DELTA_VALUES)  # Use maximum δ for black-box
    results = run_blackbox_experiment(
        test_windows=test_windows,
        detector=detector,
        surrogate_model=surrogate,
        delta_max=delta_max,
    )

    # ── Step 5: Save ────────────────────────────────────────────────────
    save_path = os.path.join(config.RESULTS_DIR, "exp3_results.csv")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pd.DataFrame([results]).to_csv(save_path, index=False)
    logger.info("Results saved to %s", save_path)

    dt = time.perf_counter() - t_start
    logger.info("Experiment 3 complete in %.1f s", dt)
    logger.info(
        "Black-box results: DR=%.4f, FPR=%.4f",
        results["DR"],
        results["FPR"],
    )

    return results
