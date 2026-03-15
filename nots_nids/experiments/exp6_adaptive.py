"""
experiments/exp6_adaptive.py
===========================
Experiment 6: Adaptive Baseline under Concept Drift and Baseline Poisoning.

This experiment evaluates Layer 3 (Adaptive Baseline) by simulating:
1. Gradual concept drift (shifting benign feature means).
2. Adversarial baseline poisoning (injecting samples near the threshold).
"""

import logging
import numpy as np
import pandas as pd

from config import Config
from detector.nots_detector import NOTSDetector
from detector.adaptive_baseline import AdaptiveBaseline
from topology.ripser_filtration import compute_persistence_diagram

logger = logging.getLogger(__name__)

def run_experiment_6(detector: NOTSDetector, test_windows: list, cfg: Config):
    """Run Experiment 6: Concept Drift and Poisoning Resistance."""
    logger.info("Starting Experiment 6: Adaptive Baseline Robustness")

    def _compute_D_live(w):
        """Compute persistence diagram for a window using the detector's projector."""
        pts_low = detector.projector.transform(w["points"])
        return compute_persistence_diagram(
            pts_low, max_dim=cfg.RIPSER_MAX_DIM, max_edge=cfg.RIPSER_MAX_EDGE
        )

    # Initialize adaptive baseline
    adaptive = AdaptiveBaseline(
        D_norm_initial=detector.D_norm,
        alpha=cfg.ALPHA_EWM,
        tau=detector.tau,
    )
    
    results = []
    
    # 1. Baseline stability (stationary benign traffic)
    logger.info("  Phase 1: Stationary benign traffic")
    benign_test = [w for w in test_windows if w['label'] == 'benign'][:50]
    for i, w in enumerate(benign_test):
        W = detector.detect(w)['W']
        D_live = _compute_D_live(w)
        adaptive.update(D_live, W)
        updated = adaptive.history[-1]["updated"]
        results.append({
            'step': i,
            'phase': 'stationary',
            'W': W,
            'updated': updated,
            'tau': detector.tau
        })
        
    # 2. Concept drift (gradual shift in features)
    logger.info("  Phase 2: Simulating gradual concept drift")
    drift_windows = []
    for i in range(50):
        w = benign_test[i % len(benign_test)].copy()
        # Gradually shift all features by up to 0.1
        shift = (i / 50.0) * 0.1
        w['points'] = np.clip(w['points'] + shift, 0, 1)
        drift_windows.append(w)
        
    for i, w in enumerate(drift_windows):
        W = detector.detect(w)['W']
        D_live = _compute_D_live(w)
        adaptive.update(D_live, W)
        updated = adaptive.history[-1]["updated"]
        results.append({
            'step': i + 50,
            'phase': 'drift',
            'W': W,
            'updated': updated,
            'tau': detector.tau
        })
        
    # 3. Baseline poisoning attack
    logger.info("  Phase 3: Simulating baseline poisoning attempt")
    poison_windows = []
    for i in range(50):
        w = benign_test[i % len(benign_test)].copy()
        # Perturb to be JUST below τ/2 or NEAR τ to try and pull the baseline
        # Here we simulate an attacker injecting samples that LOOK benign but are slightly shifted
        pert = detector.tau * 0.4  # Within poisoning resistance threshold
        w['points'] = np.clip(w['points'] + pert, 0, 1)
        poison_windows.append(w)
        
    for i, w in enumerate(poison_windows):
        W = detector.detect(w)['W']
        D_live = _compute_D_live(w)
        # The adaptive baseline should REJECT these updates if they exceed tau/2
        adaptive.update(D_live, W)
        updated = adaptive.history[-1]["updated"]
        results.append({
            'step': i + 100,
            'phase': 'poisoning',
            'W': W,
            'updated': updated,
            'tau': detector.tau
        })
        
    df = pd.DataFrame(results)
    df.to_csv(f"{cfg.RESULTS_DIR}/exp6_adaptive.csv", index=False)
    logger.info("Experiment 6 results saved to exp6_adaptive.csv")
    
    return df
