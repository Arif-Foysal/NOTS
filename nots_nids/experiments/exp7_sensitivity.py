"""
experiments/exp7_sensitivity.py
==============================
Experiment 7: Hyperparameter Sensitivity Analysis.

Evaluates how detection rate (DR) and false positive rate (FPR) change 
with respect to:
- Window Size
- Projection Components
- Ripser Max Edge
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict

from config import Config
from detector.nots_detector import NOTSDetector
from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)

def run_experiment_7(train_windows: list, val_windows: list, test_windows: list, cfg: Config):
    """Run Experiment 7: Sensitivity Analysis."""
    logger.info("Starting Experiment 7: Hyperparameter Sensitivity Analysis")
    
    results = []
    
    # 1. Sensitivity to Window Size
    # Note: Changing window size requires re-windowing the original DataFrames.
    # For this experiment, we assume the user might run it separately or we mock 
    # it if windowing is fast enough. Here we vary the parameters we can 
    # change without re-splitting.
    
    # 2. Sensitivity to Projection Components (k)
    logger.info("  Phase 1: Sensitivity to Projection Components (k)")
    for k in cfg.SENSITIVITY_N_COMPONENTS:
        logger.info("    Testing k=%d", k)
        # Temporarily override cfg
        orig_k = cfg.UMAP_N_COMPONENTS
        cfg.UMAP_N_COMPONENTS = k
        
        detector = NOTSDetector(cfg)
        detector.fit(train_windows, val_windows)
        
        preds = detector.detect_batch(test_windows)
        y_true = [1 if w['label'] == 'attack' else 0 for w in test_windows]
        y_pred = [1 if r['alert'] else 0 for r in preds]
        
        metrics = compute_all_metrics(y_true, y_pred)
        results.append({
            'parameter': 'n_components',
            'value': k,
            'dr': metrics['overall']['dr'],
            'fpr': metrics['overall']['fpr'],
            'f1': metrics['overall']['f1']
        })
        
        # Restore cfg
        cfg.UMAP_N_COMPONENTS = orig_k
        
    # 3. Sensitivity to Ripser Max Edge (filtration radius)
    logger.info("  Phase 2: Sensitivity to Max Edge radius")
    for delta in cfg.SENSITIVITY_MAX_EDGES:
        logger.info("    Testing max_edge=%.1f", delta)
        orig_edge = cfg.RIPSER_MAX_EDGE
        cfg.RIPSER_MAX_EDGE = delta
        
        # We don't necessarily need to refit EVERYTHING if only Ripser changed,
        # but for clean results we do a full fit.
        detector = NOTSDetector(cfg)
        detector.fit(train_windows, val_windows)
        
        preds = detector.detect_batch(test_windows)
        y_true = [1 if w['label'] == 'attack' else 0 for w in test_windows]
        y_pred = [1 if r['alert'] else 0 for r in preds]
        
        metrics = compute_all_metrics(y_true, y_pred)
        results.append({
            'parameter': 'max_edge',
            'value': delta,
            'dr': metrics['overall']['dr'],
            'fpr': metrics['overall']['fpr'],
            'f1': metrics['overall']['f1']
        })
        
        cfg.RIPSER_MAX_EDGE = orig_edge
        
    df = pd.DataFrame(results)
    df.to_csv(f"{cfg.RESULTS_DIR}/exp7_sensitivity.csv", index=False)
    logger.info("✅ Experiment 7 results saved to exp7_sensitivity.csv")
    
    return df
