"""
baselines/rf_baseline.py
=========================
Random Forest NIDS baseline — primary comparison method.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


class RFBaseline:
    """Random Forest baseline classifier for flow-level intrusion detection.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default 100).
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RFBaseline":
        """Train the Random Forest.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
            Binary labels (0 = benign, 1 = attack).
        """
        logger.info("Training RF baseline on %s", X_train.shape)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("RF training complete")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary predictions.

        Parameters
        ----------
        X_test : np.ndarray

        Returns
        -------
        y_pred : np.ndarray
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Parameters
        ----------
        X_test : np.ndarray

        Returns
        -------
        proba : np.ndarray, shape (n_samples, 2)
        """
        return self.model.predict_proba(X_test)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate on test data.

        Returns
        -------
        results : dict
            ``{'DR': float, 'FPR': float, 'F1': float,
               'feature_importances': np.ndarray}``
        """
        y_pred = self.predict(X_test)

        results = {
            "DR": compute_detection_rate(y_test, y_pred),
            "FPR": compute_fpr(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "feature_importances": self.model.feature_importances_,
        }

        logger.info(
            "RF Baseline: DR=%.4f, FPR=%.4f, F1=%.4f",
            results["DR"],
            results["FPR"],
            results["F1"],
        )
        return results
