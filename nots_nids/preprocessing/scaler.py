"""
preprocessing/scaler.py
=======================
MinMaxScaler fit on train, applied to val/test.  Scaler is persisted to disk.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def fit_scaler(X_train: np.ndarray, save_path: str = "results/scaler.joblib") -> Tuple[np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on training data and persist to disk.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)
    save_path : str
        Where to save the fitted scaler.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled training data in [0, 1].
    scaler : MinMaxScaler
        The fitted scaler (also saved to *save_path*).
    """
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    X_scaled = scaler.fit_transform(X_train)

    # Persist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    logger.info("Scaler fit on %s, saved to %s", X_train.shape, save_path)
    return X_scaled, scaler


def apply_scaler(
    X: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Apply a previously fitted scaler to new data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    scaler : MinMaxScaler
        Must be already fitted (on training data).

    Returns
    -------
    X_scaled : np.ndarray
        Scaled data in [0, 1] (clipped — values are guaranteed to stay within
        [0, 1] even if test data exceeds the training range).
    """
    X_scaled = scaler.transform(X)
    logger.info("Scaler applied to %s", X.shape)
    return X_scaled


def load_scaler(path: str = "results/scaler.joblib") -> MinMaxScaler:
    """Load a previously saved scaler from disk.

    Parameters
    ----------
    path : str
        File path to the joblib-serialised scaler.

    Returns
    -------
    scaler : MinMaxScaler
    """
    scaler = joblib.load(path)
    logger.info("Scaler loaded from %s", path)
    return scaler
