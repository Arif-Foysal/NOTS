"""
baselines/lucid_baseline.py
============================
Simplified LUCID CNN baseline for NIDS.

LUCID (Lightweight, Usable CNN in DDoS Detection) uses a 1-D CNN on flow
feature vectors.  This is a simplified reimplementation for comparison.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


class _LUCIDNet(nn.Module):
    """1-D CNN for binary NIDS classification."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Reshape: (batch, 1, input_dim) — treat features as 1-D signal
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim) → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.features(x)           # → (batch, 64, 1)
        x = x.squeeze(-1)              # → (batch, 64)
        return self.classifier(x)       # → (batch, 1)


class LUCIDBaseline:
    """Simplified LUCID CNN baseline.

    Parameters
    ----------
    epochs : int
    lr : float
    batch_size : int
    random_state : int
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 256,
        random_state: int = 42,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model: Optional[_LUCIDNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "LUCIDBaseline":
        """Train the LUCID CNN.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
            Binary labels.
        """
        torch.manual_seed(self.random_state)
        input_dim = X_train.shape[1]
        self.model = _LUCIDNet(input_dim).to(self.device)

        # Handle class imbalance
        n_pos = max(1, int(y_train.sum()))
        n_neg = max(1, len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.astype(np.float32)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info("Training LUCID: %d samples, %d features, %d epochs",
                     len(X_train), input_dim, self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(X_batch)

            if (epoch + 1) % 5 == 0:
                logger.debug("  Epoch %d/%d, loss=%.6f",
                             epoch + 1, self.epochs, total_loss / len(X_train))

        self.model.eval()
        logger.info("LUCID training complete")
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
        assert self.model is not None, "Call fit() first"
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

        return (probs > 0.5).astype(int)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return probability of attack class.

        Parameters
        ----------
        X_test : np.ndarray

        Returns
        -------
        proba : np.ndarray, shape (n_samples,)
        """
        assert self.model is not None
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate on test data.

        Returns
        -------
        results : dict
        """
        y_pred = self.predict(X_test)
        results = {
            "DR": compute_detection_rate(y_test, y_pred),
            "FPR": compute_fpr(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }
        logger.info(
            "LUCID: DR=%.4f, FPR=%.4f, F1=%.4f",
            results["DR"],
            results["FPR"],
            results["F1"],
        )
        return results
