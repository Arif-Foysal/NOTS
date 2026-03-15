"""
baselines/kitsune_baseline.py
==============================
Simplified Kitsune-style anomaly detector using an ensemble of autoencoders.

Architecture per autoencoder: input → 32 → 16 → 8 → 16 → 32 → input.
Ensemble of k=10 autoencoders trained on k-means clusters of benign traffic.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_detection_rate, compute_fpr

logger = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    """Simple symmetric autoencoder: input → 32 → 16 → 8 → 16 → 32 → input."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class KitsuneBaseline:
    """Simplified Kitsune anomaly detector.

    Uses k-means to find clusters in benign traffic, trains one autoencoder
    per cluster.  At inference, reconstruction error = anomaly score.

    Parameters
    ----------
    n_clusters : int
        Number of autoencoder sub-models (default 10).
    epochs : int
        Training epochs per autoencoder.
    lr : float
        Learning rate.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        epochs: int = 30,
        lr: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.autoencoders: List[_Autoencoder] = []
        self.kmeans: Optional[KMeans] = None
        self.threshold: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "KitsuneBaseline":
        """Train the Kitsune ensemble on benign training data.

        Parameters
        ----------
        X_train : np.ndarray
            Benign training samples only.
        X_val : np.ndarray or None
            Validation data (mixed) for threshold setting.
        y_val : np.ndarray or None
            Validation labels.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X_train.shape[1]
        logger.info("Training Kitsune: %d clusters, %d features", self.n_clusters, input_dim)

        # Cluster benign traffic
        nc = min(self.n_clusters, len(X_train))
        self.kmeans = KMeans(n_clusters=nc, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_train)

        # Train one autoencoder per cluster
        self.autoencoders = []
        for c in range(nc):
            mask = cluster_labels == c
            X_c = X_train[mask]
            if len(X_c) < 5:
                logger.warning("Cluster %d has %d samples — skipping", c, len(X_c))
                continue

            ae = _Autoencoder(input_dim).to(self.device)
            optimiser = optim.Adam(ae.parameters(), lr=self.lr)
            criterion = nn.MSELoss()

            dataset = TensorDataset(torch.FloatTensor(X_c))
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            ae.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    out = ae(batch)
                    loss = criterion(out, batch)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total_loss += loss.item() * len(batch)

            ae.eval()
            self.autoencoders.append(ae)
            logger.debug("  Cluster %d: %d samples, final loss=%.6f",
                         c, len(X_c), total_loss / len(X_c))

        logger.info("Trained %d autoencoders", len(self.autoencoders))

        # Set threshold on validation set
        if X_val is not None and y_val is not None:
            scores = self.score(X_val)
            benign_scores = scores[y_val == 0]
            if len(benign_scores) > 0:
                self.threshold = float(np.percentile(benign_scores, 95))
            else:
                self.threshold = float(np.percentile(scores, 95))
            logger.info("Threshold set to %.6f (95th percentile of val benign)", self.threshold)
        else:
            scores = self.score(X_train)
            self.threshold = float(np.percentile(scores, 95))

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (max reconstruction error across ensemble).

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
        """
        if not self.autoencoders:
            return np.zeros(len(X))

        X_tensor = torch.FloatTensor(X).to(self.device)
        errors = np.zeros((len(X), len(self.autoencoders)))

        with torch.no_grad():
            for i, ae in enumerate(self.autoencoders):
                recon = ae(X_tensor)
                err = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
                errors[:, i] = err

        # Max reconstruction error across ensemble
        return errors.max(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions based on threshold.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        y_pred : np.ndarray
        """
        scores = self.score(X)
        return (scores > self.threshold).astype(int)

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
        from sklearn.metrics import f1_score
        results = {
            "DR": compute_detection_rate(y_test, y_pred),
            "FPR": compute_fpr(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": self.threshold,
        }
        logger.info(
            "Kitsune: DR=%.4f, FPR=%.4f, F1=%.4f",
            results["DR"],
            results["FPR"],
            results["F1"],
        )
        return results
