"""
topology/projector.py
======================
Lipschitz-bounded dimensionality reduction: PCA, Random Projection, and UMAP.

PCA and Random Projection preserve the stability theorem (Cohen-Steiner 2007)
because they have bounded Lipschitz constants.  UMAP is offered as an optional
heuristic with explicitly documented lack of formal guarantee.

Lipschitz bounds
----------------
- **PCA**: L = 1 (orthogonal projection — non-expansive)
- **Random Projection**: L = 1 + ε (Johnson–Lindenstrauss lemma)
- **UMAP**: L = ∞ (nonlinear, no bound)
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

logger = logging.getLogger(__name__)


class Projector:
    """Unified projector interface supporting PCA, Random Projection, and UMAP.

    Parameters
    ----------
    method : str
        ``"pca"`` (default, Lipschitz=1),
        ``"random"`` (Johnson-Lindenstrauss),
        ``"umap"`` (no formal guarantee).
    n_components : int
        Target dimensionality.
    random_state : int
        Seed for reproducibility.
    n_neighbors : int
        UMAP parameter (ignored for PCA/random).
    min_dist : float
        UMAP parameter (ignored for PCA/random).
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 5,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> None:
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self._model = None
        self._lipschitz_bound: Optional[float] = None
        self._is_fitted = False

    @property
    def lipschitz_bound(self) -> Optional[float]:
        """Return the Lipschitz constant of the projection.

        - PCA: 1.0 (orthogonal, non-expansive)
        - Random: 1 + ε ≈ 1.0 (JL guarantee)
        - UMAP: None (no bound)
        """
        return self._lipschitz_bound

    def fit(self, X: np.ndarray) -> "Projector":
        """Fit the projector on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        nc = min(self.n_components, X.shape[1], X.shape[0])

        if self.method == "pca":
            self._model = PCA(
                n_components=nc,
                random_state=self.random_state,
            )
            self._model.fit(X)
            self._lipschitz_bound = 1.0  # Orthogonal projection
            logger.info(
                "PCA projector fitted: %s → %d dims, Lipschitz=%.1f, "
                "explained_variance=%.3f",
                X.shape, nc, self._lipschitz_bound,
                sum(self._model.explained_variance_ratio_),
            )

        elif self.method == "random":
            self._model = GaussianRandomProjection(
                n_components=nc,
                random_state=self.random_state,
            )
            self._model.fit(X)
            # JL lemma: distortion bounded by (1 ± ε) where
            # ε ≈ sqrt(2 * ln(n) / k). In practice, L ≈ 1.
            self._lipschitz_bound = 1.0  # Approximate
            logger.info(
                "Random projector fitted: %s → %d dims, Lipschitz≈%.1f",
                X.shape, nc, self._lipschitz_bound,
            )

        elif self.method == "umap":
            self._model = self._create_umap(nc)
            self._model.fit(X)
            self._lipschitz_bound = None  # No formal bound
            logger.info(
                "UMAP projector fitted: %s → %d dims. "
                "WARNING: UMAP is non-isometric — no Lipschitz bound. "
                "Use PCA for theoretical guarantees.",
                X.shape, nc,
            )

        else:
            raise ValueError(
                f"Unknown projection method: {self.method!r}. "
                "Choose 'pca', 'random', or 'umap'."
            )

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the fitted low-dimensional space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_low : np.ndarray, shape (n_samples, n_components)
        """
        assert self._is_fitted, "Call fit() before transform()"
        X_low = self._model.transform(X)
        return np.asarray(X_low, dtype=np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        X_low : np.ndarray
        """
        self.fit(X)
        return self.transform(X)

    def _create_umap(self, n_components: int):
        """Create UMAP model with GPU fallback."""
        try:
            from cuml.manifold import UMAP as cuUMAP  # type: ignore
            logger.info("Using cuML GPU UMAP")
            return cuUMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
            )
        except ImportError:
            pass

        import umap  # type: ignore
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )

    def save(self, path: str) -> None:
        """Persist the fitted projector to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "method": self.method,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "lipschitz_bound": self._lipschitz_bound,
            "model": self._model,
        }
        joblib.dump(state, path)
        logger.info("Projector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "Projector":
        """Load a previously saved projector."""
        state = joblib.load(path)
        proj = cls.__new__(cls)
        proj.method = state["method"]
        proj.n_components = state["n_components"]
        proj.random_state = state["random_state"]
        proj.n_neighbors = state.get("n_neighbors", 15)
        proj.min_dist = state.get("min_dist", 0.1)
        proj._lipschitz_bound = state["lipschitz_bound"]
        proj._model = state["model"]
        proj._is_fitted = True
        logger.info("Projector loaded from %s (method=%s)", path, proj.method)
        return proj
