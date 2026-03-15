"""
topology/umap_reducer.py
========================
UMAP dimensionality reduction R^d → R^k (default d=80, k=5).

Falls back gracefully from GPU (cuML) to CPU (umap-learn).
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class UMAPReducer:
    """UMAP dimensionality reduction with optional GPU acceleration.

    Parameters
    ----------
    n_components : int
        Target dimensionality (default 5).
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        Random seed for reproducibility.
    use_gpu : bool
        If True, attempt to use cuML GPU UMAP.
    """

    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        use_gpu: bool = False,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.use_gpu = use_gpu
        self._model = None
        self._backend = "cpu"

        self._init_model()

    def _init_model(self) -> None:
        """Initialise the UMAP model, falling back to CPU if GPU unavailable."""
        if self.use_gpu:
            try:
                from cuml.manifold import UMAP as cuUMAP  # type: ignore[import]
                self._model = cuUMAP(
                    n_components=self.n_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    random_state=self.random_state,
                )
                self._backend = "gpu"
                logger.info("Using cuML GPU UMAP")
                return
            except ImportError:
                logger.warning("cuML not available — falling back to CPU UMAP")

        import umap  # type: ignore[import]
        self._model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        self._backend = "cpu"
        logger.info("Using CPU UMAP (umap-learn)")

    def fit(self, X: np.ndarray) -> "UMAPReducer":
        """Fit UMAP on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        logger.info("Fitting UMAP on %s (backend=%s)", X.shape, self._backend)
        self._model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the learned low-dimensional space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_low : np.ndarray, shape (n_samples, n_components)
        """
        X_low = self._model.transform(X)
        return np.asarray(X_low, dtype=np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_low : np.ndarray, shape (n_samples, n_components)
        """
        logger.info("Fit-transforming UMAP on %s (backend=%s)", X.shape, self._backend)
        X_low = self._model.fit_transform(X)
        return np.asarray(X_low, dtype=np.float64)

    def save(self, path: str) -> None:
        """Persist the fitted UMAP model to disk.

        Parameters
        ----------
        path : str
            File path (e.g. ``'results/umap_model.joblib'``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("UMAP model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "UMAPReducer":
        """Load a previously saved UMAP model.

        Parameters
        ----------
        path : str
            File path to the joblib-serialised model.

        Returns
        -------
        reducer : UMAPReducer
        """
        model = joblib.load(path)
        reducer = cls.__new__(cls)
        reducer._model = model
        reducer._backend = "loaded"
        reducer.n_components = getattr(model, "n_components", 5)
        logger.info("UMAP model loaded from %s", path)
        return reducer
