"""
game_theory/sampler.py
======================
Runtime feature-subset sampler from the Nash-optimal distribution S*.

Sampling is O(1) per call — just ``np.random.choice``.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class NashSampler:
    """O(1) runtime sampler from the Nash distribution S*.

    Parameters
    ----------
    S_star : np.ndarray
        Probability distribution over feature subsets.
    subset_indices : list[list[int]]
        Feature index lists for each subset.  ``subset_indices[i]`` is the list
        of column indices for subset *i*.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        S_star: np.ndarray,
        subset_indices: List[List[int]],
        random_state: int = 42,
    ) -> None:
        self.S_star = np.asarray(S_star, dtype=np.float64)
        self.subset_indices = subset_indices
        self.rng = np.random.RandomState(random_state)

        assert len(self.S_star) == len(subset_indices), (
            f"S_star length ({len(self.S_star)}) != subset_indices length "
            f"({len(subset_indices)})"
        )

        # Renormalise to avoid numerical issues
        self.S_star = np.maximum(self.S_star, 0.0)
        total = self.S_star.sum()
        if total > 0:
            self.S_star /= total
        else:
            # Fallback to uniform
            self.S_star = np.ones(len(self.S_star)) / len(self.S_star)

        logger.info(
            "NashSampler initialised: %d subsets, entropy = %.3f bits",
            len(self.S_star),
            self._entropy(),
        )

    def _entropy(self) -> float:
        """Shannon entropy of S* in bits."""
        p = self.S_star[self.S_star > 0]
        return float(-np.sum(p * np.log2(p)))

    def sample(self) -> List[int]:
        """Draw one feature subset index according to S*.

        Returns
        -------
        feature_indices : list[int]
            Column indices for the sampled feature subset.
        """
        idx = int(self.rng.choice(len(self.S_star), p=self.S_star))
        return self.subset_indices[idx]

    def sample_batch(self, n: int) -> List[List[int]]:
        """Draw *n* feature subsets.

        Parameters
        ----------
        n : int

        Returns
        -------
        batch : list[list[int]]
        """
        indices = self.rng.choice(len(self.S_star), size=n, p=self.S_star)
        return [self.subset_indices[int(i)] for i in indices]
