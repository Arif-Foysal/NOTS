"""
detector/adaptive_baseline.py
==============================
Exponentially Weighted Moving-Average baseline update with poisoning resistance.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from topology.persistence import frechet_mean_diagram, merge_diagrams

logger = logging.getLogger(__name__)


class AdaptiveBaseline:
    """Adaptive baseline for concept-drift handling.

    Updates D_norm via EWM only when the live traffic is close enough to
    the current baseline, preventing adversarial poisoning.

    Parameters
    ----------
    D_norm_initial : dict
        Initial baseline persistence diagram.
    alpha : float
        EWM weight for new observations.
    tau : float
        Detection threshold.  Baseline updates are accepted only when
        ``W_current < tau / 2``.
    """

    def __init__(
        self,
        D_norm_initial: Dict[str, np.ndarray],
        alpha: float = 0.05,
        tau: float = 1.0,
    ) -> None:
        self.D_norm = {k: v.copy() for k, v in D_norm_initial.items()}
        self.alpha = alpha
        self.tau = tau
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        D_live: Dict[str, np.ndarray],
        W_current: float,
    ) -> Dict[str, np.ndarray]:
        """Conditionally update the baseline.

        If ``W_current < tau / 2``, blend D_live into D_norm with weight α.
        Otherwise, skip the update (poisoning resistance).

        Parameters
        ----------
        D_live : dict
            Live persistence diagram.
        W_current : float
            Current Wasserstein distance between D_norm and D_live.

        Returns
        -------
        D_norm_updated : dict
            The (possibly updated) baseline.
        """
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "W": W_current,
            "updated": False,
        }

        threshold = self.tau / 2.0

        if W_current < threshold:
            # Safe to update — blend diagrams
            self.D_norm = self._blend(self.D_norm, D_live, self.alpha)
            record["updated"] = True
            logger.debug("Baseline updated (W=%.4f < %.4f)", W_current, threshold)
        else:
            logger.debug(
                "Baseline update SKIPPED — poisoning resistance "
                "(W=%.4f >= %.4f)",
                W_current,
                threshold,
            )

        self.history.append(record)
        return self.D_norm

    def _blend(
        self,
        D_old: Dict[str, np.ndarray],
        D_new: Dict[str, np.ndarray],
        alpha: float,
    ) -> Dict[str, np.ndarray]:
        """Blend two persistence diagrams.

        Merge their point sets, then re-cluster to the original size using
        the Fréchet-mean approximation.

        Parameters
        ----------
        D_old : dict
        D_new : dict
        alpha : float
            Weight for D_new.

        Returns
        -------
        D_blended : dict
        """
        blended: Dict[str, np.ndarray] = {}

        for k in (0, 1):
            key = f"dgm_{k}"
            old = D_old.get(key, np.empty((0, 2)))
            new = D_new.get(key, np.empty((0, 2)))

            if old.size == 0 and new.size == 0:
                blended[key] = np.empty((0, 2), dtype=np.float64)
                continue

            if old.size == 0:
                blended[key] = new.copy()
                continue
            if new.size == 0:
                blended[key] = old.copy()
                continue

            # Weighted blend: keep (1-alpha) fraction of old, alpha of new
            # We achieve this by over-representing old points
            n_old = max(1, int(len(old) * (1 - alpha) / alpha))
            repeated_old = old[np.random.choice(len(old), size=n_old, replace=True)]
            merged = np.vstack([repeated_old, new])

            # Re-cluster to original diagram size
            target_n = len(old)
            blended_dgm = frechet_mean_diagram(
                [{"dgm_0": merged if k == 0 else np.empty((0, 2)),
                  "dgm_1": merged if k == 1 else np.empty((0, 2))}],
                n_clusters=target_n,
            )
            blended[key] = blended_dgm.get(key, np.empty((0, 2)))

        return blended

    def get_update_summary(self) -> Dict[str, Any]:
        """Return a summary of the update history.

        Returns
        -------
        summary : dict
            ``{'n_updates': int, 'n_skips': int, 'total': int,
               'update_rate': float}``
        """
        n_updates = sum(1 for r in self.history if r["updated"])
        n_total = len(self.history)
        return {
            "n_updates": n_updates,
            "n_skips": n_total - n_updates,
            "total": n_total,
            "update_rate": n_updates / max(1, n_total),
        }
