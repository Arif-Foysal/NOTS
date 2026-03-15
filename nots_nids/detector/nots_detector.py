"""
detector/nots_detector.py
=========================
Main NOTS detector — integrates dimensionality reduction (PCA/UMAP),
persistent homology (Ripser), Wasserstein distance, and game-theoretic
Nash-optimal feature sampling.

Critical design decisions
-------------------------
- **Pre-fitted projectors per subset**: During ``fit()``, one projector is
  trained per feature subset and stored.  At detection time, the stored
  projector is looked up — NO per-window refit.  This ensures Wasserstein
  distances across windows are computed in the *same* embedding space.

- **Payoff matrix columns = attacker strategies**: Columns of the payoff
  matrix represent perturbation *directions* (random, gradient-aligned,
  feature-importance-based), not data windows.

- **Lipschitz-bounded projection**: Default uses PCA (Lipschitz = 1),
  preserving the Cohen-Steiner stability guarantee.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from config import Config
from detector.baseline_builder import build_baseline
from game_theory.nash_solver import solve_nash_equilibrium
from game_theory.sampler import NashSampler
from topology.projector import Projector
from topology.ripser_filtration import compute_persistence_diagram
from topology.wasserstein import wasserstein_distance

logger = logging.getLogger(__name__)


class NOTSDetector:
    """Nash-Optimized Topological Shield detector.

    Orchestrates the full detection pipeline: projection, persistent homology,
    Wasserstein distance, and Nash-sampled feature subsets.

    Parameters
    ----------
    config : Config
        Central configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.projector: Optional[Projector] = None          # Main projector (full features)
        self.subset_projectors: Dict[int, Projector] = {}   # Pre-fitted per subset
        self.nash_sampler: Optional[NashSampler] = None
        self.D_norm: Optional[Dict[str, np.ndarray]] = None
        self.tau: Optional[float] = None
        self.epsilon_min: Optional[float] = None
        self._feature_cols: Optional[List[str]] = None
        self._subset_indices: Optional[List[List[int]]] = None

    # ── Training ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_windows: List[Dict],
        val_windows: List[Dict],
        feature_cols: Optional[List[str]] = None,
    ) -> "NOTSDetector":
        """Fit the NOTS detector.

        Steps
        -----
        1. Fit main projector on all training point clouds.
        2. Build D_norm from benign training windows.
        3. Generate feature subsets and pre-fit one projector per subset.
        4. Estimate payoff matrix U with attacker-strategy columns.
        5. Solve Nash LP → get S* and ε_min.
        6. Set τ and initialise NashSampler.

        Parameters
        ----------
        train_windows : list[dict]
            Training windows (mixed benign/attack).
        val_windows : list[dict]
            Validation windows for payoff estimation.
        feature_cols : list[str] or None
            Feature column names (for logging).

        Returns
        -------
        self
        """
        t_start = time.perf_counter()
        cfg = self.config
        self._feature_cols = feature_cols

        # ── Step 1: Fit main projector ──────────────────────────────────
        logger.info("Step 1/6: Fitting main %s projector", cfg.PROJECTION_METHOD)
        all_points = np.vstack([w["points"] for w in train_windows])
        self.projector = Projector(
            method=cfg.PROJECTION_METHOD,
            n_components=cfg.UMAP_N_COMPONENTS,
            random_state=cfg.RANDOM_SEED,
            n_neighbors=cfg.UMAP_N_NEIGHBORS,
            min_dist=cfg.UMAP_MIN_DIST,
        )
        self.projector.fit(all_points)
        if self.projector.lipschitz_bound is not None:
            logger.info("  Lipschitz bound: L = %.2f", self.projector.lipschitz_bound)
        else:
            logger.warning(
                "  No Lipschitz bound (UMAP). Stability theorem does NOT "
                "formally apply. Use PROJECTION_METHOD='pca' for guarantees."
            )

        # ── Step 2: Build D_norm ────────────────────────────────────────
        logger.info("Step 2/6: Building baseline D_norm from benign windows")
        benign_windows = [w for w in train_windows if w["label"] == "benign"]
        if not benign_windows:
            raise ValueError("No benign windows in training set for baseline")
        self.D_norm = build_baseline(
            benign_windows,
            self.projector,
            n_baseline_windows=cfg.N_BASELINE_WINDOWS,
            max_dim=cfg.RIPSER_MAX_DIM,
            max_edge=cfg.RIPSER_MAX_EDGE,
            save_dir=cfg.RESULTS_DIR,
        )

        # ── Step 3: Generate subsets + pre-fit projectors ───────────────
        logger.info("Step 3/6: Generating feature subsets and pre-fitting projectors")
        n_features = train_windows[0]["points"].shape[1]
        n_subsets = cfg.N_FEATURE_SUBSETS
        rng = np.random.RandomState(cfg.RANDOM_SEED)

        subset_size = max(3, n_features // 3)
        subset_indices: List[List[int]] = []
        self.subset_projectors = {}

        for i in range(n_subsets):
            idx = sorted(rng.choice(n_features, size=subset_size, replace=False).tolist())
            subset_indices.append(idx)

            # Pre-fit a projector for this subset
            subset_points = all_points[:, idx]
            proj = Projector(
                method=cfg.PROJECTION_METHOD,
                n_components=min(cfg.UMAP_N_COMPONENTS, len(idx) - 1),
                random_state=cfg.RANDOM_SEED,
                n_neighbors=cfg.UMAP_N_NEIGHBORS,
                min_dist=cfg.UMAP_MIN_DIST,
            )
            proj.fit(subset_points)
            self.subset_projectors[i] = proj

        self._subset_indices = subset_indices
        logger.info("  Pre-fitted %d subset projectors", len(self.subset_projectors))

        # ── Step 4: Payoff matrix with attacker-strategy columns ────────
        logger.info("Step 4/6: Estimating payoff matrix (attacker strategies as columns)")
        attack_val = [w for w in val_windows if w["label"] == "attack"]
        if not attack_val:
            logger.warning("No attack windows in validation — using all val windows")
            attack_val = val_windows

        payoff_matrix = self._build_payoff_matrix(
            subset_indices, attack_val, all_points, rng
        )

        # ── Step 5: Solve Nash LP ──────────────────────────────────────
        logger.info("Step 5/6: Solving Nash equilibrium LP")
        nash_result = solve_nash_equilibrium(payoff_matrix, n_subsets)
        S_star = nash_result["S_star"]
        self.epsilon_min = nash_result["epsilon_min"]

        # ── Step 6: Set τ and initialise sampler ────────────────────────
        self.tau = self.epsilon_min * cfg.TAU_MULTIPLIER
        logger.info("Step 6/6: τ = %.6f (ε_min=%.6f × multiplier=%.2f)",
                     self.tau, self.epsilon_min, cfg.TAU_MULTIPLIER)

        self.nash_sampler = NashSampler(
            S_star=S_star,
            subset_indices=subset_indices,
            random_state=cfg.RANDOM_SEED,
        )

        dt = time.perf_counter() - t_start
        logger.info("NOTS detector fitted in %.1f s", dt)
        return self

    def _build_payoff_matrix(
        self,
        subset_indices: List[List[int]],
        attack_windows: List[Dict],
        all_training_points: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Build payoff matrix U where columns = attacker strategies.

        Attacker strategies (columns):
          0..K-1 : random perturbation directions
          K..2K-1 : feature-importance-aligned perturbations
          2K..3K-1 : coordinate-aligned perturbations on most variable features

        Parameters
        ----------
        subset_indices : list[list[int]]
        attack_windows : list[dict]
        all_training_points : np.ndarray
        rng : np.random.RandomState

        Returns
        -------
        U : np.ndarray, shape (n_subsets, n_attacker_strategies)
        """
        n_features = attack_windows[0]["points"].shape[1]
        n_subsets = len(subset_indices)
        n_sample = min(5, len(attack_windows))
        delta_max = max(self.config.DELTA_VALUES)
        perturbable = self.config.PERTURBABLE_FEATURES or list(range(n_features))

        # Generate K attacker perturbation directions per category
        K = 3

        # Category 1: Random directions
        random_dirs = []
        for _ in range(K):
            d = np.zeros(n_features)
            d[perturbable] = rng.randn(len(perturbable))
            d /= (np.linalg.norm(d) + 1e-10)
            random_dirs.append(d * delta_max)

        # Category 2: Feature-importance (high-variance features)
        variances = all_training_points.var(axis=0)
        top_var_idx = np.argsort(variances)[-len(perturbable):]
        importance_dirs = []
        for k in range(K):
            d = np.zeros(n_features)
            # Perturb top features proportional to variance
            for idx in top_var_idx[-max(3, len(perturbable)//4):]:
                if idx in perturbable:
                    d[idx] = delta_max * ((-1) ** k)
            importance_dirs.append(d)

        # Category 3: Coordinate-aligned
        coord_dirs = []
        for k in range(K):
            d = np.zeros(n_features)
            if k < len(perturbable):
                d[perturbable[k % len(perturbable)]] = delta_max
            coord_dirs.append(d)

        all_perturbations = random_dirs + importance_dirs + coord_dirs
        n_strategies = len(all_perturbations)

        # Compute payoff U[i, j]: detection payoff when defender uses subset i
        # and attacker uses perturbation j
        U = np.zeros((n_subsets, n_strategies), dtype=np.float64)

        for i, subset in enumerate(subset_indices):
            proj = self.subset_projectors[i]
            for j, perturbation in enumerate(all_perturbations):
                # Average over sample of attack windows
                W_total = 0.0
                for k in range(n_sample):
                    pts = attack_windows[k]["points"].copy()
                    # Apply perturbation (clip to [0, 1])
                    pts_adv = np.clip(pts + perturbation[np.newaxis, :], 0.0, 1.0)
                    pts_sub = pts_adv[:, subset]
                    pts_low = proj.transform(pts_sub)
                    dgm = compute_persistence_diagram(
                        pts_low,
                        max_dim=self.config.RIPSER_MAX_DIM,
                        max_edge=self.config.RIPSER_MAX_EDGE,
                    )
                    W = wasserstein_distance(dgm, self.D_norm)
                    W_total += W
                U[i, j] = W_total / n_sample

            if (i + 1) % 5 == 0:
                logger.info("  Payoff matrix: %d / %d subsets", i + 1, n_subsets)

        logger.info(
            "Payoff matrix: shape=%s, mean=%.4f, min=%.4f",
            U.shape, U.mean(), U.min(),
        )
        return U

    # ── Detection ────────────────────────────────────────────────────────

    def detect(self, window: Dict) -> Dict[str, Any]:
        """Run detection on a single window.

        Steps
        -----
        1. Sample feature subset ω from NashSampler.
        2. Look up pre-fitted projector for ω (no refit!).
        3. Project window[:, ω] through the stored projector.
        4. Compute persistence diagram D_live.
        5. Compute W = wasserstein_distance(D_norm, D_live).
        6. Return alert decision.

        Parameters
        ----------
        window : dict
            Must have ``'points'`` key.

        Returns
        -------
        result : dict
            ``{'alert': bool, 'W': float, 'omega': list[int], 'tau': float}``
        """
        assert self.nash_sampler is not None, "Call fit() before detect()"
        assert self.D_norm is not None
        assert self.tau is not None

        # Step 1: sample feature subset
        omega = self.nash_sampler.sample()

        # Step 2-3: look up pre-fitted projector and project
        subset_idx = self._find_subset_index(omega)
        proj = self.subset_projectors[subset_idx]
        pts = window["points"][:, omega]
        pts_low = proj.transform(pts)

        # Step 4: persistence diagram
        dgm = compute_persistence_diagram(
            pts_low,
            max_dim=self.config.RIPSER_MAX_DIM,
            max_edge=self.config.RIPSER_MAX_EDGE,
        )

        # Step 5: Wasserstein distance
        W = wasserstein_distance(dgm, self.D_norm)

        # Step 6: alert decision
        alert = W >= self.tau

        return {
            "alert": bool(alert),
            "W": float(W),
            "omega": omega,
            "tau": float(self.tau),
        }

    def _find_subset_index(self, omega: List[int]) -> int:
        """Find the subset index for a given feature list."""
        for i, subset in enumerate(self._subset_indices):
            if subset == omega:
                return i
        # Fallback: shouldn't happen if sampler is correct
        logger.warning("Subset not found in pre-fitted projectors, using index 0")
        return 0

    def detect_batch(self, windows: List[Dict]) -> List[Dict[str, Any]]:
        """Run detection on a batch of windows.

        Parameters
        ----------
        windows : list[dict]

        Returns
        -------
        results : list[dict]
        """
        results = []
        for i, w in enumerate(windows):
            result = self.detect(w)
            results.append(result)
            if (i + 1) % self.config.BATCH_SIZE_WINDOWS == 0:
                logger.info("Detected %d / %d windows", i + 1, len(windows))
        return results

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist all detector components to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "D_norm": self.D_norm,
            "tau": self.tau,
            "epsilon_min": self.epsilon_min,
            "subset_indices": self._subset_indices,
            "nash_sampler_S_star": self.nash_sampler.S_star if self.nash_sampler else None,
        }
        joblib.dump(state, str(p / "nots_state.joblib"))

        if self.projector is not None:
            self.projector.save(str(p / "projector_main.joblib"))

        # Save subset projectors
        for i, proj in self.subset_projectors.items():
            proj.save(str(p / f"projector_subset_{i}.joblib"))

        logger.info("NOTS detector saved to %s", path)

    def load(self, path: str) -> "NOTSDetector":
        """Load detector state from disk.

        Parameters
        ----------
        path : str
            Directory path previously passed to ``save()``.

        Returns
        -------
        self
        """
        p = Path(path)
        state = joblib.load(str(p / "nots_state.joblib"))
        self.config = state["config"]
        self.D_norm = state["D_norm"]
        self.tau = state["tau"]
        self.epsilon_min = state["epsilon_min"]
        self._subset_indices = state["subset_indices"]

        if state.get("nash_sampler_S_star") is not None:
            self.nash_sampler = NashSampler(
                S_star=state["nash_sampler_S_star"],
                subset_indices=self._subset_indices,
                random_state=self.config.RANDOM_SEED,
            )

        # Load main projector
        main_proj_path = p / "projector_main.joblib"
        if main_proj_path.exists():
            self.projector = Projector.load(str(main_proj_path))

        # Load subset projectors
        self.subset_projectors = {}
        if self._subset_indices:
            for i in range(len(self._subset_indices)):
                subset_path = p / f"projector_subset_{i}.joblib"
                if subset_path.exists():
                    self.subset_projectors[i] = Projector.load(str(subset_path))

        logger.info("NOTS detector loaded from %s", path)
        return self
