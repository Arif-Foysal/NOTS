"""
NOTS-NIDS Configuration
=======================
Nash-Optimized Topological Shields: all hyperparameters and paths in one place.

Target venue: IEEE Transactions on Information Forensics and Security (T-IFS).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Central configuration for the NOTS-NIDS framework."""

    # ── Reproducibility ──────────────────────────────────────────────────
    RANDOM_SEED: int = 42

    # ── Windowing ────────────────────────────────────────────────────────
    WINDOW_SIZE: int = 100           # N flows per point cloud
    WINDOW_STEP: int = 50            # 50 % overlap by default

    # ── Projection (R^d → R^k) ──────────────────────────────────────────
    # "pca" = Lipschitz-bounded (provable stability),
    # "umap" = better manifold capture but no formal guarantee,
    # "random" = Johnson-Lindenstrauss random projection
    PROJECTION_METHOD: str = "pca"
    UMAP_N_COMPONENTS: int = 5
    UMAP_N_NEIGHBORS: int = 15
    UMAP_MIN_DIST: float = 0.1

    # ── Persistent Homology (Ripser) ─────────────────────────────────────
    RIPSER_MAX_DIM: int = 1          # Compute β₀ and β₁ only
    RIPSER_MAX_EDGE: float = 2.0     # Maximum filtration radius

    # ── Adversarial Parameters ───────────────────────────────────────────
    DELTA_VALUES: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    # Which feature indices the attacker can perturb (None = all).
    # In real networks, protocol-constrained features (TCP flags, ports)
    # cannot be freely perturbed — only timing/size features can.
    PERTURBABLE_FEATURES: Optional[List[int]] = None
    MAX_ATTACK_WINDOWS: int = 100    # Cap for attack windows in experiments

    # ── Adaptive Baseline (EWM) ──────────────────────────────────────────
    ALPHA_EWM: float = 0.05
    TAU_MULTIPLIER: float = 1.0      # τ = ε_min × TAU_MULTIPLIER

    # ── Data Splits ──────────────────────────────────────────────────────
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    TEST_RATIO: float = 0.20

    # ── Processing ───────────────────────────────────────────────────────
    BATCH_SIZE_WINDOWS: int = 50     # Save checkpoint every N windows

    # ── Paths ────────────────────────────────────────────────────────────
    RESULTS_DIR: str = "results/"
    FIGURES_DIR: str = "results/figures/"

    # ── CICIDS-2017 ──────────────────────────────────────────────────────
    CICIDS_LABEL_COL: str = " Label"
    CICIDS_BENIGN_LABEL: str = "BENIGN"

    # ── Feature Selection ────────────────────────────────────────────────
    FEATURE_COLS: Optional[List[str]] = None   # None → auto-detect numerics
    N_FEATURE_SUBSETS: int = 8                 # Feature subsets in Nash LP

    # ── Baseline Builder ─────────────────────────────────────────────────
    N_BASELINE_WINDOWS: int = 20

    # ── Kitsune Baseline ─────────────────────────────────────────────────
    KITSUNE_N_CLUSTERS: int = 10
    KITSUNE_EPOCHS: int = 30
    KITSUNE_LR: float = 1e-3

    # ── White-box Attack ─────────────────────────────────────────────────
    FGSM_N_STEPS: int = 10
    FGSM_H: float = 1e-4             # Numerical gradient step

    # ── Efficiency Benchmark ─────────────────────────────────────────────
    BENCHMARK_N_VALUES: List[int] = field(default_factory=lambda: [50, 100, 500])
    BENCHMARK_N_REPEATS: int = 30

    # ── Multi-run (confidence intervals) ─────────────────────────────────
    N_EXPERIMENT_RUNS: int = 3       # Run each exp N times with different seeds

    # ── Sensitivity Analysis ─────────────────────────────────────────────
    SENSITIVITY_WINDOW_SIZES: List[int] = field(
        default_factory=lambda: [50, 100, 250]
    )
    SENSITIVITY_N_COMPONENTS: List[int] = field(
        default_factory=lambda: [3, 5, 10]
    )
    SENSITIVITY_MAX_EDGES: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )

    def __post_init__(self) -> None:
        """Create output directories if they do not exist."""
        Path(self.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.FIGURES_DIR).mkdir(parents=True, exist_ok=True)
