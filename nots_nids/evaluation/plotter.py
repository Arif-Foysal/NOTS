"""
evaluation/plotter.py
=====================
Generate all figures for the IEEE T-IFS paper.

Uses matplotlib with IEEE double-column style (3.5″ single / 7.0″ double).
All figures saved as PDF + PNG @ 300 DPI.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── IEEE style defaults ──────────────────────────────────────────────────────
_IEEE_RCPARAMS = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
}

SINGLE_COL_WIDTH = 3.5   # inches
DOUBLE_COL_WIDTH = 7.0   # inches

METHOD_COLORS = {
    "NOTS": "#2196F3",
    "Kitsune": "#FF9800",
    "RF": "#4CAF50",
    "LUCID": "#9C27B0",
}


def _apply_style() -> None:
    """Apply IEEE-compatible matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(_IEEE_RCPARAMS)


def _save_fig(fig: plt.Figure, save_path: str) -> None:
    """Save figure as PDF and PNG."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".png")), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s (.pdf + .png)", p.stem)


# ── Figure 1: Detection Rate Comparison ──────────────────────────────────────

def plot_detection_rate_comparison(
    exp1_results: Dict[str, pd.DataFrame],
    save_path: str,
) -> None:
    """Grouped bar chart: x = attack classes, y = DR, one bar per method.

    Parameters
    ----------
    exp1_results : dict
        ``{method_name: pd.DataFrame}`` where DataFrame has 'DR' column
        indexed by attack type.
    save_path : str
    """
    _apply_style()

    methods = list(exp1_results.keys())
    attack_types = exp1_results[methods[0]].index.tolist()
    n_groups = len(attack_types)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
    x = np.arange(n_groups)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        dr_values = exp1_results[method]["DR"].values
        color = METHOD_COLORS.get(method, f"C{i}")
        bars = ax.bar(x + i * width, dr_values, width, label=method,
                      color=color, alpha=0.85)

    ax.set_xlabel("Attack Class")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Attack Class")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(attack_types, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, save_path)


# ── Figure 2: DR vs δ (Theorem Validation) ──────────────────────────────────

def plot_wasserstein_vs_delta(
    exp2_results: Dict[str, Dict[float, Dict]],
    epsilon_min: float,
    save_path: str,
) -> None:
    """Line plot: x = δ, y = DR. Horizontal dashed line at ε_min.

    Parameters
    ----------
    exp2_results : dict
        ``{method_name: {delta: {'DR': float}}}``
    epsilon_min : float
    save_path : str
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 3.0))

    for method, results in exp2_results.items():
        deltas = sorted(results.keys())
        dr_values = [results[d]["DR"] for d in deltas]
        color = METHOD_COLORS.get(method, None)
        ax.plot(deltas, dr_values, "o-", label=method, color=color, markersize=5)

    ax.axhline(y=epsilon_min, color="red", linestyle="--", linewidth=1.5,
               label=f"Theorem bound (ε_min={epsilon_min:.3f})")

    ax.set_xlabel("Perturbation Budget δ")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate Under White-Box Attack")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(alpha=0.3)

    _save_fig(fig, save_path)


# ── Figure 3: Persistence Diagrams (3-panel) ────────────────────────────────

def plot_persistence_diagrams(
    D_norm: Dict[str, np.ndarray],
    D_live_benign: Dict[str, np.ndarray],
    D_live_attack: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """3-panel persistence diagram comparison.

    Left: D_norm  |  Centre: benign live  |  Right: attack live.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.5), sharex=True, sharey=True)
    titles = ["D_norm (Baseline)", "D_live (Benign)", "D_live (Attack)"]
    diagrams = [D_norm, D_live_benign, D_live_attack]

    for ax, dgm, title in zip(axes, diagrams, titles):
        # Diagonal
        lims = [0, 2.5]
        ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5)

        # H0
        h0 = dgm.get("dgm_0", np.empty((0, 2)))
        if h0.size > 0:
            ax.scatter(h0[:, 0], h0[:, 1], c="#2196F3", marker="o", s=15,
                       alpha=0.7, label="H₀", edgecolors="none")

        # H1
        h1 = dgm.get("dgm_1", np.empty((0, 2)))
        if h1.size > 0:
            ax.scatter(h1[:, 0], h1[:, 1], c="#FF5722", marker="^", s=20,
                       alpha=0.7, label="H₁", edgecolors="none")

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Birth")
        ax.legend(fontsize=6, loc="lower right")

    axes[0].set_ylabel("Death")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── Figure 4: Wasserstein Time Series ────────────────────────────────────────

def plot_wasserstein_timeseries(
    W_trajectory: np.ndarray,
    labels: np.ndarray,
    tau: float,
    save_path: str,
) -> None:
    """Time series: x = window index, y = W distance.  Background coloured by label.

    Parameters
    ----------
    W_trajectory : np.ndarray, shape (n_windows,)
    labels : np.ndarray, shape (n_windows,)
        Binary labels (1 = attack).
    tau : float
        Detection threshold.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.5))
    n = len(W_trajectory)
    x = np.arange(n)

    # Background colouring
    for i in range(n):
        if labels[i] == 1:
            ax.axvspan(i - 0.5, i + 0.5, facecolor="#FFCDD2", alpha=0.4)

    ax.plot(x, W_trajectory, color="#1565C0", linewidth=0.8, alpha=0.9)
    ax.axhline(y=tau, color="red", linestyle="--", linewidth=1.2,
               label=f"τ = {tau:.3f}")

    ax.set_xlabel("Window Index")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_title("Detection Signal Over Time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    _save_fig(fig, save_path)


# ── Figure 5: Ablation Bar Chart ────────────────────────────────────────────

def plot_ablation_bars(
    exp4_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Horizontal bar chart: y = variant, x = DR.

    Parameters
    ----------
    exp4_results : dict
        ``{variant_name: {'DR': float, 'FPR': float}}``
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.5))

    variants = list(exp4_results.keys())
    dr_values = [exp4_results[v]["DR"] for v in variants]
    colors = ["#2196F3" if "Full" in v else "#90CAF9" for v in variants]

    y_pos = np.arange(len(variants))
    ax.barh(y_pos, dr_values, color=colors, alpha=0.85, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants, fontsize=8)
    ax.set_xlabel("Detection Rate")
    ax.set_title("Ablation Study")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for i, v in enumerate(dr_values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    _save_fig(fig, save_path)


# ── Figure 6: Efficiency Table Figure ────────────────────────────────────────

def plot_efficiency_table(
    exp5_results: pd.DataFrame,
    save_path: str,
) -> None:
    """Render efficiency results as a formatted table figure.

    Parameters
    ----------
    exp5_results : pd.DataFrame
        Rows = N values, columns = pipeline stages with time in ms.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=exp5_results.round(2).values,
        colLabels=exp5_results.columns.tolist(),
        rowLabels=exp5_results.index.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(exp5_results.columns))))

    ax.set_title("Pipeline Stage Latency (ms per window)", fontsize=9, pad=20)
    _save_fig(fig, save_path)


# ── Figure 7: ROC Curves ────────────────────────────────────────────────────

def plot_roc_curves(
    all_results: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
) -> None:
    """ROC curves for all methods.

    Parameters
    ----------
    all_results : dict
        ``{method: {'fpr_curve': array, 'tpr_curve': array, 'auc': float}}``
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 3.5))

    for method, data in all_results.items():
        color = METHOD_COLORS.get(method, None)
        ax.plot(
            data["fpr_curve"],
            data["tpr_curve"],
            color=color,
            linewidth=1.5,
            label=f"{method} (AUC={data['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    _save_fig(fig, save_path)
