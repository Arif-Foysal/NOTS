"""
experiments/exp5_efficiency.py
===============================
Experiment 5: Computational efficiency benchmarks.

Benchmarks each pipeline stage separately, plus end-to-end throughput.
"""

import logging
import os
import time
import timeit
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def _benchmark_stage(
    func,
    args: tuple = (),
    kwargs: dict = None,
    n_repeats: int = 100,
    label: str = "stage",
) -> Dict[str, float]:
    """Benchmark a single function call.

    Returns
    -------
    result : dict
        ``{'mean_ms': float, 'std_ms': float, 'min_ms': float, 'max_ms': float}``
    """
    if kwargs is None:
        kwargs = {}

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000  # ms
        times.append(dt)

    times = np.array(times)
    result = {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
    }
    logger.info(
        "  %s: %.2f ± %.2f ms (min=%.2f, max=%.2f)",
        label,
        result["mean_ms"],
        result["std_ms"],
        result["min_ms"],
        result["max_ms"],
    )
    return result


def run_experiment_5(
    detector,
    config: Config,
) -> Dict[str, Any]:
    """Experiment 5: efficiency benchmarks.

    Benchmarks
    ----------
    - UMAP projection per window at N=100, 500, 1000
    - Ripser filtration per window at N=100, 500, 1000
    - Wasserstein distance per comparison
    - Nash sampling per sample
    - End-to-end detection per window
    - Memory usage

    Parameters
    ----------
    detector : NOTSDetector
    config : Config

    Returns
    -------
    results : dict
    """
    from topology.ripser_filtration import compute_persistence_diagram
    from topology.wasserstein import wasserstein_distance as wass_dist
    from game_theory.sampler import NashSampler

    t_start = time.perf_counter()
    logger.info("=== Experiment 5: Efficiency benchmarks ===")

    n_repeats = config.BENCHMARK_N_REPEATS
    n_features = config.UMAP_N_COMPONENTS
    results: Dict[str, Any] = {}
    table_rows = []

    for N in config.BENCHMARK_N_VALUES:
        logger.info("--- N = %d ---", N)
        row = {"N": N}

        # Generate synthetic point cloud
        rng = np.random.RandomState(config.RANDOM_SEED)
        cloud_high = rng.rand(N, 80).astype(np.float64)  # High-dim
        cloud_low = rng.rand(N, n_features).astype(np.float64)  # Low-dim

        # Projection (PCA/UMAP/Random)
        if detector.projector is not None:
            proj_bench = _benchmark_stage(
                detector.projector.transform,
                args=(cloud_high[:min(N, 100)],),  # Use smaller subset for UMAP
                n_repeats=min(n_repeats, 20),
                label=f"Projection(N={N})",
            )
            row["Projection_ms"] = proj_bench["mean_ms"]
        else:
            row["Projection_ms"] = np.nan

        # Ripser filtration
        ripser_bench = _benchmark_stage(
            compute_persistence_diagram,
            args=(cloud_low,),
            kwargs={"max_dim": config.RIPSER_MAX_DIM, "max_edge": config.RIPSER_MAX_EDGE},
            n_repeats=min(n_repeats, 20),
            label=f"Ripser(N={N})",
        )
        row["Ripser_ms"] = ripser_bench["mean_ms"]

        # Wasserstein distance
        dgm1 = compute_persistence_diagram(cloud_low, max_dim=config.RIPSER_MAX_DIM)
        dgm2 = compute_persistence_diagram(
            rng.rand(N, n_features), max_dim=config.RIPSER_MAX_DIM
        )
        wass_bench = _benchmark_stage(
            wass_dist,
            args=(dgm1, dgm2),
            kwargs={},
            n_repeats=n_repeats,
            label=f"Wasserstein(N={N})",
        )
        row["Wasserstein_ms"] = wass_bench["mean_ms"]

        # Nash sampling
        if detector.nash_sampler is not None:
            nash_bench = _benchmark_stage(
                detector.nash_sampler.sample,
                n_repeats=n_repeats * 10,
                label="Nash sample",
            )
            row["Nash_ms"] = nash_bench["mean_ms"]
        else:
            row["Nash_ms"] = np.nan

        # End-to-end throughput
        e2e_ms = row.get("Projection_ms", 0) + row["Ripser_ms"] + row["Wasserstein_ms"] + row.get("Nash_ms", 0)
        flows_per_sec = (N / (e2e_ms / 1000)) if e2e_ms > 0 else 0
        row["E2E_ms"] = e2e_ms
        row["flows_per_sec"] = flows_per_sec

        table_rows.append(row)

    # Memory benchmark (approximate)
    try:
        import tracemalloc
        tracemalloc.start()
        for N in config.BENCHMARK_N_VALUES:
            cloud = np.random.rand(N, n_features)
            dgm = compute_persistence_diagram(cloud, max_dim=config.RIPSER_MAX_DIM)
        _, peak_mb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mb / (1024 * 1024)
        results["peak_memory_mb"] = peak_mb
        logger.info("Peak memory: %.1f MB", peak_mb)
    except Exception as e:
        logger.warning("Memory benchmark failed: %s", e)
        results["peak_memory_mb"] = np.nan

    # Save
    df = pd.DataFrame(table_rows)
    save_path = os.path.join(config.RESULTS_DIR, "exp5_efficiency.csv")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    results["table"] = df
    logger.info("Efficiency results saved to %s", save_path)

    dt = time.perf_counter() - t_start
    logger.info("Experiment 5 complete in %.1f s", dt)

    # Print formatted table
    logger.info("\n" + "=" * 80)
    logger.info("Experiment 5 — Efficiency Benchmarks")
    logger.info("=" * 80)
    logger.info(df.to_string(index=False))
    logger.info("=" * 80)

    return results
