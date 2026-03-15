"""
experiments/multi_run.py
========================
Wrapper to run experiments multiple times with different seeds
to compute confidence intervals (mean +/- std).
"""

import logging
import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def run_multiple(
    run_func: Callable[..., Dict[str, Any]],
    n_runs: int,
    base_seed: int,
    config=None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Run an experiment function N times and aggregate results.

    Parameters
    ----------
    run_func : function
        The experiment function to run.
    n_runs : int
        Number of iterations.
    base_seed : int
        Initial random seed.
    config : Config or None
        If provided, ``config.RANDOM_SEED`` is updated each run so that
        internal components (Projector, NashSampler, etc.) use different seeds.
    *args, **kwargs
        Arguments passed to run_func.

    Returns
    -------
    aggregated : dict
        Mean and standard deviation of numeric results.
    """
    all_results = []

    for i in range(n_runs):
        seed = base_seed + i
        logger.info("Starting run %d/%d with seed %d", i+1, n_runs, seed)

        # Set global numpy seed
        np.random.seed(seed)

        # Also update config so internal RandomState instances use new seed
        if config is not None:
            config.RANDOM_SEED = seed

        res = run_func(*args, **kwargs)
        all_results.append(res)

    # Restore original seed in config
    if config is not None:
        config.RANDOM_SEED = base_seed

    # Aggregate (assuming results are dicts of scalars or DataFrames)
    if isinstance(all_results[0], pd.DataFrame):
        concat_df = pd.concat(all_results)
        group_cols = [c for c in concat_df.columns if not np.issubdtype(concat_df[c].dtype, np.number)]
        if not group_cols:
            summary = concat_df.agg(['mean', 'std']).T
        else:
            summary = concat_df.groupby(group_cols).agg(['mean', 'std'])
        return {"summary": summary, "all_runs": all_results}

    return {"all_runs": all_results}
