"""
preprocessing/cleaner.py
========================
Handle NaN, Inf, duplicates, label normalisation, and zero-variance removal.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(
    df: pd.DataFrame,
    label_col: str,
    nan_threshold: float = 0.50,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean a raw DataFrame.

    Steps
    -----
    1. Replace ±Inf with NaN.
    2. Drop columns with more than ``nan_threshold`` fraction NaN.
    3. Fill remaining NaN with column median.
    4. Drop exact duplicate rows.
    5. Remove columns with zero variance.
    6. For CICIDS-2017: normalise label strings (strip + title case).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from a loader.
    label_col : str
        Name of the label column.
    nan_threshold : float
        Fraction threshold above which a column is dropped entirely.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame.
    dropped_cols : list[str]
        Names of columns that were removed.
    """
    dropped_cols: List[str] = []
    n_initial = len(df)

    # 1. Replace inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced inf values with NaN in %d numeric columns", len(numeric_cols))

    # 2. Drop columns with excessive NaN
    nan_frac = df[numeric_cols].isna().mean()
    high_nan = nan_frac[nan_frac > nan_threshold].index.tolist()
    if high_nan:
        logger.info("Dropping %d columns with >%.0f%% NaN: %s",
                     len(high_nan), nan_threshold * 100, high_nan)
        df = df.drop(columns=high_nan)
        dropped_cols.extend(high_nan)
        numeric_cols = [c for c in numeric_cols if c not in high_nan]

    # 3. Fill remaining NaN with column median
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    logger.info("Filled remaining NaN with column medians")

    # 4. Drop exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        logger.info("Dropped %d duplicate rows (%.1f%%)",
                     n_dupes, 100 * n_dupes / n_before)

    # 5. Remove zero-variance columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    variances = df[numeric_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logger.info("Dropping %d zero-variance columns: %s", len(zero_var), zero_var)
        df = df.drop(columns=zero_var)
        dropped_cols.extend(zero_var)

    # 6. Normalise CICIDS-2017 labels (strip + title case)
    if label_col in df.columns and df[label_col].dtype == object:
        df[label_col] = df[label_col].str.strip().str.title()

    logger.info(
        "Cleaning done: %d → %d rows, %d columns dropped",
        n_initial, len(df), len(dropped_cols),
    )
    return df, dropped_cols


def encode_labels(
    df: pd.DataFrame,
    label_col: str,
    benign_label: str = "Benign",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Map string labels to integers.

    * ``BENIGN`` (or the dataset equivalent) → 0
    * All attacks → 1, 2, 3, … (multiclass integer encoding)
    * A binary column ``is_attack`` is also created (0 = benign, 1 = attack).

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
    benign_label : str
        Expected (cleaned) benign string.  ``clean_dataframe`` title-cases
        CICIDS labels so *"BENIGN"* becomes *"Benign"*.

    Returns
    -------
    df : pd.DataFrame
        With new columns ``label_int`` and ``is_attack``.
    label_map : dict
        ``{string_label: int}`` mapping.
    """
    unique_labels = sorted(df[label_col].unique())

    # Ensure benign is label 0
    if benign_label in unique_labels:
        unique_labels.remove(benign_label)
        unique_labels = [benign_label] + unique_labels
    else:
        # Try case-insensitive match
        match = [l for l in unique_labels if l.lower() in ("benign", "normal")]
        if match:
            benign_label = match[0]
            unique_labels.remove(benign_label)
            unique_labels = [benign_label] + unique_labels

    label_map: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}

    df["label_int"] = df[label_col].map(label_map)
    df["is_attack"] = (df["label_int"] > 0).astype(int)

    logger.info("Label encoding: %s", label_map)
    logger.info("Attack distribution: %d benign, %d attack",
                (df["is_attack"] == 0).sum(),
                (df["is_attack"] == 1).sum())
    return df, label_map
