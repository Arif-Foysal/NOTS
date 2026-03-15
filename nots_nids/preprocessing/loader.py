"""
preprocessing/loader.py
=======================
Load and merge CSV files for CICIDS-2017, UNSW-NB15, and NSL-KDD datasets.

Each loader returns ``(df, label_col)`` where *label_col* is the name of the
column that holds string labels.  All other columns are kept as-is (cleaning
happens in ``cleaner.py``).
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_class_distribution(df: pd.DataFrame, label_col: str) -> None:
    """Log class distribution table."""
    dist = df[label_col].value_counts()
    pct = df[label_col].value_counts(normalize=True) * 100
    table = pd.DataFrame({"count": dist, "pct": pct.round(2)})
    logger.info("Class distribution:\n%s", table.to_string())


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df.columns = df.columns.str.strip()
    return df


# ── CICIDS-2017 ──────────────────────────────────────────────────────────────

def load_cicids2017(data_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load all CICIDS-2017 CSVs, concatenate, and return ``(df, label_col)``.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the CICIDS-2017 CSV files.

    Returns
    -------
    df : pd.DataFrame
        Concatenated DataFrame with all flows.
    label_col : str
        Name of the label column (``'Label'`` after stripping).
    """
    p = Path(data_dir)
    csv_files = sorted(p.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download MachineLearningCSV.zip from the CIC website."
        )

    logger.info("Loading %d CICIDS-2017 CSV files from %s", len(csv_files), data_dir)

    chunks = []
    for fp in csv_files:
        logger.info("  Reading %s …", fp.name)
        try:
            chunk = pd.read_csv(fp, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            chunk = pd.read_csv(fp, encoding="latin-1", low_memory=False)
        chunk = _strip_columns(chunk)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Identify label column (may be " Label" or "Label" after strip)
    label_col = "Label"
    if label_col not in df.columns:
        candidates = [c for c in df.columns if "label" in c.lower()]
        if candidates:
            label_col = candidates[0]
        else:
            raise KeyError("Cannot find a label column in CICIDS-2017 data")

    # Drop rows where label is NaN
    n_before = len(df)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d rows with NaN label", n_dropped)

    logger.info("CICIDS-2017 loaded: shape=%s", df.shape)
    _print_class_distribution(df, label_col)
    return df, label_col


# ── UNSW-NB15 ────────────────────────────────────────────────────────────────

def load_unsw_nb15(data_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load UNSW-NB15 train + test CSVs.

    Returns ``(df, label_col)`` where *label_col* = ``'attack_cat'``.
    The binary column ``'label'`` (0/1) is also present.
    """
    p = Path(data_dir)
    train_path = p / "UNSW_NB15_training-set.csv"
    test_path = p / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing UNSW-NB15 files in {data_dir}. "
            "Need UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv."
        )

    logger.info("Loading UNSW-NB15 from %s", data_dir)
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_test = pd.read_csv(test_path, encoding="utf-8")
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = _strip_columns(df)

    label_col = "attack_cat"
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found in UNSW-NB15")

    # Normalise labels
    df[label_col] = df[label_col].fillna("Normal").str.strip()

    # Drop rows with NaN label
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    logger.info("UNSW-NB15 loaded: shape=%s", df.shape)
    _print_class_distribution(df, label_col)
    return df, label_col


# ── NSL-KDD ──────────────────────────────────────────────────────────────────

# NSL-KDD attack type → category mapping
_NSL_KDD_CATEGORY_MAP = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    "httptunnel": "R2L",
}

_NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty_level",
]


def load_nsl_kdd(data_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load NSL-KDD train + test files.

    Attack types are mapped to 4 categories (DoS, Probe, R2L, U2R).
    Normal traffic is labelled ``'BENIGN'``.

    Returns ``(df, label_col)`` where *label_col* = ``'attack_cat'``.
    """
    p = Path(data_dir)
    train_path = p / "KDDTrain+.txt"
    test_path = p / "KDDTest+.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"KDDTrain+.txt not found in {data_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"KDDTest+.txt not found in {data_dir}")

    logger.info("Loading NSL-KDD from %s", data_dir)

    df_train = pd.read_csv(train_path, header=None, names=_NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=_NSL_KDD_COLUMNS)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Map attack types to categories
    label_col = "attack_cat"
    df[label_col] = df["attack_type"].str.strip().str.rstrip(".")
    df[label_col] = df[label_col].map(
        lambda x: "BENIGN" if x == "normal" else _NSL_KDD_CATEGORY_MAP.get(x, "Other")
    )

    # Drop difficulty_level (not a feature)
    df = df.drop(columns=["difficulty_level", "attack_type"], errors="ignore")

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != label_col]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    logger.info("NSL-KDD loaded: shape=%s", df.shape)
    _print_class_distribution(df, label_col)
    return df, label_col
