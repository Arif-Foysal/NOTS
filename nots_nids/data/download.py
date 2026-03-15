"""
Dataset Download Helpers
========================
Auto-download CICIDS-2017 from Kaggle if not already present.
Also supports UNSW-NB15 and NSL-KDD verification.

Usage (in Colab)::

    from data.download import ensure_cicids2017
    data_dir = ensure_cicids2017()  # Downloads if needed, returns path
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Kaggle dataset slugs (tried in order) ──────────────────────────────────
# We need the original MachineLearningCSV files with ~78 features and
# multi-class Label column (BENIGN, DDoS, DoS Hulk, PortScan, etc.)
CICIDS2017_KAGGLE_SLUGS = [
    "cicdataset/cicids2017",          # Official CIC upload
    "chethuhn/network-intrusion-dataset",  # Community mirror (full CSVs)
    "dhoogla/cicids2017",             # Parquet version (fallback)
]


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_csvs(data_dir: str) -> list:
    """Recursively find CSV files in data_dir."""
    p = Path(data_dir)
    if not p.exists():
        return []
    return list(p.rglob("*.csv"))


def _setup_kaggle_token() -> None:
    """Try to set KAGGLE_API_TOKEN from Colab secrets if not already set."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            logger.info("Kaggle API token set from Colab secrets")
    except Exception:
        pass


def _validate_cicids2017_csvs(data_dir: str) -> bool:
    """Check that CSVs in data_dir look like real CICIDS-2017 (multi-class Label)."""
    csvs = _find_csvs(data_dir)
    if not csvs:
        return False
    try:
        import pandas as pd
        # Read just the header + a few rows from the first CSV
        sample = pd.read_csv(str(csvs[0]), nrows=5, encoding="utf-8", low_memory=False)
        sample.columns = sample.columns.str.strip()
        # Must have a Label column
        label_col = None
        for c in sample.columns:
            if c.lower() == "label":
                label_col = c
                break
        if label_col is None:
            print(f"  Validation failed: no 'Label' column in {csvs[0].name}")
            print(f"  Columns found: {list(sample.columns)[:10]}...")
            return False
        # Must have at least 40 feature columns (original has ~78)
        if len(sample.columns) < 40:
            print(f"  Validation failed: only {len(sample.columns)} columns (need 40+)")
            return False
        return True
    except Exception as e:
        print(f"  Validation error: {e}")
        return False


def _copy_cache_to_data_dir(cache_path: str, data_dir: str) -> int:
    """Copy CSV/parquet files from kagglehub cache to data_dir. Returns CSV count."""
    cache = Path(cache_path)
    target = Path(data_dir)

    # Try CSVs first
    cache_csvs = list(cache.rglob("*.csv"))
    if cache_csvs:
        for f in cache_csvs:
            dest = target / f.name
            if not dest.exists():
                shutil.copy2(str(f), str(dest))
        return len(_find_csvs(data_dir))

    # Try parquet -> CSV conversion
    cache_parquets = list(cache.rglob("*.parquet"))
    if cache_parquets:
        print(f"  Found {len(cache_parquets)} parquet files — converting to CSV...")
        try:
            import pandas as pd
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import pandas as pd
        for pq in cache_parquets:
            dest = target / (pq.stem + ".csv")
            if not dest.exists():
                print(f"    {pq.name} -> {dest.name}")
                pd.read_parquet(str(pq)).to_csv(str(dest), index=False)
        return len(_find_csvs(data_dir))

    return 0


def download_cicids2017_kaggle(data_dir: str = "data/cicids2017") -> str:
    """Download CICIDS-2017 from Kaggle using kagglehub.

    Tries multiple dataset slugs in order. Validates that the downloaded
    data has the expected multi-class Label column and ~78 features.

    Parameters
    ----------
    data_dir : str
        Target directory for the CSV files.

    Returns
    -------
    data_dir : str
        Path where CSVs were saved.
    """
    _setup_kaggle_token()

    # Install kagglehub if needed
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kagglehub"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import kagglehub

    ensure_dir(data_dir)

    errors = []
    for slug in CICIDS2017_KAGGLE_SLUGS:
        print(f"\nTrying Kaggle dataset: {slug}")
        try:
            cache_path = kagglehub.dataset_download(slug)
            print(f"  Downloaded to: {cache_path}")

            n_csvs = _copy_cache_to_data_dir(cache_path, data_dir)
            if n_csvs == 0:
                # List what was actually downloaded for debugging
                all_files = list(Path(cache_path).rglob("*"))
                exts = set(f.suffix for f in all_files if f.is_file())
                print(f"  No CSV/parquet found. File types: {exts}")
                errors.append(f"{slug}: no CSV/parquet files")
                continue

            if _validate_cicids2017_csvs(data_dir):
                csvs = _find_csvs(data_dir)
                print(f"Dataset ready: {len(csvs)} CSV files in {data_dir}")
                return data_dir

            # Validation failed — clean up and try next slug
            print(f"  Dataset from {slug} doesn't match expected CICIDS-2017 format")
            for f in _find_csvs(data_dir):
                f.unlink()
            errors.append(f"{slug}: validation failed (wrong format)")

        except Exception as e:
            print(f"  Failed: {e}")
            errors.append(f"{slug}: {e}")

    raise FileNotFoundError(
        f"Could not download valid CICIDS-2017 from any Kaggle source.\n"
        f"Errors: {errors}\n\n"
        f"Please download manually:\n"
        f"  1. Go to https://www.kaggle.com and search 'CICIDS 2017'\n"
        f"  2. Download the dataset with MachineLearningCSV files (~8 CSVs, ~78 features)\n"
        f"  3. Extract CSVs into: {data_dir}\n"
    )


def ensure_cicids2017(data_dir: str = "data/cicids2017") -> str:
    """Ensure CICIDS-2017 is available — download from Kaggle if missing.

    This is the main entry point. Call this before loading data.

    Parameters
    ----------
    data_dir : str
        Target directory.

    Returns
    -------
    data_dir : str
        Path where CSVs are located (may be a subdirectory if zip was nested).
    """
    csvs = _find_csvs(data_dir)

    if csvs:
        csv_dir = str(csvs[0].parent)
        print(f"CICIDS-2017 already present: {len(csvs)} CSV files in {csv_dir}")
        return csv_dir

    print("CICIDS-2017 not found locally. Attempting Kaggle download...")
    download_cicids2017_kaggle(data_dir)

    # Re-check (CSVs might be in a subdirectory after copy)
    csvs = _find_csvs(data_dir)
    if csvs:
        csv_dir = str(csvs[0].parent)
        return csv_dir

    raise FileNotFoundError(
        f"No CSV files found in {data_dir} even after download. "
        "Please download manually from "
        "https://www.unb.ca/cic/datasets/ids-2017.html"
    )


# ── UNSW-NB15 and NSL-KDD verification ──────────────────────────────────────

def check_unsw_nb15(data_dir: str = "data/unsw_nb15") -> bool:
    """Check whether UNSW-NB15 CSV files are present."""
    p = Path(data_dir)
    required = [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        logger.warning(
            "Missing UNSW-NB15 files: %s. Download from "
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            missing,
        )
        return False
    logger.info("UNSW-NB15 files present in %s", data_dir)
    return True


def check_nsl_kdd(data_dir: str = "data/nsl_kdd") -> bool:
    """Check whether NSL-KDD text files are present."""
    p = Path(data_dir)
    required = ["KDDTrain+.txt", "KDDTest+.txt"]
    missing = [f for f in required if not (p / f).exists()]
    if missing:
        logger.warning(
            "Missing NSL-KDD files: %s. Download from "
            "https://www.unb.ca/cic/datasets/nsl.html",
            missing,
        )
        return False
    logger.info("NSL-KDD files present in %s", data_dir)
    return True


def verify_all_datasets(base_dir: str = "data") -> dict:
    """Check availability of all datasets and return status dict."""
    status = {
        "cicids2017": len(_find_csvs(os.path.join(base_dir, "cicids2017"))) > 0,
        "unsw_nb15": check_unsw_nb15(os.path.join(base_dir, "unsw_nb15")),
        "nsl_kdd": check_nsl_kdd(os.path.join(base_dir, "nsl_kdd")),
    }
    available = sum(status.values())
    logger.info("Datasets available: %d / %d", available, len(status))
    return status
