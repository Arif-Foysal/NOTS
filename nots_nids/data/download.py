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

# ── Kaggle dataset slugs ────────────────────────────────────────────────────
CICIDS2017_KAGGLE_SLUG = "dhoogla/cicids2017"


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


def download_cicids2017_kaggle(data_dir: str = "data/cicids2017") -> str:
    """Download CICIDS-2017 from Kaggle using kagglehub.

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
        import kagglehub  # noqa: F401
    except ImportError:
        print("Installing kagglehub...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kagglehub"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import kagglehub  # noqa: F811

    ensure_dir(data_dir)

    print("Downloading CICIDS-2017 from Kaggle (this may take a few minutes)...")

    # kagglehub downloads to its own cache and returns the path
    cache_path = kagglehub.dataset_download(CICIDS2017_KAGGLE_SLUG)
    print(f"Downloaded to cache: {cache_path}")

    target = Path(data_dir)

    # Look for CSVs first
    cache_csvs = list(Path(cache_path).rglob("*.csv"))
    if cache_csvs:
        for csv_file in cache_csvs:
            dest = target / csv_file.name
            if not dest.exists():
                shutil.copy2(str(csv_file), str(dest))
        csvs = _find_csvs(data_dir)
        print(f"Dataset ready: {len(csvs)} CSV files in {data_dir}")
        return data_dir

    # No CSVs — check for parquet files and convert
    cache_parquets = list(Path(cache_path).rglob("*.parquet"))
    if cache_parquets:
        print(f"Found {len(cache_parquets)} parquet files — converting to CSV...")
        try:
            import pandas as pd
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import pandas as pd
        for pq_file in cache_parquets:
            csv_name = pq_file.stem + ".csv"
            dest = target / csv_name
            if not dest.exists():
                print(f"  Converting {pq_file.name} -> {csv_name}")
                pd.read_parquet(str(pq_file)).to_csv(str(dest), index=False)
        csvs = _find_csvs(data_dir)
        print(f"Dataset ready: {len(csvs)} CSV files in {data_dir}")
        return data_dir

    # List what's actually there for debugging
    all_files = list(Path(cache_path).rglob("*"))
    file_list = [str(f.relative_to(cache_path)) for f in all_files if f.is_file()]
    raise FileNotFoundError(
        f"No CSV or parquet files found in kagglehub cache at {cache_path}. "
        f"Files found ({len(file_list)}): {file_list[:20]}"
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
