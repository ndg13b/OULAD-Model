"""Where things live on disk.

Every other module imports paths from here rather than hardcoding strings, so
that moving the data directory is a one-line change.
"""

from pathlib import Path

# This file is at <repo>/src/oulad/paths.py, so the repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # the seven CSVs, exactly as downloaded
INTERIM_DIR = DATA_DIR / "interim"  # cached intermediate tables (parquet)

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the data/report directories if they do not already exist."""
    for d in (RAW_DIR, INTERIM_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
