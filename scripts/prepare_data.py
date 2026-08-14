"""Convert the downloaded OULAD CSVs to parquet, so they fit in git.

WHY YOU NEED THIS
-----------------
`studentVle.csv` has ~10.7 million rows and is roughly 300-450 MB on disk.
GitHub rejects any single file over 100 MB, and warns above 50 MB. So the raw
CSVs cannot simply be committed.

Parquet solves it. It is a column-oriented file format: instead of storing row
by row like a CSV, it stores each column together and compresses it. That works
extremely well on data like this, where the same student IDs and page IDs repeat
millions of times and the numbers are small. Expect the clickstream to shrink by
roughly 10x, comfortably under the limit.

Two other things you get for free: parquet loads several times faster than CSV,
and it remembers column types, so `date` stays an integer instead of being
re-guessed on every read.

USAGE
-----
Download and unzip the dataset, then:

    python scripts/prepare_data.py --source ~/Downloads/oulad

It validates every file against src/oulad/schema.py, writes parquet into
data/raw/, and reports what is safe to commit.

The CSVs stay gitignored; the parquet files are committed. `oulad.load` prefers
parquet automatically, so nothing else in the project changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from oulad import paths
from oulad.load import SchemaError, _check
from oulad.schema import TABLES

GITHUB_HARD_LIMIT_MB = 100
GITHUB_WARN_MB = 50


def human_mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing the seven downloaded .csv files.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the .parquet files. Defaults to data/raw/.",
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli"],
        help="Parquet compression. zstd gives the smallest files; snappy is "
        "fastest to read. Either is fine here.",
    )
    args = ap.parse_args()

    out = args.out or paths.RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    if not args.source.is_dir():
        raise SystemExit(f"--source {args.source} is not a directory")

    print(f"Reading CSVs from : {args.source}")
    print(f"Writing parquet to: {out}")
    print(f"Compression       : {args.compression}\n")

    rows = []
    problems = []

    for name, spec in TABLES.items():
        src = args.source / spec.filename
        if not src.exists():
            problems.append(f"missing: {spec.filename}")
            print(f"  {name:22s} NOT FOUND ({spec.filename})")
            continue

        csv_mb = human_mb(src.stat().st_size)
        print(f"  {name:22s} reading {csv_mb:>7.1f} MB CSV ...", end="", flush=True)

        df = pd.read_csv(src)

        # Validate before writing, so a bad download is caught now rather than
        # halfway through Phase 2.
        try:
            _check(df, spec, strict_grain=(name != "studentVle"))
        except SchemaError as e:
            problems.append(f"{spec.filename}: {e}")
            print(" FAILED VALIDATION")
            print(f"      {e}")
            continue

        dest = out / f"{Path(spec.filename).stem}.parquet"
        df.to_parquet(dest, compression=args.compression, index=False)
        pq_bytes = dest.stat().st_size
        pq_mb = human_mb(pq_bytes)

        # Ratio from raw byte counts, not the rounded MB figures, so tiny
        # tables do not report a nonsense "0.0x".
        ratio = src.stat().st_size / max(pq_bytes, 1)
        note = f"{ratio:.1f}x smaller" if ratio >= 1 else f"{1 / ratio:.1f}x larger"
        print(f" -> {pq_mb:>7.2f} MB parquet  ({note})")

        rows.append(
            {
                "table": name,
                "rows": len(df),
                "csv_mb": round(csv_mb, 1),
                "parquet_mb": round(pq_mb, 1),
            }
        )

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    summary = pd.DataFrame(rows)
    total_pq = summary["parquet_mb"].sum()

    print(f"\n  {'table':22s} {'rows':>12s} {'CSV MB':>9s} {'parquet MB':>12s}  commit?")
    for r in summary.itertuples(index=False):
        if r.parquet_mb > GITHUB_HARD_LIMIT_MB:
            verdict = "NO - over GitHub's 100 MB limit"
        elif r.parquet_mb > GITHUB_WARN_MB:
            verdict = "ok, but GitHub will warn"
        else:
            verdict = "yes"
        print(
            f"  {r.table:22s} {r.rows:>12,} {r.csv_mb:>9.1f} "
            f"{r.parquet_mb:>12.1f}  {verdict}"
        )
    print(f"\n  total parquet: {total_pq:.1f} MB")

    if total_pq > 400:
        print("\n  Note: that is a large repo. Consider Git LFS, or committing")
        print("  only the six small tables and keeping studentVle local.")

    print("\nDone. Next:")
    print("  git add data/raw/*.parquet")
    print("  git commit -m 'Add OULAD data as parquet'")
    print("\nThen: python scripts/phase1_inspect.py")


if __name__ == "__main__":
    main()
