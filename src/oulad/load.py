"""Reading the seven OULAD CSVs, with checks.

Every read goes through `load_table`, which validates the file against the spec
in `schema.py`. The point of validating on load is that a bad file should stop
the program at the earliest possible moment, while the error still points at the
cause. A missing column discovered here says "studentInfo.csv is missing
final_result"; the same problem discovered three joins later says "KeyError" in
a function that has nothing to do with it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths
from .schema import TABLES, VALUE_FIXES, TableSpec


class SchemaError(RuntimeError):
    """Raised when a loaded file does not match its expected shape."""


def _check(df: pd.DataFrame, spec: TableSpec, *, strict_grain: bool) -> None:
    """Validate a loaded dataframe against its spec."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{spec.filename}: missing expected column(s) {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # Nulls in a column we did not expect to be nullable mean either a damaged
    # file or a wrong assumption on our part. Both are worth stopping for.
    unexpected_nulls = {
        c: int(df[c].isna().sum())
        for c in spec.columns
        if c not in spec.nullable and df[c].isna().any()
    }
    if unexpected_nulls:
        raise SchemaError(
            f"{spec.filename}: unexpected nulls in {unexpected_nulls}. "
            "Either the file is damaged or schema.py needs updating."
        )

    if strict_grain:
        dupes = int(df.duplicated(subset=list(spec.grain)).sum())
        if dupes:
            raise SchemaError(
                f"{spec.filename}: {dupes:,} rows duplicate the declared grain "
                f"{spec.grain}. Joining on this table would multiply rows."
            )


def normalise_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Correct known data-entry inconsistencies in the published files.

    Currently one: ``imd_band`` contains ``"10-20"`` where every other category
    carries a percent sign. See ``schema.VALUE_FIXES`` for why this is handled
    centrally rather than left to whoever happens to touch the column.
    """
    fixes = VALUE_FIXES.get(name)
    if not fixes:
        return df

    for column, mapping in fixes.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
    return df


def load_table(
    name: str,
    *,
    data_dir: Path | None = None,
    strict_grain: bool = True,
) -> pd.DataFrame:
    """Load one OULAD table by its logical name (e.g. ``"studentInfo"``).

    Parameters
    ----------
    name
        Key into ``schema.TABLES``.
    data_dir
        Directory holding the CSVs. Defaults to ``data/raw``. Point it at
        ``data/synthetic`` to run the same code against generated data.
    strict_grain
        If True, raise when rows duplicate the table's declared grain. Left on
        by default; ``studentVle`` is the one table where the published file has
        a small number of exact-grain repeats, so it is loaded with this off.
    """
    if name not in TABLES:
        raise KeyError(f"Unknown table {name!r}. Known: {sorted(TABLES)}")

    directory = Path(data_dir) if data_dir is not None else paths.RAW_DIR
    spec = TABLES[name]

    # Prefer parquet when it is there. It loads far faster than CSV and is
    # small enough to live in git, which the raw studentVle.csv is not --
    # see scripts/prepare_data.py.
    parquet_path = directory / f"{Path(spec.filename).stem}.parquet"
    csv_path = directory / spec.filename

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Found neither {parquet_path.name} nor {csv_path.name} in {directory}.\n"
            "See README.md > Getting the data."
        )

    df = normalise_values(df, name)
    _check(df, spec, strict_grain=strict_grain)
    return df


def load_all(
    *, data_dir: Path | None = None, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """Load all seven tables into a dict keyed by logical name."""
    out: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        # studentVle is the known exception to strict grain checking.
        strict = name != "studentVle"
        out[name] = load_table(name, data_dir=data_dir, strict_grain=strict)
        if verbose:
            df = out[name]
            print(f"  {name:22s} {len(df):>10,} rows x {df.shape[1]:>2d} cols")
    return out


def data_available(data_dir: Path | None = None) -> bool:
    """True if all seven tables are present in ``data_dir``, as CSV or parquet."""
    directory = Path(data_dir) if data_dir is not None else paths.RAW_DIR
    return all(
        (directory / s.filename).exists()
        or (directory / f"{Path(s.filename).stem}.parquet").exists()
        for s in TABLES.values()
    )
