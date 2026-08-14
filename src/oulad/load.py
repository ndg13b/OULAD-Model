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
from .schema import TABLES, TableSpec


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
    path = directory / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "The raw CSVs are not in the repo (they are gitignored). "
            "See README.md > Getting the data."
        )

    df = pd.read_csv(path)
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
    """True if all seven CSVs are present in ``data_dir`` (default data/raw)."""
    directory = Path(data_dir) if data_dir is not None else paths.RAW_DIR
    return all((directory / s.filename).exists() for s in TABLES.values())
