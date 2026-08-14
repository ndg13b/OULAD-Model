"""Phase 1: load the data, check it, build the label, and look at what we have.

Run this before writing a single feature. It answers the questions that
determine everything downstream:

  1. Did all seven files load, and do they have the shape we expect?
  2. What is the grain of each table, and how much is missing?
  3. What does the outcome look like?
  4. **How many students have already left before each candidate cutoff day?**

Question 4 is the one that shapes the project. See docs/02-leakage.md.

Usage:
    python scripts/phase1_inspect.py                      # uses data/raw
    python scripts/phase1_inspect.py --data-dir data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from oulad import labels as lab
from oulad import load
from oulad.schema import TABLES

CANDIDATE_CUTOFFS = [7, 14, 28, 56, 84]

RULE = "=" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def section_1_load(data_dir: Path) -> dict[str, pd.DataFrame]:
    header("1. LOADING AND SCHEMA CHECK")
    print("Each file is validated against src/oulad/schema.py as it loads:")
    print("expected columns present, no nulls where we did not expect them,")
    print("and no rows duplicating the table's declared grain.\n")
    tables = load.load_all(data_dir=data_dir)
    print("\nAll seven tables loaded and passed their checks.")
    return tables


def section_2_grain(tables: dict[str, pd.DataFrame]) -> None:
    header("2. GRAIN AND MISSINGNESS")
    print("'Grain' = what one row means. Getting this wrong when joining is the")
    print("single most common way to silently corrupt a dataset: join two tables")
    print("with mismatched grain and rows multiply without any error.\n")

    for name, df in tables.items():
        spec = TABLES[name]
        print(f"--- {name} ({len(df):,} rows) ---")
        print(f"  grain: one row per {' x '.join(spec.grain)}")
        null_pct = (df.isna().mean() * 100).round(2)
        nulls = null_pct[null_pct > 0]
        if len(nulls):
            for col, pct in nulls.items():
                print(f"  missing: {col:24s} {pct:6.2f}%")
        else:
            print("  missing: none")
        print()


def section_3_outcome(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    header("3. THE OUTCOME AND THE LABEL")

    si = tables["studentInfo"]
    print("final_result, as recorded:\n")
    counts = si["final_result"].value_counts()
    pct = si["final_result"].value_counts(normalize=True) * 100
    for k in counts.index:
        print(f"  {k:14s} {counts[k]:>8,}  {pct[k]:5.1f}%")

    print("\nCollapsing to a binary label:")
    print("  at_risk = 1  <-  Fail, Withdrawn")
    print("  at_risk = 0  <-  Pass, Distinction")

    labels = lab.build_labels(si, tables["studentRegistration"])

    base = labels["at_risk"].mean()
    print(f"\n  positives: {int(labels['at_risk'].sum()):,} of {len(labels):,}")
    print(f"  base rate: {base:.3f}  ({base * 100:.1f}% of all enrolments)")
    print()
    print("  The base rate is the reference point for every model score that")
    print("  follows. A model that always predicts 'at risk' is right this")
    print("  often, so any real model has to beat it to be worth anything.")

    n_students = labels["id_student"].nunique()
    print(f"\n  distinct students: {n_students:,}")
    print(f"  enrolments:        {len(labels):,}")
    repeats = len(labels) - n_students
    print(f"  -> {repeats:,} enrolments belong to a student who appears more than once.")
    print()
    print("  This is why rows cannot be split randomly into train and test sets.")
    print("  The same person landing on both sides lets the model recognise the")
    print("  individual rather than learn the pattern, and the test score stops")
    print("  measuring what we want it to. Splits must group on id_student.")

    return labels


def section_4_cutoffs(labels: pd.DataFrame) -> pd.DataFrame:
    header("4. WHO IS STILL THERE? (the decision that shapes the project)")
    print("Every date in OULAD is a day offset from the start of the course.")
    print("If we score students on day 28, anyone who unregistered on or before")
    print("day 28 has already gone. Predicting their outcome is not prediction:")
    print("the answer is already recorded.\n")

    withdrawn = labels[labels["withdrew"] == 1]
    if len(withdrawn):
        q = withdrawn["date_unregistration"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print("When withdrawals happen (day relative to course start):")
        for p, v in q.items():
            print(f"  {int(p * 100):>3d}th percentile   day {v:7.0f}")
        pre_start = int((withdrawn["date_unregistration"] < 0).sum())
        print(f"\n  {pre_start:,} of {len(withdrawn):,} withdrawals "
              f"({100 * pre_start / len(withdrawn):.1f}%) happen before day 0 --")
        print("  students who registered and left before teaching even began.\n")

    summary = lab.cutoff_population_summary(labels, CANDIDATE_CUTOFFS)

    print("Population and base rate at each candidate cutoff:\n")
    print(f"  {'cutoff':>7} {'still enrolled':>15} {'% of all':>9} "
          f"{'already left':>13} {'base rate':>10}")
    for r in summary.itertuples(index=False):
        print(f"  day {r.cutoff_day:>3} {r.n_eligible:>15,} {r.pct_of_all:>8.1f}% "
              f"{r.n_already_left:>13,} {r.base_rate_at_risk:>10.3f}")

    print()
    print("Read this table carefully -- it is the core tension of the project.")
    print()
    print("  Moving the cutoff later gives more behavioural data per student,")
    print("  which should make prediction easier. But two things push the other")
    print("  way: the students who most needed help have already left (you")
    print("  cannot intervene on someone who is gone), and the ones who remain")
    print("  are the harder cases, so the base rate falls and the problem gets")
    print("  genuinely more difficult.")
    print()
    print("  If instead we scored *everyone* at day 28, including those who left")
    print("  on day 3, the model would look far better -- and the improvement")
    print("  would be entirely fake. Those students have no clicks after day 3,")
    print("  so 'no recent activity' predicts them perfectly. That rule would")
    print("  dominate the model while telling a tutor nothing a database query")
    print("  could not.")

    return summary


def section_5_activity(tables: dict[str, pd.DataFrame], labels: pd.DataFrame) -> None:
    header("5. IS THERE SIGNAL IN THE CLICKSTREAM? (a first look)")
    print("Before building features, check that the behavioural data separates")
    print("outcomes at all. If it does not, no amount of modelling will help.\n")

    vle = tables["studentVle"]
    cutoff = 28
    early = vle[vle["date"] <= cutoff]

    agg = (
        early.groupby(lab.KEY)
        .agg(total_clicks=("sum_click", "sum"), active_days=("date", "nunique"))
        .reset_index()
    )

    eligible = labels.loc[lab.eligible_at_cutoff(labels, cutoff)]
    merged = eligible.merge(agg, on=lab.KEY, how="left").fillna(
        {"total_clicks": 0, "active_days": 0}
    )

    print(f"Among students still enrolled at day {cutoff} "
          f"(n = {len(merged):,}), using only clicks up to day {cutoff}:\n")
    grouped = merged.groupby("at_risk")[["total_clicks", "active_days"]].median()
    print(f"  {'':16s} {'median clicks':>14} {'median active days':>20}")
    for at_risk, row in grouped.iterrows():
        name = "at risk (1)" if at_risk == 1 else "not at risk (0)"
        print(f"  {name:16s} {row['total_clicks']:>14,.0f} "
              f"{row['active_days']:>20,.0f}")

    zero = merged.groupby("at_risk").apply(
        lambda g: (g["total_clicks"] == 0).mean() * 100, include_groups=False
    )
    print(f"\n  Students with zero clicks in the first {cutoff} days:")
    for at_risk, pct in zero.items():
        name = "at risk (1)" if at_risk == 1 else "not at risk (0)"
        print(f"    {name:16s} {pct:5.1f}%")

    print()
    print("  A gap here means the clickstream carries information about the")
    print("  outcome. It does not tell us how much, or whether it survives once")
    print("  we account for everything else -- that is what the models are for.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory holding the seven CSVs. Defaults to data/raw.",
    )
    args = ap.parse_args()

    data_dir = args.data_dir
    if data_dir is not None and "synthetic" in str(data_dir):
        print("\n" + "!" * 78)
        print("RUNNING ON SYNTHETIC DATA. Every number below is made up.")
        print("Useful for checking the pipeline runs; not a finding about anyone.")
        print("!" * 78)

    if not load.data_available(data_dir):
        raise SystemExit(
            f"\nThe seven CSVs were not found in "
            f"{data_dir or 'data/raw'}.\nSee README.md > Getting the data."
        )

    tables = section_1_load(data_dir)
    section_2_grain(tables)
    labels = section_3_outcome(tables)
    section_4_cutoffs(labels)
    section_5_activity(tables, labels)

    header("WHAT PHASE 1 SETTLED")
    print("  - The seven tables load and match their expected shape.")
    print("  - The label is built from final_result; the base rate is recorded.")
    print("  - Students repeat across presentations, so splits must group on")
    print("    id_student.")
    print("  - The prediction population at a cutoff is students still enrolled")
    print("    at that cutoff, not everyone.")
    print("\nNext: Phase 2, build_features(cutoff_day). Nothing in that function")
    print("may look at any row dated after the cutoff.\n")


if __name__ == "__main__":
    main()
