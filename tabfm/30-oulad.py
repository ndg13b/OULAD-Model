"""The real test: foundation models against a known baseline on real, messy data.

    python tabfm/30-oulad.py
    python tabfm/30-oulad.py --n-train 2000 --sweep

The breast-cancer dataset in 10-first-contact.py is the easy case: small, clean,
entirely numeric, well separated. OULAD at day 28 is not. It has 45 features,
missing values that mean three different things, a free-text categorical, wildly
different scales, heavy skew, and classes that genuinely overlap.

It also has something a benchmark dataset does not: a baseline we built
ourselves and understand. Gradient boosting on the full 27,538 rows scores
PR-AUC 0.766 against a 0.441 floor (docs/06-phase2-features.md). So "did the
foundation model win?" has a real answer here rather than a leaderboard one.

Two honest caveats, both printed by the script:

  * On CPU, TabPFN caps the training set at 5,000 rows. The full baseline used
    ~19,000. So the comparison here is at a size the baseline is handicapped at
    too -- which is the point of --sweep.
  * The split is grouped on id_student, as everything in this project is. A
    random split would let the same person appear on both sides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabfm_bench as T  # noqa: E402

RULE = "=" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=4000,
                    help="Context/training rows. Keep <= 5000 on CPU.")
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--cutoff", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="Also run a training-set-size sweep.")
    args = ap.parse_args()

    header("1. THE DATA")
    print("""
OULAD features at the day-28 cutoff: one row per student still enrolled four
weeks in, 45 columns, label is "will fail or withdraw".

This is what real tabular data looks like, and every one of these properties is
something a foundation model has to cope with without being told:
""")

    data = T.sample_oulad(
        n_train=args.n_train, n_test=args.n_test,
        cutoff_day=args.cutoff, seed=args.seed,
    )
    X = data.X_train
    n_missing_cols = int((X.isna().mean() > 0).sum())
    print(f"  rows (context)        : {len(X):,}")
    print(f"  rows (to predict)     : {len(data.X_test):,}")
    print(f"  features              : {X.shape[1]}")
    print(f"  columns with missing  : {n_missing_cols} of {X.shape[1]}")
    print(f"  categorical columns   : {len(data.categorical_indices)} "
          f"(index {data.categorical_indices})")
    print(f"  base rate             : {data.base_rate:.3f}")
    num = X.select_dtypes("number")
    print(f"  value range across cols: {num.min().min():,.1f} to {num.max().max():,.0f}")
    print("""
Note the scale range. One column is a fraction near zero, another is a click
count in the thousands. A linear model needs those standardised or it will be
dominated by whichever column happens to have big numbers. A foundation model
is supposed to handle it raw -- we pass exactly the same frame to both and let
each do its own preprocessing.
""")

    header("2. THE COMPARISON")
    print(f"""
Same split, same rows, same metric for every model.

PR-AUC's floor is the base rate ({data.base_rate:.3f}), so the raw number is not
interpretable alone -- `lift` divides by it. A lift of 1.00 is worthless, and
the full-data gradient boosting baseline from Phase 2 sits at 1.74.
""")

    results = [
        T.run_majority(data),
        T.run_logistic(data),
        T.run_gbm(data),
        T.run_tabpfn(data),
        T.run_tabicl(data),
    ]
    T.print_report(results, data)

    available = [r for r in results if np.isfinite(r.pr_auc)]
    unavailable = [r for r in results if not np.isfinite(r.pr_auc)]

    if unavailable:
        print("\n  Not run:")
        for r in unavailable:
            print(f"    {r.name:10s} {r.note}")
        print("\n  See tabfm/01-setup.md for how to fix each of these.")

    if len(available) > 1:
        best = max(available, key=lambda r: r.pr_auc)
        print(f"\n  Best here: {best.name} at PR-AUC {best.pr_auc:.4f} "
              f"(lift {best.pr_auc / data.base_rate:.2f})")
        print(f"  Phase 2 full-data gradient boosting, for reference: 0.766 (lift 1.74)")
        print("""
  If the score here is below that, the likely reason is the row cap rather than
  the model: this run used a fraction of the training data the Phase 2 number
  had. --sweep separates those two explanations.""")

    if args.sweep:
        header("3. DOES THE ADVANTAGE DEPEND ON SIZE?")
        print("""
The claim for foundation models is specifically about the small-data regime:
with a few hundred rows they win, and the advantage narrows as data grows,
because gradient boosting gets better with more data and they largely do not.

This sweep tests that directly. It holds the test set fixed and varies only the
number of context rows. Watch whether the gap closes.
""")
        sizes = [250, 500, 1000, 2000, 4000]
        print(f"  {'n_train':>8s} {'logistic':>10s} {'gbm':>10s} "
              f"{'TabPFN':>10s} {'TabICL':>10s}")
        print("  " + "-" * 54)
        for n in sizes:
            d = T.sample_oulad(n_train=n, n_test=args.n_test,
                               cutoff_day=args.cutoff, seed=args.seed)
            row = [T.run_logistic(d), T.run_gbm(d), T.run_tabpfn(d), T.run_tabicl(d)]
            cells = "".join(
                f"{r.pr_auc:>10.4f}" if np.isfinite(r.pr_auc) else f"{'--':>10s}"
                for r in row
            )
            print(f"  {n:>8,}{cells}")
        print("""
  Read the columns, not the rows. A model whose column barely moves is one that
  does not benefit from more data; a model whose column climbs steadily is one
  you should give everything you have.""")

    header("WHAT THIS TELLS YOU")
    print("""
  The question the book asks -- whether to bet on these models -- does not have
  a general answer. It has an answer for YOUR data, and this is how you get it:
  a fixed grouped split, a baseline you understand, and the same metric for
  everything.

  Things worth trying from here:

    --n-train 1000    does the gap widen when data is scarce? (the regime the
                      claim is actually about)
    --cutoff 7        a much harder problem: one week of behaviour
    --cutoff 84       easier, more data per student
    --seed 1 2 3      how much of any difference is just the split?

  That last one matters most. A 0.01 PR-AUC difference on one split is noise.
  Run three seeds before believing any ranking.
""")


if __name__ == "__main__":
    main()
