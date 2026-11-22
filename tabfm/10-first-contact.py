"""First contact: run a tabular foundation model and watch it behave strangely.

    python tabfm/10-first-contact.py

Uses a small built-in dataset so it runs in seconds and needs no OULAD data.
The point is not the accuracy number -- it is to see, concretely, that `fit()`
does nothing and `predict()` does everything, which is backwards from every
other model you have used.

Read the printed commentary as you go. If the foundation models cannot load
their weights, the script says exactly why and still runs the baselines, so you
can see the shape of the comparison before you sort out access.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabfm_bench import CPU_LIMITS, _explain  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    header("1. A SMALL DATASET")
    print("""
The breast-cancer dataset that ships with scikit-learn: 569 rows, 30 numeric
columns, binary outcome. It is small, clean and entirely numeric -- which is
deliberately the easiest possible case for a tabular foundation model.

Small matters. Foundation models are claimed to win in the few-hundred-to-few-
thousand-row regime, where there is not enough data to fit a big model well but
the problem still has structure. That is this dataset.
""")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    base_rate = y_test.mean()
    print(f"  train      : {X_train.shape[0]:,} rows x {X_train.shape[1]} columns")
    print(f"  test       : {X_test.shape[0]:,} rows")
    print(f"  base rate  : {base_rate:.3f}  (the PR-AUC floor)")

    results = []

    header("2. THE BASELINES")
    print("""
Two ordinary models, both of which have to be *trained* on these 341 rows.

Watch the fit times. They are small but non-zero, and they are where the work
happens: the model is adjusting its own parameters until it predicts the
training data well.
""")

    for name, model in [
        ("logistic regression",
         Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(max_iter=3000))])),
        ("gradient boosting",
         HistGradientBoostingClassifier(random_state=0)),
    ]:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = model.predict_proba(X_test)[:, 1]
        t_pred = time.perf_counter() - t0
        results.append((name, average_precision_score(y_test, p),
                        roc_auc_score(y_test, p), t_fit, t_pred, ""))
        print(f"  {name:24s} fit {t_fit:6.3f}s   predict {t_pred:6.3f}s")

    header("3. THE FOUNDATION MODELS")
    print("""
Now the same data through TabPFN and TabICL.

Three things are different, and all three are deliberate:

  * No scaling. No one-hot encoding. No imputation. The raw array goes in.
    The documentation is explicit that preprocessing it yourself makes results
    WORSE, because the model was pretrained on particular distributions and
    expects to do its own normalisation.

  * No hyperparameters. There is nothing to tune, so there is no tuning step
    and no validation split spent on one.

  * Nothing is being learned from your data in the usual sense. The weights
    were fixed before you ran this. Your training rows are CONTEXT -- closer to
    a prompt than to a training set.

Watch where the time goes compared to section 2.
""")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          f" | using device={device}")
    if device == "cpu":
        print(f"  NOTE: on CPU the training-set cap is {CPU_LIMITS.get('v3.5', 5000):,} "
              f"rows. This dataset has {len(X_train):,}, so we are fine.")

    for name, loader in [("TabPFN", _load_tabpfn), ("TabICL", _load_tabicl)]:
        model, why = loader(device)
        if model is None:
            print(f"\n  {name:8s} UNAVAILABLE -- {why}")
            results.append((name, np.nan, np.nan, np.nan, np.nan, why))
            continue
        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = model.predict_proba(X_test)[:, 1]
            t_pred = time.perf_counter() - t0
            results.append((name, average_precision_score(y_test, p),
                            roc_auc_score(y_test, p), t_fit, t_pred, f"device={device}"))
            print(f"\n  {name:8s} fit {t_fit:6.3f}s   predict {t_pred:6.3f}s")
            ratio = t_pred / max(t_fit, 1e-6)
            print(f"           predict took {ratio:.0f}x as long as fit")
        except Exception as exc:
            why = _explain(exc)
            print(f"\n  {name:8s} FAILED -- {why}")
            results.append((name, np.nan, np.nan, np.nan, np.nan, why))

    header("4. THE COMPARISON")
    print(f"\n  {'model':24s} {'PR-AUC':>8s} {'lift':>6s} {'ROC':>7s} "
          f"{'fit s':>8s} {'pred s':>8s}  note")
    print("  " + "-" * 88)
    for name, ap, roc, tf, tp, note in results:
        if not np.isfinite(ap):
            print(f"  {name:24s} {'--':>8s} {'--':>6s} {'--':>7s} {'--':>8s} "
                  f"{'--':>8s}  {note}")
        else:
            print(f"  {name:24s} {ap:>8.4f} {ap/base_rate:>6.2f} {roc:>7.4f} "
                  f"{tf:>8.3f} {tp:>8.3f}  {note}")

    header("WHAT TO TAKE FROM THIS")
    print("""
  1. The fit/predict split is inverted. An ordinary model spends its time in
     fit() and predicts almost for free. A foundation model stores your data in
     fit() and does all the work in predict(). If you plan to score one row at
     a time in production, that is the number that will hurt.

  2. There were no hyperparameters to choose. For the baselines you could spend
     a day tuning. Here there is nothing to turn.

  3. On a dataset this small and clean, the comparison is close and the
     foundation model needed no tuning to get there. That is the actual claim
     being made for these models -- not that they are unbeatable, but that they
     get most of the way with zero effort.

  Next: 20-what-fit-actually-does.py, which takes the inversion in point 1
  apart, then 30-oulad.py for a harder, messier, more realistic dataset.
""")


def _load_tabpfn(device):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None, "not installed (pip install tabpfn)"
    try:
        return TabPFNClassifier(device=device), ""
    except Exception as exc:
        return None, _explain(exc)


def _load_tabicl(device):
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        return None, "not installed (pip install tabicl)"
    try:
        return TabICLClassifier(device=device), ""
    except Exception as exc:
        return None, _explain(exc)


if __name__ == "__main__":
    main()
