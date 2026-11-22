"""A like-for-like comparison harness: tabular foundation models vs. the usual suspects.

The point of this module is that the comparison is *fair*. It is easy to make a
foundation model look good or bad by accident -- give it a different train/test
split from the baselines, scale the features for one and not the other, or let
the same student appear on both sides. Everything here runs on identical splits
with identical data.

Three things it handles that are specific to this project:

**Grouped splitting.** OULAD has 3,808 enrolments belonging to students who
appear more than once. A random split lets the model recognise the individual
rather than learn the pattern. Every split here groups on ``id_student``. See
``docs/02-leakage.md``.

**The CPU row cap.** TabPFN refuses to run above a hard row limit on CPU --
5,000 for TabPFN-3 and 3.5, 1,000 for the 2.x models. OULAD at day 28 has
27,538 rows, so a subsample is mandatory, and the subsample has to respect the
grouping. ``sample_oulad`` does both.

**Preprocessing asymmetry.** Foundation models want *raw* data: no scaling, no
one-hot encoding, NaN left alone. The TabPFN docs say so explicitly. Baselines
need the opposite. So each model family gets the preprocessing it wants, and
the comparison is on the output, not the input.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

# The hard ceilings TabPFN enforces when no GPU is present. Read from the
# package itself rather than hardcoded, so this does not silently go stale.
try:
    from tabpfn.constants import ModelVersion
    from tabpfn.inference_config import cpu_sample_limit

    CPU_LIMITS = {m.value: cpu_sample_limit(m) for m in ModelVersion}
except Exception:  # tabpfn not installed
    CPU_LIMITS = {"v3.5": 5000, "v2": 1000}


@dataclass
class Result:
    """One model's score on one split."""

    name: str
    pr_auc: float
    roc_auc: float
    fit_seconds: float
    predict_seconds: float
    note: str = ""

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.predict_seconds


@dataclass
class Dataset:
    """A train/test split, carried in both raw and encoded form."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    categorical_indices: list[int] = field(default_factory=list)

    @property
    def base_rate(self) -> float:
        """The share of positives in the test set -- the PR-AUC floor."""
        return float(self.y_test.mean())

    def describe(self) -> str:
        return (
            f"train {len(self.X_train):,} x {self.X_train.shape[1]} | "
            f"test {len(self.X_test):,} | base rate {self.base_rate:.3f}"
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sample_oulad(
    n_train: int = 4000,
    n_test: int = 4000,
    cutoff_day: int = 28,
    seed: int = 0,
    data_dir: Path | None = None,
) -> Dataset:
    """A grouped, size-capped sample of the OULAD feature set.

    Parameters
    ----------
    n_train
        Rows in the context/training set. Keep at or below 5,000 if you are on
        CPU -- TabPFN refuses above that. This is the number that matters.
    n_test
        Rows to predict. Not capped by TabPFN, but prediction is the expensive
        call for a foundation model, so it drives the runtime.
    cutoff_day
        Passed to ``build_features``. Day 28 is the project's primary cutoff.

    The split is made with ``GroupShuffleSplit`` on ``id_student`` *first*, and
    only then subsampled, so no student can appear on both sides.
    """
    from oulad import features, load

    # build_labels prints a data-quirk note on every call; in a sweep that is
    # the same line five times. Say it once, via the caller, not here.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        tables = load.load_all(data_dir=data_dir, verbose=False)
        frame = features.build_features(tables, cutoff_day)

    cols = features.feature_columns(frame)
    X = frame[cols].copy()
    y = frame["at_risk"].to_numpy()
    groups = frame["id_student"].to_numpy()

    # Split by student before any subsampling.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    rng = np.random.default_rng(seed)
    if len(train_idx) > n_train:
        train_idx = rng.choice(train_idx, size=n_train, replace=False)
    if len(test_idx) > n_test:
        test_idx = rng.choice(test_idx, size=n_test, replace=False)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    # `region` is the only free-text categorical left. Foundation models take
    # categoricals directly, told which columns they are; they must not be
    # one-hot encoded first.
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "str", "category")]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y[train_idx],
        y_test=y[test_idx],
        categorical_indices=cat_idx,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    cat = [c for c in X.columns if str(X[c].dtype) in ("object", "str", "category")]
    num = [c for c in X.columns if c not in cat]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ]
    )


def _score(name, model, data: Dataset, note: str = "") -> Result:
    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = model.predict_proba(data.X_test)[:, 1]
    t_pred = time.perf_counter() - t0

    return Result(
        name=name,
        pr_auc=average_precision_score(data.y_test, proba),
        roc_auc=roc_auc_score(data.y_test, proba),
        fit_seconds=t_fit,
        predict_seconds=t_pred,
        note=note,
    )


def run_majority(data: Dataset) -> Result:
    """The floor. Predicts the same score for everyone, so it cannot rank."""
    return Result(
        name="majority baseline",
        pr_auc=data.base_rate,
        roc_auc=0.5,
        fit_seconds=0.0,
        predict_seconds=0.0,
        note="scores every student identically",
    )


def run_logistic(data: Dataset) -> Result:
    model = Pipeline(
        [
            ("pre", _preprocessor(data.X_train, scale=True)),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )
    return _score("logistic regression", model, data, "scaled, one-hot, imputed")


def run_gbm(data: Dataset, **kwargs) -> Result:
    """Gradient boosting -- the incumbent champion on tabular data."""
    model = Pipeline(
        [
            ("pre", _preprocessor(data.X_train, scale=False)),
            ("model", HistGradientBoostingClassifier(random_state=0, **kwargs)),
        ]
    )
    label = "gradient boosting" + (" (tuned)" if kwargs else " (defaults)")
    return _score(label, model, data, "handles NaN natively")


# ---------------------------------------------------------------------------
# Foundation models
# ---------------------------------------------------------------------------

def run_tabpfn(data: Dataset, device: str = "auto", **kwargs) -> Result:
    """TabPFN, with the failure modes reported rather than raised.

    Raw data goes in: no scaling, no one-hot encoding, NaN left as NaN. The
    documentation is explicit that preprocessing it yourself makes it worse,
    because the model expects the distributions it was pretrained on.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return _unavailable("TabPFN", "package not installed (pip install tabpfn)")

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    limit = CPU_LIMITS.get("v3.5", 5000)
    note = f"device={device}"
    if device == "cpu" and len(data.X_train) > limit:
        return _unavailable(
            "TabPFN",
            f"{len(data.X_train):,} training rows exceeds the CPU cap of "
            f"{limit:,}; subsample or use a GPU",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = TabPFNClassifier(
                device=device,
                categorical_features_indices=data.categorical_indices or None,
                **kwargs,
            )
            return _score("TabPFN", model, data, note)
    except Exception as exc:
        return _unavailable("TabPFN", _explain(exc))


def run_tabicl(data: Dataset, device: str = "auto", **kwargs) -> Result:
    """TabICL. Same contract as ``run_tabpfn``."""
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        return _unavailable("TabICL", "package not installed (pip install tabicl)")

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = TabICLClassifier(device=device, **kwargs)
            return _score("TabICL", model, data, f"device={device}")
    except Exception as exc:
        return _unavailable("TabICL", _explain(exc))


def _unavailable(name: str, why: str) -> Result:
    return Result(name, np.nan, np.nan, np.nan, np.nan, note=why)


def _explain(exc: Exception) -> str:
    """Turn the common failures into something actionable."""
    text = str(exc)
    kind = type(exc).__name__

    if "GatedRepo" in kind or "gated" in text.lower():
        return (
            "weights are gated: accept the terms on the model's Hugging Face "
            "page, then set HF_TOKEN"
        )
    if "Proxy" in kind or "no rule or network setting allows this host" in text:
        return "huggingface.co is blocked by this environment's network policy"
    if "401" in text or "authentication" in text.lower():
        return "Hugging Face authentication failed: set a valid HF_TOKEN"
    if "out of memory" in text.lower():
        return "out of memory: reduce n_train or n_test"
    return f"{kind}: {text[:120]}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: list[Result], data: Dataset) -> pd.DataFrame:
    """A comparison table, with lift over the base rate.

    Lift is reported because PR-AUC's floor *is* the base rate, so the raw
    number is not interpretable on its own. A model scoring 0.60 on a 0.55 base
    rate is doing almost nothing; the same score on a 0.10 base rate is
    excellent.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "PR-AUC": r.pr_auc,
                "lift": r.pr_auc / data.base_rate if np.isfinite(r.pr_auc) else np.nan,
                "ROC-AUC": r.roc_auc,
                "fit s": r.fit_seconds,
                "predict s": r.predict_seconds,
                "note": r.note,
            }
        )
    return pd.DataFrame(rows)


def print_report(results: list[Result], data: Dataset) -> None:
    print(f"\n  {data.describe()}")
    print(f"  PR-AUC floor is the base rate: {data.base_rate:.3f}\n")
    print(f"  {'model':30s} {'PR-AUC':>8s} {'lift':>6s} {'ROC':>7s} "
          f"{'fit s':>8s} {'pred s':>8s}  note")
    print("  " + "-" * 94)
    for r in results:
        if not np.isfinite(r.pr_auc):
            print(f"  {r.name:30s} {'--':>8s} {'--':>6s} {'--':>7s} "
                  f"{'--':>8s} {'--':>8s}  {r.note}")
        else:
            lift = r.pr_auc / data.base_rate
            print(f"  {r.name:30s} {r.pr_auc:>8.4f} {lift:>6.2f} {r.roc_auc:>7.4f} "
                  f"{r.fit_seconds:>8.2f} {r.predict_seconds:>8.2f}  {r.note}")
