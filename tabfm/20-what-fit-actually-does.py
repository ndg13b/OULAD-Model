"""Taking the inversion apart: where the time goes, what the context window is, what breaks.

    python tabfm/20-what-fit-actually-does.py
    python tabfm/20-what-fit-actually-does.py --quick
    python tabfm/20-what-fit-actually-does.py --max-context 0   # skip the big-context section

`10-first-contact.py` ended on an observation: `fit()` is nearly free and
`predict()` does all the work. That is the headline, and on its own it is not
actionable. This script measures the shape of it, because the shape is what
decides whether you can deploy one of these models.

Three questions, one section each:

  1. WHERE THE TIME GOES. `predict()` is expensive -- but expensive in what? The
     number of rows you are scoring, or the number you handed to `fit()`? The
     answer determines whether batching helps you.

  2. WHAT THE CONTEXT WINDOW IS. These models have a hard ceiling on context
     rows, not a soft degradation. Where is it, what sets it, and what happens
     at the edge.

  3. WHAT BREAKS. Three ways to make a foundation model slow or wrong that all
     look like good practice if you come from ordinary ML.

Everything is measured on this machine, in this process. No numbers are quoted
from documentation unless labelled as such. Expect roughly 6-7 minutes on a
4-core CPU, or 3-4 with --quick. Section 2 is the memory-hungry part and peaks
around 8 GB; --max-context 0 skips it.

Uses the OULAD features, so it needs `data/raw/`. If the foundation model
cannot load its weights the script says why and skips the sections that need
it, rather than dying.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabfm_bench as T  # noqa: E402
from tabfm_bench import CPU_LIMITS, _explain  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def tabicl(device: str = "cpu", **kwargs):
    """Return (classifier, "") or (None, reason)."""
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        return None, "tabicl not installed (pip install tabicl)"
    try:
        return TabICLClassifier(device=device, **kwargs), ""
    except Exception as exc:
        return None, _explain(exc)


def timed_fit_predict(model, d, n_test=None):
    """Fit on d's context, predict on (a prefix of) its test set. Returns seconds."""
    X_test = d.X_test if n_test is None else d.X_test.iloc[:n_test]
    t0 = time.perf_counter()
    model.fit(d.X_train, d.y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    t_pred = time.perf_counter() - t0
    return t_fit, t_pred, proba


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-test", type=int, default=500,
                    help="Rows to score in section 1a (held fixed there).")
    ap.add_argument("--cutoff", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-context", type=int, default=5500,
                    help="Context size for section 2's above-the-cap run. "
                         "0 skips it; it is the most memory-hungry step.")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller sizes throughout. Same lessons, ~2 min.")
    args = ap.parse_args()

    if args.quick:
        contexts = [250, 1000]
        test_sizes = [100, 250, 500]
        n_test_1a = min(args.n_test, 250)
        chunk_ctx, chunk_test, chunk = 250, 200, 25
        pre_ctx = pre_test = 500
        if args.max_context:
            args.max_context = min(args.max_context, 5100)
    else:
        contexts = [250, 1000, 4000]
        test_sizes = [250, 1000, 2000]
        n_test_1a = args.n_test
        chunk_ctx, chunk_test, chunk = 500, 500, 50
        pre_ctx = pre_test = 1000

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"torch {torch.__version__} | CUDA available: "
              f"{torch.cuda.is_available()} | device={device}")
    except ImportError:
        print("torch not importable; assuming CPU")

    probe, why = tabicl(device)
    if probe is None:
        print(f"\nTabICL unavailable -- {why}")
        print("Sections 1-3 all need it. See tabfm/01-setup.md.")
        return

    # ---------------------------------------------------------------- 1a
    header("1a. DOES predict() COST SCALE WITH THE CONTEXT?")
    print(f"""
Hold the thing being scored CONSTANT at {n_test_1a:,} rows. Vary only how many
rows were handed to fit(). If predict() were an ordinary lookup, this table
would be flat: the same {n_test_1a:,} students, the same work.
""")

    rows_1a = []
    for n in contexts:
        d = T.sample_oulad(n_train=n, n_test=max(n_test_1a, 1),
                           cutoff_day=args.cutoff, seed=args.seed)
        model, why = tabicl(device)
        if model is None:
            print(f"  context {n:>6,}: unavailable -- {why}")
            continue
        try:
            t_fit, t_pred, _ = timed_fit_predict(model, d, n_test_1a)
        except Exception as exc:
            print(f"  context {n:>6,}: FAILED -- {_explain(exc)}")
            continue
        rows_1a.append((n, t_fit, t_pred))
        print(f"  context {n:>6,} rows | fit {t_fit:6.2f}s | "
              f"predict {t_pred:7.2f}s | {t_pred / max(t_fit, 1e-9):>7.0f}x fit",
              flush=True)

    if len(rows_1a) >= 2:
        (n_lo, _, p_lo), (n_hi, _, p_hi) = rows_1a[0], rows_1a[-1]
        fits = [f for _, f, _ in rows_1a]
        print(f"""
  It is not flat. Scoring the same {n_test_1a:,} rows cost {p_lo:.1f}s with {n_lo:,}
  context rows and {p_hi:.1f}s with {n_hi:,} -- {p_hi / max(p_lo, 1e-9):.1f}x, for identical output.

  Meanwhile fit() stayed between {min(fits):.2f}s and {max(fits):.2f}s: flat, and
  unrelated to how much data it was given.

  That is the whole inversion in one table. Your training rows are not compiled
  into parameters during fit(); they are kept, and re-read on every predict()
  call. They behave like a prompt, and a longer prompt costs more every time you
  use it -- not once.""")

    # ---------------------------------------------------------------- 1b
    header("1b. ... AND WITH THE NUMBER OF ROWS YOU SCORE?")
    print(f"""
Now the other axis. ONE model, fitted once on {pre_ctx:,} context rows, asked to
score different numbers of rows. Because it is the same fitted model throughout,
any difference here is pure prediction cost.
""")

    d = T.sample_oulad(n_train=pre_ctx, n_test=max(test_sizes),
                       cutoff_day=args.cutoff, seed=args.seed)
    model, why = tabicl(device)
    rows_1b = []
    if model is None:
        print(f"  unavailable -- {why}")
    else:
        t0 = time.perf_counter()
        model.fit(d.X_train, d.y_train)
        print(f"  fit on {len(d.X_train):,} context rows: "
              f"{time.perf_counter() - t0:.2f}s (once)\n")
        for k in test_sizes:
            try:
                t0 = time.perf_counter()
                model.predict_proba(d.X_test.iloc[:k])
                dt = time.perf_counter() - t0
            except Exception as exc:
                print(f"  score {k:>6,}: FAILED -- {_explain(exc)}")
                continue
            rows_1b.append((k, dt))
            print(f"  score {k:>6,} rows | {dt:7.2f}s | {dt / k:.4f}s per row",
                  flush=True)

    if len(rows_1b) >= 2:
        (k_lo, t_lo), (k_hi, t_hi) = rows_1b[0], rows_1b[-1]
        # Least-squares line through the measured points: cost = fixed + m * k.
        ks = np.array([k for k, _ in rows_1b], dtype=float)
        ts = np.array([t for _, t in rows_1b], dtype=float)
        m, fixed = np.polyfit(ks, ts, 1)
        print(f"""
  Read the per-row column, not the total. It FALLS as you score more rows
  ({t_lo / k_lo:.4f}s per row at {k_lo:,}, {t_hi / k_hi:.4f}s at {k_hi:,}), which means
  the cost is not really per-row at all.

  Fitting a line through those points:

      predict cost  ~  {fixed:.1f}s  +  {m:.4f}s x (rows scored)

  The {fixed:.1f}s is the fixed cost of reading the context. You pay it on every
  single call to predict_proba, whether you then score one row or two thousand.
  The marginal cost of an extra row is only {m * 1000:.1f}ms.

  This is the most useful number in the script, and section 3a is what it
  implies.""")

    # ---------------------------------------------------------------- 2
    header("2. WHAT THE CONTEXT WINDOW IS")
    print("""
The ceiling is not a soft degradation. Past it, these models raise.

TabPFN's CPU caps, read from the installed package rather than from docs
(tabpfn.inference_config.cpu_sample_limit):
""")
    for ver, lim in sorted(CPU_LIMITS.items()):
        print(f"    {ver:12s} {lim:>7,} context rows")
    print("""
Two things worth knowing about those numbers:

  * They are a CPU restriction, not the model's real limit. The documented GPU
    ceiling for TabPFN-3.5 is 1,000,000 rows and 20,000 features -- three orders
    of magnitude more. (That figure is from the docs; this script has no GPU to
    check it with.)

  * They are what makes OULAD awkward. Day 28 has 27,538 rows, so on CPU every
    comparison in 30-oulad.py is run on a subsample out of necessity.

TabICL is the other family, and it was built for larger tables. It enforces no
such CPU cap, which is testable rather than something to take on faith:""")

    if args.max_context and args.max_context > 0:
        cap = max(CPU_LIMITS.values()) if CPU_LIMITS else 5000
        print(f"""
  Running TabICL at {args.max_context:,} context rows -- above TabPFN's
  {cap:,}-row CPU cap. This step is memory-hungry (it peaked near 9 GB at 6,000
  rows on the machine this was written on); --max-context 0 skips it.
""")
        d_big = T.sample_oulad(n_train=args.max_context, n_test=200,
                               cutoff_day=args.cutoff, seed=args.seed)
        model, why = tabicl(device)
        if model is None:
            print(f"  unavailable -- {why}")
        else:
            try:
                t_fit, t_pred, _ = timed_fit_predict(model, d_big, 200)
                print(f"  context {len(d_big.X_train):,} rows | fit {t_fit:.2f}s | "
                      f"predict 200 rows {t_pred:.2f}s  -- no cap, it just runs")
                print(f"""
  So the ceiling is a property of the particular model, not of in-context
  learning. If your table is bigger than TabPFN's CPU cap and you have no GPU,
  that is a reason to reach for TabICL rather than to give up on the idea.

  Note how little the 200 scored rows cost relative to the context: by section
  1b's decomposition, almost all of that {t_pred:.0f}s is the fixed cost of reading
  {len(d_big.X_train):,} context rows -- which is also why this is the slowest step here
  despite scoring the fewest rows.""")
            except MemoryError:
                print("  ran out of memory -- lower --max-context")
            except Exception as exc:
                print(f"  FAILED -- {_explain(exc)}")
    else:
        print("\n  (skipped: --max-context 0)")

    # ---------------------------------------------------------------- 3a
    header("3a. WHAT BREAKS: SCORING IN A LOOP")
    print(f"""
Section 1b found a large fixed cost per predict_proba call. So the single worst
thing you can do is call it repeatedly on small batches -- which is exactly what
a row-at-a-time production service, or a tidy-looking `for chunk in chunks`,
does.

Same {chunk_test} rows, same fitted model, scored two ways.
""")

    d = T.sample_oulad(n_train=chunk_ctx, n_test=chunk_test,
                       cutoff_day=args.cutoff, seed=args.seed)
    model, why = tabicl(device)
    if model is None:
        print(f"  unavailable -- {why}")
    else:
        model.fit(d.X_train, d.y_train)

        t0 = time.perf_counter()
        p_one = model.predict_proba(d.X_test)[:, 1]
        t_one = time.perf_counter() - t0
        print(f"  {'1 call':>3s},  {chunk_test:>4,} rows at once : {t_one:7.2f}s")

        t0 = time.perf_counter()
        parts = [model.predict_proba(d.X_test.iloc[i:i + chunk])[:, 1]
                 for i in range(0, len(d.X_test), chunk)]
        t_chunked = time.perf_counter() - t0
        p_chunked = np.concatenate(parts)
        n_calls = len(parts)
        print(f"  {n_calls:>3,} calls, {chunk:>4,} rows each  : {t_chunked:7.2f}s")

        agree = float(np.abs(p_one - p_chunked).max())
        ratio = t_chunked / max(t_one, 1e-9)
        print(f"""
  Identical answers (max difference {agree:.2e}), {ratio:.1f}x the time.

  The {n_calls} calls each re-read the same {len(d.X_train):,} context rows. You paid for
  that {n_calls} times instead of once, and got nothing for it.""")
        if ratio < 1.5:
            print("""
  On this run the penalty came out small. That happens when the context is
  short enough that the fixed cost is not the dominant term -- re-run with a
  larger --n-test or a bigger context and the gap opens up.""")
        else:
            print("""
  The rule follows directly: predict on the largest batch you can hold in
  memory, once. If a deployment genuinely needs single-row latency, a
  foundation model used this way is the wrong tool, and that is a design
  decision to make before you build on it, not after.""")

    # ---------------------------------------------------------------- 3b
    header("3b. WHAT BREAKS: HELPFUL PREPROCESSING")
    print(f"""
01-setup.md states three rules that invert normal practice: do not scale, do not
one-hot encode, do not impute. They are easy to nod along to and hard to
believe, so here they are tested.

Same {pre_ctx:,} context rows and {pre_test:,} scored rows, twice: once raw --
NaN left alone, the categorical passed as-is -- and once through the same
standardise + one-hot + median-impute pipeline the logistic-regression baseline
uses.
""")

    d = T.sample_oulad(n_train=pre_ctx, n_test=pre_test,
                       cutoff_day=args.cutoff, seed=args.seed)
    scores = {}
    for label, prep in (("raw (as the docs say)", False),
                        ("scaled + one-hot + imputed", True)):
        model, why = tabicl(device)
        if model is None:
            print(f"  {label:28s} unavailable -- {why}")
            continue
        if prep:
            pre = T._preprocessor(d.X_train, scale=True)
            Xtr = pre.fit_transform(d.X_train)
            Xte = pre.transform(d.X_test)
        else:
            Xtr, Xte = d.X_train, d.X_test
        try:
            t0 = time.perf_counter()
            model.fit(Xtr, d.y_train)
            proba = model.predict_proba(Xte)[:, 1]
            dt = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {label:28s} FAILED -- {_explain(exc)}")
            continue
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(d.y_test, proba)
        cols = Xtr.shape[1]
        scores[label] = ap
        print(f"  {label:28s} PR-AUC {ap:.4f}  ({cols} columns in, {dt:.1f}s)",
              flush=True)

    if len(scores) == 2:
        raw = scores["raw (as the docs say)"]
        done = scores["scaled + one-hot + imputed"]
        delta = raw - done
        favour = "raw" if delta > 0 else "preprocessing"
        print(f"""
  Difference: {delta:+.4f} PR-AUC in favour of {favour}.
  Base rate here is {d.base_rate:.3f}, on {pre_test:,} scored rows.""")

        # A PR-AUC gap on one split of this size is not evidence of much. Be
        # strict about what counts as a real difference, or this section ends up
        # asserting the docs are right on the strength of noise.
        if abs(delta) < 0.02:
            print(f"""
  That is too small to mean anything. On {pre_test:,} scored rows a PR-AUC
  difference of ~0.01-0.02 is about the sampling error, so this run neither
  confirms nor refutes the advice -- it only shows preprocessing did not help
  enough to be visible.

  The mechanism the docs give is still the reason to follow the rule: the model
  was pretrained on particular distributions and does its own normalisation, and
  one-hot encoding destroys the column-is-categorical signal it would otherwise
  use. But the honest argument for passing raw data is that it is less code and
  makes fewer silent claims about your data, not that it reliably scores better.
  Vary --seed if you want to see how much this number moves on its own.""")
        elif delta > 0:
            print("""
  That is a large enough gap to take seriously, on this split at least:
  preprocessing hurt. The mechanism is that the model was pretrained on
  particular distributions and does its own normalisation, so standardising
  first hands it something it was not trained to expect, and one-hot encoding
  destroys the column-is-categorical signal it wants to use directly.

  Still one split. Vary --seed before quoting the number.""")
        else:
            print("""
  Preprocessing won, and by a margin too wide to dismiss as noise -- which is
  NOT what the docs predict. One split cannot overturn the advice, so vary
  --seed before concluding anything, but it is worth knowing the rule is not
  absolute on messy real data.""")

    # ---------------------------------------------------------------- 3c
    header("3c. WHAT BREAKS: NaN, AND WHO MINDS")
    print("""
The OULAD features carry NaN that MEANS something -- no submission is different
from a submission of zero. Ordinary models mostly refuse to see it at all.
""")
    d = T.sample_oulad(n_train=300, n_test=100, cutoff_day=args.cutoff,
                       seed=args.seed)
    n_nan_cols = int((d.X_train.isna().mean() > 0).sum())
    print(f"  {n_nan_cols} of {d.X_train.shape[1]} columns contain NaN\n")

    from sklearn.linear_model import LogisticRegression
    num = d.X_train.select_dtypes("number")
    try:
        LogisticRegression(max_iter=200).fit(num, d.y_train)
        print("  logistic regression : accepted NaN (unexpected)")
    except Exception as exc:
        first = str(exc).strip().splitlines()[0]
        print(f"  logistic regression : {type(exc).__name__} -- {first[:88]}")

    model, why = tabicl(device)
    if model is None:
        print(f"  TabICL              : unavailable -- {why}")
    else:
        try:
            model.fit(d.X_train, d.y_train)
            model.predict_proba(d.X_test.iloc[:20])
            print("  TabICL              : fitted and scored it, NaN and "
                  "free-text categorical included")
        except Exception as exc:
            print(f"  TabICL              : FAILED -- {_explain(exc)}")

    print("""
  This is the one place the foundation model is unambiguously less work. The
  baselines in tabfm_bench.py need a ColumnTransformer with an imputer and a
  one-hot encoder before they will run at all; TabICL takes the frame.

  Note what imputation costs you conceptually, beyond the code: filling NaN with
  a median asserts that a student who submitted nothing is an average student.
  That is a claim about the data, inserted silently by a preprocessing step.""")

    header("WHAT TO TAKE FROM THIS")
    print("""
  1. predict() cost is dominated by the CONTEXT, not by how much you score.
     There is a large fixed cost per call for re-reading the training rows, and
     a small marginal cost per scored row. Section 1b measured both.

  2. Therefore: batch as hard as you can. One call for the whole test set.
     Looping is the single easiest way to make these models look unusable.

  3. The context ceiling is a hard error, and on CPU it is low enough to be the
     binding constraint -- 5,000 rows for TabPFN-3.5 against OULAD's 27,538.
     TabICL does not share that cap.

  4. The preprocessing advice is real but its margin is small; the robust
     argument for passing raw data is that it is less code and makes fewer
     silent claims about your data, not that it always wins.

  Next: 30-oulad.py, which puts this against the tuned gradient boosting
  baseline on the full 45-feature problem. RESULTS.md records what that
  measured, including the fit-versus-predict table across training sizes.
""")


if __name__ == "__main__":
    main()
