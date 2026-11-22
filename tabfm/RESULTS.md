# Measured results: TabICL on OULAD

Real numbers from an actual run, **22 November 2026**. Everything below was
produced by `tabfm/30-oulad.py` on the committed OULAD parquet files — no
estimates, no figures carried over from the book.

**Headline: TabICL beat gradient boosting at every training size tested and on
every seed tested — 9 for 9 on the three-seed check — and it cost about 5,000x
more per prediction.**

TabPFN did not run. See [Why TabPFN did not run](#why-tabpfn-did-not-run).

---

## Hardware and versions

Both matter more than usual here, because the interesting result is a
time-versus-accuracy trade-off and these numbers are **CPU-only**.

```
CPU       Intel(R) Xeon(R) @ 2.10GHz, 4 cores
GPU       none  (torch.cuda.is_available() == False)
RAM       15 GiB  (peak RSS observed ~4.8 GiB during TabICL predict)
OS        Ubuntu 24.04.5 LTS
Python    3.13.18

tabicl          2.2.0     checkpoint tabicl-classifier-v2-20260212.ckpt (110,368,038 bytes)
tabpfn         10.2.0     did not run
torch          2.15.0+cu132
numpy           2.5.5
pandas          3.1.1
scikit-learn    1.9.1
pyarrow        26.0.1
```

The installed versions of `tabpfn`, `tabicl` and `torch` match what
[`01-setup.md`](01-setup.md) claims, as does its CPU row-cap table.

## The split

```
python tabfm/30-oulad.py --n-train 4000 --n-test 4000
```

Day-28 cutoff, grouped on `id_student`, `seed=0`.

| | |
|---|---|
| rows (context) | 4,000 |
| rows (to predict) | 4,000 |
| features | 45 |
| columns with missing values | 14 of 45 |
| categorical columns | 1 (index 43, `region`) |
| base rate (PR-AUC floor) | 0.436 |
| numeric range across columns | −995.7 to 5,379 |

## 1. The comparison

PR-AUC's floor *is* the base rate, so `lift` (PR-AUC ÷ 0.4363) is the number to
read. Phase 2's full-data gradient boosting reference is **0.766 (lift 1.74)**.

| model | PR-AUC | lift | ROC-AUC | fit s | predict s |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.4363 | 1.00 | 0.5000 | 0.00 | 0.00 |
| logistic regression | 0.7312 | 1.68 | 0.7627 | 0.07 | 0.01 |
| gradient boosting (defaults) | 0.7348 | 1.68 | 0.7661 | 0.45 | 0.02 |
| TabPFN | — | — | — | — | — |
| **TabICL** | **0.7585** | **1.74** | **0.7860** | **0.81** | **93.31** |

TabICL wins by **+0.0237 PR-AUC** over gradient boosting on the same 4,000 rows,
with no tuning at all.

The comparison against Phase 2 is the more striking one. Phase 2's gradient
boosting needed roughly 19,000 training rows to reach lift 1.74. TabICL reached
the same lift on **4,000** rows — about a fifth of the data. On this dataset the
CPU row cap cost less than you would expect.

Scores are deterministic: three separate invocations returned 0.7585 / 0.7348 /
0.7312 / 0.4363 identically. Only the timings moved (TabICL predict 92.1–96.6 s
across runs).

Wall-clock for the whole run above: **1 m 46 s**, essentially all of it TabICL's
`predict_proba`.

## 2. Does the advantage depend on size?

```
python tabfm/30-oulad.py --n-train 4000 --n-test 4000 --sweep
```

Test set held fixed at 4,000 rows; only the context grows. PR-AUC:

| n_train | logistic | gbm | TabPFN | TabICL | TabICL − gbm |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.6780 | 0.6597 | — | **0.7029** | +0.0432 |
| 500 | 0.6915 | 0.6955 | — | **0.7187** | +0.0232 |
| 1,000 | 0.7247 | 0.7041 | — | **0.7413** | +0.0372 |
| 2,000 | 0.7418 | 0.7306 | — | **0.7616** | +0.0310 |
| 4,000 | 0.7312 | 0.7348 | — | **0.7585** | +0.0237 |

Wall-clock for the sweep: **6 m 58 s**.

Two things to read off this, one expected and one not:

**Expected.** TabICL's margin is largest where the claim says it should be — at
250 rows it is +0.043, the widest gap in the table, and the trend is broadly
downward as data grows. This is the small-data regime the book is about, and the
effect is visible.

**Not expected.** The gap does not close to zero by 4,000 rows; TabICL still
leads by +0.024. Nothing here shows where it *would* close, because the CPU row
cap stops the sweep well before gradient boosting has the ~19,000 rows it used
in Phase 2. **This sweep does not test the claim at the sizes where the
incumbent is supposed to catch up.** Treat the last row as the limit of what was
measured, not as a converged answer.

One caveat on reading individual cells: logistic regression goes *down* from
2,000 to 4,000 rows (0.7418 → 0.7312). That is a single grouped split, and the
script's own closing advice applies — a 0.01 PR-AUC difference on one split is
noise. Everything in this section is `seed=0`; the next section re-runs the
ranking on three seeds.

## 3. Does the ranking survive a change of split?

The sweep above is one split. `--seed` changes the `GroupShuffleSplit` *and* the
subsample, so each seed is a genuinely different grouped partition of the same
27,538 enrolments. Three seeds x three sizes, test set fixed at 4,000, PR-AUC:

| n_train | seed | base rate | logistic | gbm | TabICL | TabICL − gbm | winner |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 250 | 0 | 0.4385 | 0.6780 | 0.6597 | 0.7029 | +0.0432 | TabICL |
| 250 | 1 | 0.4475 | 0.6417 | 0.6467 | 0.7053 | +0.0586 | TabICL |
| 250 | 2 | 0.4320 | 0.6208 | 0.6405 | 0.6673 | +0.0268 | TabICL |
| 1,000 | 0 | 0.4365 | 0.7247 | 0.7041 | 0.7413 | +0.0372 | TabICL |
| 1,000 | 1 | 0.4537 | 0.7077 | 0.6898 | 0.7303 | +0.0405 | TabICL |
| 1,000 | 2 | 0.4457 | 0.7095 | 0.7045 | 0.7372 | +0.0327 | TabICL |
| 4,000 | 0 | 0.4363 | 0.7312 | 0.7348 | 0.7585 | +0.0237 | TabICL |
| 4,000 | 1 | 0.4452 | 0.7247 | 0.7374 | 0.7579 | +0.0204 | TabICL |
| 4,000 | 2 | 0.4415 | 0.7138 | 0.7338 | 0.7448 | +0.0111 | TabICL |

Summary over seeds:

| n_train | gbm mean | TabICL mean | gap mean | gap min | gap max | TabICL wins |
|---:|---:|---:|---:|---:|---:|:--:|
| 250 | 0.6490 | 0.6918 | +0.0428 | +0.0268 | +0.0586 | 3/3 |
| 1,000 | 0.6995 | 0.7362 | +0.0368 | +0.0327 | +0.0405 | 3/3 |
| 4,000 | 0.7353 | 0.7537 | +0.0184 | +0.0111 | +0.0237 | 3/3 |

Wall-clock: **10 m 11 s**.

**The ranking holds. TabICL won 9 out of 9**, and the gap never once crossed
zero. The consistency of the *sign* is the real evidence here, more than any
single margin.

It also sharpens the size story. Averaged over seeds the gap shrinks
monotonically — **+0.043 → +0.037 → +0.018** as context grows from 250 to 4,000
— which is the book's claim behaving as advertised, and it was partly obscured
on `seed=0` alone (where 1,000 rows happened to show a wider gap than 500).

Two honest qualifications:

* **At 4,000 rows the margin is getting close to the noise floor.** The
  narrowest single result is +0.0111 (seed 2), and on a 4,000-row test set a
  PR-AUC difference of ~0.01 is around the size of the sampling error. What
  carries the conclusion at this size is that all three seeds agree in
  direction, not the magnitude of any one of them.
* **Three seeds is enough to kill a coin-flip, not to put an error bar on the
  gap.** If you need a publishable interval rather than a ranking, this wants
  more seeds and ideally repeated grouped cross-validation.

One incidental finding about the baselines: on `seed=0` gradient boosting led
logistic regression at 4,000 rows by only +0.0036 (0.7348 vs 0.7312), close
enough to call a tie. Across three seeds it leads in all three, by +0.0121 on
average (0.7353 vs 0.7232), so the near-tie was a feature of that one split.
The 2,000-row cell where logistic actually *beat* gbm (0.7418 vs 0.7306) was not
re-run across seeds, so it stands untested.

TabICL's own `random_state` is left at its default of 42 throughout, so this
varies the data split only, not the model's internal sampling.

## 4. fit() versus predict(): the cost

This is the part worth internalising. Same splits, TabICL against gradient
boosting, seconds:

| n_train | GBM fit | GBM predict | TabICL fit | TabICL predict | predict ratio |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.13 | 0.018 | 0.77 | 41.08 | 2,230x |
| 500 | 0.23 | 0.023 | 0.69 | 43.27 | 1,869x |
| 1,000 | 0.32 | 0.017 | 0.71 | 50.56 | 2,926x |
| 2,000 | 0.36 | 0.017 | 0.67 | 66.13 | 3,820x |
| 4,000 | 0.42 | 0.017 | 0.76 | 96.58 | 5,662x |

The three-seed run reproduces this independently: TabICL `fit` stayed in
0.62–0.82 s at every size, while `predict` moved with the context only —
37.8–40.1 s at 250 rows, 50.3–51.2 s at 1,000, 96.0–98.2 s at 4,000. The
pattern is a property of the method, not of one split.

The inversion the book describes is real and it is extreme:

* **Gradient boosting** spends its time in `fit` (0.13 → 0.42 s, growing with
  data) and predicts in a flat ~0.017 s regardless of training size.
* **TabICL** has a nearly constant `fit` (~0.7 s) that does not grow with data,
  and all the cost lands in `predict` — which grows with the *context* even
  though the test set never changes.

That last point is the one to sit with. Scoring the *same* 4,000 students cost
41 s with 250 context rows and 97 s with 4,000. For an ordinary model, more
training data makes prediction no slower. For a tabular foundation model the
training data is *part of the prompt*, so it is re-read on every prediction call.

The practical reading: on CPU, TabICL bought +0.024 PR-AUC for roughly a
**5,000x** increase in per-prediction cost. For a batch job scoring students
once a week, 97 seconds is nothing and the accuracy is free. For anything
interactive, or for scoring the full 27,538-row cohort, this is the number that
decides it — and it is an argument for a GPU, not against the model.

One correction to the framing in [`README.md`](README.md), which says `fit()`
"stores your data — milliseconds": TabICL's `fit` is ~0.7 s, not milliseconds,
because it loads the 110 MB checkpoint from disk and fits its ensemble
generator. It is still effectively constant in the data size, which is the real
point.

## Why TabPFN did not run

Verbatim, `tabpfn.errors.TabPFNLicenseError`:

```
TabPFN requires a one-time license acceptance to download model weights
for local inference. Your datasets stay local and are never sent to Prior
Labs, and TabPFN works offline once the weights are downloaded.

There is no interactive terminal available.

To authenticate in a non-interactive environment:
  1. Open https://ux.priorlabs.ai in a browser and log in (or register)
  2. Accept the license on the Licenses tab
  3. Copy your API Key from https://ux.priorlabs.ai/account
  4. Set the environment variable: export TABPFN_TOKEN="<your-api-key>"
     or in Python (before calling .fit()): import os; os.environ["TABPFN_TOKEN"] = "<your-api-key>"
```

Raised from `tabpfn/browser_auth.py:695`, via
`model_loading.py:569 _download_model → ensure_license_accepted`.

**This is not the Hugging Face gate `01-setup.md` describes**, and no `HF_TOKEN`
would have fixed it. See [Corrections](#corrections-to-01-setupmd) below.

## Corrections to 01-setup.md

Found by actually running it. These are the parts that are wrong.

### 1. Blocker 1 is wrong: TabPFN's weights are not gated on Hugging Face

`01-setup.md` says TabPFN's repository is gated and that the fix is an
`HF_TOKEN`. Neither holds with `tabpfn` 10.2.0:

* The HF API reports `"gated": false` for `Prior-Labs/tabpfn_3_5`. The
  `.safetensors` files are listed and publicly resolvable.
* The gate is a **Prior Labs licence check that runs locally**, inside
  `tabpfn/browser_auth.py`, *before any Hugging Face request is made*. It wants
  a **`TABPFN_TOKEN` from `ux.priorlabs.ai`** — a different service, a different
  account and a different variable from Hugging Face.
* The exception raised is `TabPFNLicenseError`, **not**
  `TabPFNHuggingFaceGatedRepoError`. That class does still exist in
  `tabpfn.errors`, which is presumably why the doc expects it, but it is not
  what fires.
* The error text the doc quotes — "HuggingFace authentication error downloading
  from 'Prior-Labs/tabpfn_3_5'. This model is gated…" — never appears.

So steps 1–4 of that section (make an HF account, accept terms on the HF model
page, create an HF read token, `export HF_TOKEN=...`) would not have unblocked
anything. The whole blocker moved from Hugging Face to Prior Labs' own licence
service.

### 2. The cloud-session host list is incomplete — this is what actually blocked TabICL

`01-setup.md` says the `OULAD-Model` environment has "the needed hosts allowed"
and lists seven. **huggingface.co was reachable (HTTP 200) and the TabICL
download still failed.** The listed hosts are not sufficient.

`huggingface_hub` 2.4.0 ships `hf-xet`, so weight downloads go over the Xet
protocol, which needs two hosts that were not on the list:

| host | role | was it allowed? |
|---|---|---|
| `cas-server.xethub.hf.co` | Xet reconstruction API | **no — had to be added** |
| `us.aws.cdn.hf.co` | Xet chunk data (`/xorbs/…`) | **no — had to be added** |
| `transfer.xethub.hf.co` | on the list already | never contacted |

Both had to be added before the 110 MB checkpoint would download; after that it
took **4.4 s**. Note that the non-Xet fallback does not rescue you — a plain
`resolve/main/...` request 302-redirects to `us.aws.cdn.hf.co` too.

The failure also does not look like the doc's example. It is not
`ProxyError: 403 Forbidden`. The download hangs for ~90 s retrying and then
raises, with the real cause only visible in
`~/.cache/huggingface/xet/logs/`:

```
Reqwest(reqwest::Error { kind: Request,
  url: "https://cas-server.xethub.hf.co/v2/reconstructions/45c13f18…",
  source: hyper_util::client::legacy::Error(Connect, TunnelUnsuccessful) })
```

`transfer.xethub.hf.co` appears to be stale — it is on the allowlist and was
never used.

### 3. Minor

* "sixteen `nvidia-*` packages" — it is **15** (19 counting `cuda-toolkit`,
  `cuda-bindings`, `cuda-pathfinder` and `triton`). The ~5 GB figure is right:
  3.0 GB of `nvidia/`, 1.2 GB of `torch/`, **5.7 GB** venv total.
* `download.pytorch.org` is **not** allowed by the environment, so the CPU-only
  torch install in the Install section cannot be run as written. The doc already
  flags this further down; worth hoisting, since it is the first command a
  reader meets.
* [`README.md`](README.md)'s file table lists three `.ipynb` notebooks
  (`10-first-contact.ipynb`, `20-what-fit-actually-does.ipynb`,
  `30-oulad-the-real-test.ipynb`). The actual files are `10-first-contact.py`
  and `30-oulad.py`, and there is no `20-` file at all.

### Confirmed correct

Worth saying, since most of the doc held up: the pinned versions (tabpfn 10.2.0,
tabicl 2.2.0, torch 2.15.0); the CPU row-cap table (v2/v2.5/v2.6 → 1,000;
v3/v3.5/v3.5-fast → 5,000, read from the installed package); the accepted
`ModelVersion` strings; every parameter in the API section
(`categorical_features_indices`, `ignore_pretraining_limits`, `n_estimators`,
`create_default_for_version`); the cache locations `~/.cache/tabpfn/` and
`~/.cache/huggingface/`; and that TabICL's `jingang/TabICL` is not gated
(`"gated": false`, downloads with no credentials — only a rate-limit warning).

## Reproducing

```bash
python -m venv .venv
.venv/bin/pip install -e ".[tabfm]"        # ~5.7 GB, several minutes
.venv/bin/python tabfm/30-oulad.py --n-train 4000 --n-test 4000
.venv/bin/python tabfm/30-oulad.py --n-train 4000 --n-test 4000 --sweep
```

In a cloud session, the environment needs `cas-server.xethub.hf.co` and
`us.aws.cdn.hf.co` allowed in addition to the hosts `01-setup.md` lists.

For the seed check in section 3, vary `--seed` over 0, 1 and 2 at each
`--n-train`:

```bash
for s in 0 1 2; do
  for n in 250 1000 4000; do
    .venv/bin/python tabfm/30-oulad.py --n-train $n --n-test 4000 --seed $s
  done
done
```

The per-size timings in section 4, and the compact seed tables in section 3,
come from short scripts calling `tabfm_bench.run_gbm` and `run_tabicl` directly;
`--sweep` itself prints only PR-AUC, and the harness prints no cross-seed
summary.

Budget roughly 20 minutes of CPU for everything above, and expect TabPFN to be
absent from every table unless you hold a Prior Labs `TABPFN_TOKEN`.
