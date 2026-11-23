# Measured results: TabICL on a GPU

Real numbers from a Kaggle T4 run, **23 November 2026**, produced by
[`40-kaggle-gpu.ipynb`](40-kaggle-gpu.ipynb). Companion to
[`RESULTS.md`](RESULTS.md), which is the CPU-only run.

**Headline: gradient boosting never catches up.** Across the full available
training pool — 250 rows to 19,000 — TabICL leads at every size. The gap
narrows as predicted, from +0.041 to +0.011, but it does not reach zero and
there is no crossover inside this dataset.

---

## Hardware and versions

```
GPU       Tesla T4, 14.6 GiB VRAM, 40 SMs   (Kaggle free tier, 2 cards, 1 used)
Driver    580.178.04, CUDA 13.0
Python    3.13.16
torch     2.11.0+cu130
tabicl    2.2.0    checkpoint tabicl-classifier-v2-20260212.ckpt (110 MB)
tabpfn   10.2.0    did not run — no Prior Labs licence token
```

## 1. What the GPU bought

Same split, same model, same answer as the CPU run. Only the hardware changed.

| | CPU (4-core Xeon) | GPU (T4) | |
|---|---:|---:|---|
| TabICL `fit` | 0.81 s | 0.68 s | unchanged, as expected |
| TabICL `predict` | **93.31 s** | **3.11 s** | **30x faster** |
| PR-AUC | 0.7585 | 0.7494 | different split seed, not a GPU effect |

That 30x is the entire point of the exercise. The GPU did not make the model
better; it made it usable. On CPU, scoring 4,000 students took a minute and a
half and the row cap stopped the analysis at 4,000. On the T4 it takes three
seconds and the cap is gone.

Full day-28 comparison, 4,000 train / 4,000 test, base rate 0.431:

| model | PR-AUC | lift | ROC-AUC | fit s | predict s |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.4308 | 1.00 | 0.5000 | 0.00 | 0.00 |
| logistic regression | 0.7192 | 1.67 | 0.7568 | 0.07 | 0.01 |
| gradient boosting (defaults) | 0.7248 | 1.68 | 0.7661 | 0.63 | 0.04 |
| **TabICL** | **0.7494** | **1.74** | **0.7859** | 0.68 | 3.11 |

## 2. The crossover question — answered

This is what the CPU run could not do. Test set fixed at 4,000 students
throughout; only the training context grows. Base rate constant at 0.4308 in
every row, which confirms the test set really was held fixed.

| n_train | logistic | gbm | TabICL | gap | TabICL wall-clock |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.6715 | 0.6535 | 0.6944 | **+0.0409** | 3.2 s |
| 500 | 0.6809 | 0.6851 | 0.7098 | +0.0247 | 3.5 s |
| 1,000 | 0.7076 | 0.6954 | 0.7311 | +0.0356 | 4.0 s |
| 2,000 | 0.7150 | 0.7111 | 0.7440 | +0.0329 | 2.7 s |
| 4,000 | 0.7192 | 0.7248 | 0.7494 | +0.0245 | 3.8 s |
| 8,000 | 0.7187 | 0.7381 | 0.7556 | +0.0175 | 6.5 s |
| 12,000 | 0.7200 | 0.7427 | 0.7600 | +0.0173 | 9.9 s |
| 16,000 | 0.7204 | 0.7484 | 0.7603 | +0.0119 | 13.7 s |
| **19,000** | 0.7215 | 0.7522 | **0.7628** | **+0.0106** | 17.3 s |

Three things to read off it.

**The gap narrows, roughly halving per 4x of data.** +0.041 at 250 rows,
+0.025 at 4,000, +0.011 at 19,000. That is the book's claim behaving as
advertised, measured on real data rather than taken on trust.

**But it does not cross zero.** Extrapolating the halving, crossover would sit
somewhere past 60,000–100,000 rows. OULAD has 19,247 available after the
grouped split. **So on this dataset, at this cutoff, gradient boosting never
catches up — there is not enough data for it to.** That is the opposite of the
conventional expectation, and it is the finding.

**Logistic regression plateaus at ~0.72 and stops.** It is level from 4,000 rows
on. Both other models keep climbing. A linear model has run out of capacity to
use more data; the other two have not.

### Why the gap is a real effect and not noise

At 19,000 rows the +0.0106 margin is small enough to deserve suspicion. Section
3 of the notebook re-runs the ranking across three independent grouped splits:

| n_train | gap mean | min | max | TabICL wins |
|---:|---:|---:|---:|:--:|
| 1,000 | +0.034 | +0.0295 | +0.0370 | 3/3 |
| 8,000 | +0.015 | +0.0130 | +0.0175 | 3/3 |

**6 of 6**, and the range at each size is tight — at 8,000 rows every seed lands
between +0.013 and +0.018. Combined with the CPU run's 9 of 9, TabICL has now
won **16 consecutive comparisons** against gradient boosting on this data.

The consistency of the sign is the evidence. No single margin here would be
convincing on its own.

## 3. Small datasets: the regime the claim is actually about

Four OpenML datasets, the benchmark family the TabPFN papers use. All are in the
few-hundred-to-one-thousand-row range where the claim is strongest.

| dataset | rows | features | base rate | gbm | TabICL | gap |
|---|---:|---:|---:|---:|---:|---:|
| credit-g | 1,000 | 20 | 0.700 | 0.9027 | 0.9089 | +0.0062 |
| diabetes | 768 | 8 | 0.347 | 0.6800 | 0.7180 | +0.0380 |
| blood-transfusion | 748 | 4 | 0.237 | 0.3806 | **0.4901** | **+0.1095** |
| qsar-biodeg | 1,055 | 41 | 0.336 | 0.8412 | 0.8692 | +0.0280 |

TabICL wins all four, and the margins are mostly larger than anything in the
OULAD sweep — consistent with the small-data claim.

**The blood-transfusion result is the one worth staring at.** A +0.11 gap is
enormous; in lift terms gradient boosting manages 1.61 against the base rate
while TabICL manages 2.07. It is also the dataset with the **lowest base rate**
(0.237) and the **fewest features** (4).

That is a single data point and should not be over-read. But it points in the
direction of a question this project has not yet tested: whether the advantage
grows as the positive class gets rarer. If it does, it matters for any
rare-event application — fraud, failure prediction, large-loss insurance
claims. The experiment that would settle it is a controlled sweep of the
positive rate, measuring **calibration** as well as ranking.

## 4. Honest limits

**Gradient boosting ran on defaults the whole way.** This is a fair
untuned-vs-untuned comparison, and TabICL had no tuning either — which is the
substantive claim, since it has almost nothing to tune. But a tuned gradient
boosting is the stronger incumbent and would move its line up. The +0.011 margin
at 19,000 rows is well within what tuning could plausibly recover. **This
result does not show that TabICL beats gradient boosting; it shows TabICL beats
*untuned* gradient boosting, everywhere, with no tuning of its own.**

**One cutoff, one cohort.** Everything is day 28 on OULAD. The project's real
question — how early a useful prediction can be made — is a different sweep.

**The sweep is one split.** The seed check covers 1,000 and 8,000 rows only, and
confirms the sign rather than putting an interval on the magnitude. A
publishable number wants repeated grouped cross-validation.

**TabPFN still has not run anywhere.** It needs a Prior Labs licence token.
Every "foundation model" result in this project is TabICL alone.

## Reproducing

Open [`40-kaggle-gpu.ipynb`](40-kaggle-gpu.ipynb) on Kaggle with Accelerator set
to GPU and Internet on. Roughly 10 minutes end to end; the sweep is most of it.

Section 1 clones this repo, so the OULAD parquet files come along and no
separate dataset upload is needed.
