# Phase 2 — the feature set

`build_features(tables, cutoff_day)` in `src/oulad/features.py`. One row per
enrolment, 45 model-input columns at the day-28 cutoff.

Everything routes through that one function so the cutoff rule has exactly one
place it can be violated, and one place to test.

---

## The families, and what each is for

| Family | Columns | Rationale |
|---|---|---|
| **Volume** | `total_clicks`, `active_days`, `distinct_sites`, `clicks_per_active_day` | The obvious signal, and the weakest — EDA showed the distributions overlap heavily |
| **Consistency** | `gap_cv`, `max_gap`, `mean_gap` | Whether engagement is even or bursty, separately from how much. **Measured null — see below** |
| **Trajectory** | `clicks_week_0..N`, `click_slope`, `last_week_share`, `days_since_last_activity`, `first_active_day` | A single total is direction-blind |
| **Diversity** | `clicks_<activity_type>` × 11, `n_activity_types` | Activity types differ in how strongly they separate outcomes |
| **Assessment** | `assessment_due`, `submitted`, `n_submitted`, `first_score`, `mean_score`, `days_before_deadline` | Strongest single signal in the EDA, and the trickiest missingness |
| **Demographics** | `imd_band_ord`, `education_ord`, `age_ord`, `is_female`, `has_disability`, `num_of_prev_attempts`, `studied_credits`, `region` | Modest, and deliberately not central |
| **Context** | `date_registration`, `cutoff_fraction` | Known before the course starts |

Weekly columns scale with the cutoff: day 14 produces `clicks_week_0..2`, day 28
produces `clicks_week_0..4`. Each cutoff is modelled separately, so this is fine.

## Decisions that needed making

### Missing values are left missing

No imputation happens in `build_features`. Filling a NaN with a median computed
over the whole dataset leaks test-fold information into training through the
fill value. Imputation belongs inside the cross-validation pipeline, fitted on
training folds only.

The exception is columns where **zero is an observation, not an absence**: a
student with no clickstream rows genuinely made zero clicks. Those are filled
with 0. But `days_since_last_activity` for that student stays NaN — "never
active" has no last-activity day, and 0 would be a lie in the opposite
direction.

### Three assessment situations, kept distinct

A naive encoding collapses three different things into "no submission":

1. The course had **nothing due yet** at the cutoff.
2. Something was due and they **submitted**.
3. Something was due and they **did not**.

Merging 1 and 3 tells a model that students on slow-starting modules resemble
students who missed a deadline. So `assessment_due` is its own flag, and
`submitted` is **NaN when nothing was due** — genuinely unknown, not false.

At day 28 only 17 assessments fall due across 16 of the 22 module-presentations,
so 18.1% of students have nothing due. This is not a rare edge case.

### Ordinal variables get integers, and keep their NaN

`imd_band` and `highest_education` have a real order. Encoding them as unordered
categories throws it away and forces a model to rediscover it. They become
ordered integers via `schema.IMD_ORDER` / `EDUCATION_ORDER`.

`imd_band` is missing for 3.7% of the population. It stays NaN rather than being
assigned a middle value — a made-up band would be indistinguishable from a real
one downstream. The pipeline's imputer adds a missingness indicator so a model
can use "we don't know" as information.

### Forbidden columns are checked, not just avoided

`_assert_no_leakage` runs on every build and raises if `date_unregistration`,
`final_result`, `at_risk` or `withdrew` reach the feature set. `feature_columns()`
derives the model-input list rather than hardcoding it, so adding a feature does
not require remembering to update a second place.

---

## Results

Five-fold cross-validation grouped on `id_student`, PR-AUC, day-28 cutoff.
Base rate 0.441 — that is the floor.

| Feature set | Model | PR-AUC | Lift |
|---|---|---:|---:|
| The four quick features from notebook 03 | logistic regression | 0.6137 | 1.39 |
| Full set | logistic regression | 0.7359 | 1.67 |
| Full set | gradient boosting | **0.7657** | **1.74** |

The full feature set is a large improvement over the four-feature sketch — 0.61
to 0.77. Gradient boosting beats logistic regression, as expected on tabular
data, but not by much, which suggests most of the signal is close to additive.

These are **not** the Phase 3 numbers. There is no tuning here, no model ladder,
no proper comparison. This is a check that the features work.

### The leakage test passes

Shuffle the labels and rerun the entire pipeline unchanged:

```
shuffled labels, full set, gradient boosting: PR-AUC 0.4432
base rate of the shuffled labels:             0.4413
```

Sitting on the floor, as it must. If information were reaching the model by a
route we did not intend, this would score above the base rate. It is one line
and it rules out a whole category of mistake.

### The consistency features add nothing

`gap_cv`, `max_gap` and `mean_gap` were added on the hypothesis that *how evenly*
someone engages carries information beyond *how much*. Gradient boosting, with
and without them:

| Cutoff | Median active days | Without | With | Delta |
|---|---:|---:|---:|---:|
| day 28 | 13 | 0.7653 | 0.7657 | +0.0004 |
| day 84 | 27 | 0.8264 | 0.8261 | −0.0002 |
| day 168 | 47 | 0.9038 | 0.9040 | +0.0003 |

Noise at every cutoff, and not for want of history — by day 168 the median
student has 47 active days.

Two things worth saying about this null.

**The measure is not broken.** `gap_cv` correlates only −0.235 with log active
days, so it is carrying genuinely different information from frequency. A
weekday-concentration measure was tested alongside and correlated **+0.855** —
that one *is* a session counter in disguise, and was dropped rather than
included. The construct was measured cleanly; it just does not predict here.

**The resolution is wrong for the construct.** OULAD records no time of day at
all, only integer days. Regularity of the kind this was borrowed from is about
*when in the day* someone engages, and that is untestable on this data. The null
is about day-level spacing, and says nothing about the finer-grained claim.

The features are kept, marked as a measured null. A documented negative is worth
more than a quietly deleted branch, and they cost nothing. See
[`05-proposal-regularity.md`](05-proposal-regularity.md) for the study that
would test this properly.

---

## Carried into Phase 3

1. **Baseline to beat: 0.441** (the base rate) at day 28.
2. **Reference point: 0.766**, untuned gradient boosting on the full set.
3. **Region is the only remaining nominal categorical** — one-hot it inside the
   pipeline.
4. **Imputation strategy matters** for logistic regression and not at all for
   gradient boosting, which handles NaN natively. Worth reporting rather than
   hiding inside a pipeline.
5. **`scale_pos_weight` still looks unnecessary** at a 44% base rate. Test it,
   expect nothing.
