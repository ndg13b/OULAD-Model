# Proposal: testing the regularity–retention hypothesis on OULAD

**Status:** proposal only. Nothing here has been run.
**Where it belongs:** the `adherence` repo, not this one. OULAD would be the
dataset; the code and the claim live over there.

---

## 1. The question

From `adherence/docs/OVERVIEW.md`, the third of three claims:

> **Retention** — regularity predicts who keeps going.
> **Open.** Two attempts, both on data that turned out unable to answer it.

The hypothesis: *people who engage on a more regular schedule are more likely to
still be engaging months later, separately from how often they engage.*

Two designs on FitRec running data returned null — hazard ratio 0.955 (95% CI
0.827–1.103) frozen-at-baseline, null at every memory setting for the
time-varying version. The write-up is explicit that this is not evidence against
the hypothesis, because the data could not test it.

## 2. Why OULAD is worth trying

The FitRec write-up names three reasons that dataset could not answer the
question. OULAD addresses all three.

| Why FitRec failed | OULAD |
|---|---|
| **Outcome is leaving a platform, not stopping the behaviour.** Data ends as Strava displaced Endomondo; app-switching dilutes any real effect | Outcome is **institutional withdrawal** — recorded by the university, with an exact date. Nobody withdraws from a degree because a competitor app launched |
| **No true enrolment date.** First workout in the file is where sampling opens, not where behaviour began — and habit formation happens at the start, invisibly | **Day 0 is the course start**, for everyone. `date_registration` gives the sign-up date too. The formation window is fully observed |
| **Severe selection.** Inclusion required 20+ runs over 120+ days, so the cohort had already persisted | **Every enrolled student is in from day 0**, including 1,226 who never click once. No survivorship filter |

The event structure also fits what `survival.py` wants:

- 32,593 enrolments, 28,785 distinct people
- **10,156 withdrawals with exact dates** (uncensored events)
- **22,437 right-censored** at course end
- Median withdrawal day 27; 75th percentile day 109 — events spread across the
  presentation rather than bunched
- Clean grouping variable (`id_student`) for the repeat-enrolment problem

That is an unusually clean survival dataset, and it is public.

## 3. The disqualifying limitation, stated up front

**OULAD has no time of day.** `studentVle.date` is an integer day, range −25 to
269. Verified: no fractional values, no timestamp column, nothing to recover it
from.

So of the package's four measures:

| measure | computable on OULAD? |
|---|---|
| `timing_consistency` | **No** — needs time within the day |
| `anchor_precision` | **No** |
| `timing_bits` | **No** |
| `weekday_regularity` | Technically yes, but see below |

This means **OULAD cannot test the package's distinctive claim.** The result
that makes the score interesting — Cohen's *d* of 4.46 against SRM's 0.77 in the
sub-20-minute band — is about resolution that OULAD simply does not have. Any
study here tests a day-resolution shadow of the construct.

Two specific traps:

**Do not synthesise midnight timestamps and read `timing_consistency`.** Every
event would land at the same within-day phase and the score would come back
near-perfect for everyone. That is an artefact of the encoding, not a finding.

**Do not use weekday regularity.** Measured on OULAD at day 28, weekday
concentration correlates **+0.855** with log active-days. That is the exact
failure mode the project already identified in the Sleep Regularity Index
(−0.94, "a session counter wearing a regularity label"). It is mechanical: with
ten active days you cannot spread across seven weekday phases. Gap-spacing CV is
much cleaner at **−0.235**, comparable to the package's own −0.07.

## 4. Evidence already collected, which lowers the prior

Phase 2 of this project added gap-spacing regularity (`gap_cv`, `max_gap`,
`mean_gap`) to a 45-feature model and measured its marginal contribution.
Gradient boosting, five-fold CV grouped on student:

| cutoff | median active days | without regularity | with | delta |
|---|---:|---:|---:|---:|
| day 28 | 13 | 0.7653 | 0.7657 | +0.0004 |
| day 84 | 27 | 0.8264 | 0.8261 | −0.0002 |
| day 168 | 47 | 0.9038 | 0.9040 | +0.0003 |

**Noise at every cutoff**, and not for want of history — by day 168 the median
student has 47 active days.

This is relevant but not decisive, because it is a different question:

- It asks whether regularity improves a **binary classifier's ranking**, not
  whether it predicts **time to event**.
- Regularity was competing against 40+ correlated behavioural features that
  already encode consistency indirectly (`active_days`,
  `days_since_last_activity`, weekly click columns, `click_slope`,
  `last_week_share`). A Cox model with regularity and frequency as the only two
  covariates is a far more focused test.
- Survival framing uses *when* someone left, which the classifier discards.

So: worth running, with a lowered prior. If the proposal is run and comes back
null, the two results together would be reasonably strong evidence that
day-resolution regularity does not predict retention — while still saying
nothing about the sub-hour construct.

## 5. Proposed design

### Cohort

All enrolments with a run-in window fully observed:

- Run-in: days 0–56. Long enough for a usable gap-based score, short enough to
  leave most of the course as follow-up.
- **Include only students still enrolled at day 56** — anyone who left during
  the run-in has no follow-up period and their score is measured on a truncated
  history. (Same population logic as `docs/02-leakage.md`.)

Measured, not estimated: **26,522 enrolments, of which 4,001 withdraw after day
56** — a 15.1% event rate.

### Exposure

Scored on the run-in window only:

- `gap_cv` — coefficient of variation of gaps between active days (**the
  exposure of interest**)
- `active_days`, `total_clicks` — the frequency covariates the effect must
  survive adjustment for

Standardise `gap_cv` within cohort so the hazard ratio reads per SD, matching
how the FitRec results were reported.

### Outcome

Time from day 56 to withdrawal, in days. `date_unregistration` gives the event
time exactly. Students who pass, fail, or reach course end without withdrawing
are **right-censored** at `module_presentation_length`.

Note this differs from the main project's label: here the event is *withdrawal
specifically*, not Fail-or-Withdrawn, because only withdrawal has a date. Fail
is a state discovered at the end and has no time-to-event.

### Model

Cox proportional hazards — `adherence.survival` for the primary fit, `lifelines`
to confirm and to check the proportional-hazards assumption, which the minimal
implementation does not test.

```
h(t) = h₀(t) · exp(β₁·gap_cv + β₂·log(active_days) + β₃·log(total_clicks+1) + controls)
```

Controls: `education_ord`, `imd_band_ord`, `age_ord`, `num_of_prev_attempts`,
`studied_credits`, and module-presentation as a stratum (courses differ in
length and difficulty, so a shared baseline hazard is wrong).

**Clustering:** 3,808 enrolments are repeat appearances by the same person. Use
a robust sandwich variance clustered on `id_student`, or restrict to one
enrolment per student. Do not ignore it.

### Pre-specified analysis

State before running:

- **Primary:** β₁ ≠ 0 for `gap_cv`, adjusted for frequency, with the CI reported
  whatever it says.
- **Sanity check first:** does frequency predict retention at all? On FitRec it
  did not (HR 0.922, p = 0.33), which was the tell that the outcome was broken.
  If `active_days` shows nothing on OULAD either, stop and suspect the outcome
  before interpreting anything about regularity.
- **Power:** with 4,001 events, detecting HR 0.90 per SD at α = 0.05 has power
  well above 0.95 (Schoenfeld: ~1,050 events needed). `adherence.simulate` can
  confirm. The study is not underpowered; a null would be a real null.
- **Secondary:** time-varying version, re-scoring every 28 days on prior history
  only, matching the second FitRec design.

## 6. Threats to validity

**Construct validity — the serious one.** The package requires that *the person
chose the time*: "if a protocol assigns a slot, the score measures compliance
with an instruction instead." OULAD engagement is partly externally scheduled —
assignment deadlines, weekly material releases, tutorial dates. Students choose
which days they study, but within a structure that pushes everyone toward
similar rhythms. This sits somewhere between a self-organised habit and
compliance with a timetable, and closer to the latter than FitRec running was.
It should be stated as a limitation, not waved at.

**Short horizon.** A course presentation is ~260 days. The hypothesis is about
habits persisting over months to years. FitRec had a median 702 days of
follow-up; here the ceiling is ~200 days post-run-in.

**Withdrawal is administrative.** `date_unregistration` is when paperwork was
processed, which may lag disengagement by weeks. This adds noise to event times
and biases toward the null.

**Not the target construct.** Repeating for emphasis: day-resolution spacing is
not the sub-hour timing regularity the package measures. A null here does not
refute the hypothesis the package is actually about.

## 7. What each outcome would mean

| Result | Reading |
|---|---|
| **Positive**, β₁ < 0, survives frequency adjustment | Regularity predicts retention at day resolution, on data without FitRec's three defects. Genuinely novel, and worth publishing |
| **Null**, tight CI around 1.0, frequency *does* predict | Day-resolution regularity does not predict retention. Combined with the Phase 2 null, reasonably strong. Leaves the sub-hour claim open and sharpens what data would be needed |
| **Null**, and frequency also null | Suspect the outcome, as on FitRec. Investigate before concluding anything |

The second outcome is the most likely given the Phase 2 result, and it is still
worth having: "tested on data that could answer it, at the resolution available,
and found nothing" is a much better position than "never tested."

## 8. Effort

Perhaps a day's work. The data is already in this repo as parquet, the run-in
scoring is a small variation on `oulad.features._gap_features`, and
`adherence.survival` already implements Efron-tied Cox with the right ties
handling for whole-day event times.

The write-up would take longer than the analysis, which is usually the sign of a
well-posed study.

## 9. What would actually test the real hypothesis

Since OULAD cannot: the dataset needed has **timestamped** engagement events at
sub-hour resolution, a **true enrolment date**, an outcome that is **stopping the
behaviour** rather than leaving a platform, and **no survivorship filter** on
entry. Candidates worth scoping — language-learning apps with full event logs,
medication-adherence trials with electronic pill-bottle caps, or an institutional
LMS that retains timestamps rather than daily rollups. The OU itself holds the
underlying timestamped data; OULAD is a daily aggregate of it.

That last point is worth chasing before anything else. If a timestamped
extract of the same population were obtainable, it would satisfy every
requirement at once.
