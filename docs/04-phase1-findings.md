# Phase 1 findings — the real data

First run against the actual OULAD files. Everything here is measured, not
assumed. Reproduce with `python scripts/phase1_inspect.py`.

---

## Everything loads and matches the published description

| Table | Rows | Grain |
|---|---:|---|
| `studentInfo` | 32,593 | student × module × presentation |
| `studentRegistration` | 32,593 | student × module × presentation |
| `studentVle` | 10,655,280 | student × module × presentation × site × day |
| `vle` | 6,364 | site |
| `assessments` | 206 | assessment |
| `studentAssessment` | 173,912 | assessment × student |
| `courses` | 22 | module × presentation |

All seven passed schema validation: expected columns present, no unexpected
nulls, no rows duplicating the declared grain. Row counts match Kuzilek et al.
(2017) exactly.

## Missingness

| Column | Missing | Reading |
|---|---:|---|
| `studentRegistration.date_unregistration` | 69.10% | Not missing data — empty *means* "never withdrew". 30.9% non-null ≈ the 31.2% Withdrawn rate |
| `vle.week_from` / `week_to` | 82.39% | Most VLE pages are not tied to a specific teaching week |
| `assessments.date` | 5.34% | Mostly final exams with no fixed date |
| `studentInfo.imd_band` | 3.41% | Genuinely missing. Needs a decision in Phase 2 |
| `studentAssessment.score` | 0.10% | Negligible |
| `studentRegistration.date_registration` | 0.14% | Negligible |

The first row is the one to internalise: a null there is not absence of
information, it is the *most* informative value in the column. Which is exactly
why it cannot be a feature.

---

## The base rate is 52.8%, not ~33%

| Outcome | Count | Share |
|---|---:|---:|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |

**`at_risk` (Fail or Withdrawn) = 17,208 / 32,593 = 52.8%.**

This project was planned expecting roughly a third, with "imbalanced data" as
the stated reason for choosing PR-AUC. That premise is wrong: **the positive
class is the majority class.** This is not a rare-event problem.

Consequences, worked through in `00-project-design.md`:

- The imbalance justification for PR-AUC does not hold. PR-AUC is still the
  right headline metric, but because we act on the top of a ranking — not
  because positives are rare.
- Techniques aimed at imbalance (`scale_pos_weight`, resampling, threshold
  shifting) have much less to do here than planned. Test rather than assume.
- Accuracy is still the wrong metric, but not for the usual reason. At a 52.8%
  base rate, always predicting "at risk" barely beats a coin flip.
- Operationally, "the model flags 53% of your students" is not an intervention
  plan. This has to be a ranking, worked down as far as tutor time allows.

## Students repeat

28,785 distinct students across 32,593 enrolments — **3,808 enrolments belong to
someone who appears more than once.** Confirms that splits must group on
`id_student`.

---

## Withdrawal timing

| Percentile | Day |
|---|---:|
| 10th | −38 |
| 25th | −2 |
| 50th | **27** |
| 75th | 109 |
| 90th | 170 |

**2,676 of 10,156 withdrawals (26.3%) happen before day 0** — students who
registered and left before teaching began.

But note the 75th percentile: **withdrawal is not purely an early-course
phenomenon.** A quarter of withdrawals happen after day 109, well into the
second half. Any framing that treats dropout as something that happens in the
first few weeks and then stops is wrong.

## The population at each cutoff

| Cutoff | Still enrolled | % of all | Already left | Base rate |
|---|---:|---:|---:|---:|
| — (everyone) | 32,593 | 100.0% | 0 | 0.528 |
| day 7 | 29,178 | 89.5% | 3,415 | 0.473 |
| day 14 | 28,119 | 86.3% | 4,474 | 0.453 |
| **day 28** | **27,538** | **84.5%** | **5,055** | **0.441** |
| day 56 | 26,522 | 81.4% | 6,071 | 0.420 |
| day 84 | 25,724 | 78.9% | 6,869 | 0.402 |

At the primary day-28 cutoff, applying the still-enrolled rule removes 15.5% of
enrolments and drops the base rate from 52.8% to 44.1%.

**The base-rate change is the smaller half of the story.** The 5,055 removed
students are precisely those with weeks of empty clickstream, and they are all
positives. Left in, they would be the easiest cases in the dataset and would
teach the model that silence means failure — a rule that is correct about them
and useless about everyone else. Removing them is worth far more than the 8.7
percentage points suggests.

---

## There is signal in the clickstream

Among the 27,538 students still enrolled at day 28, using only clicks from day
28 or earlier:

| | Median clicks | Median active days |
|---|---:|---:|
| Not at risk | 298 | 16 |
| At risk | 152 | 9 |

Roughly a 2× difference in both volume and consistency. And:

| | Zero clicks in first 28 days |
|---|---:|
| Not at risk | 1.1% |
| At risk | 6.2% |

Clear separation, so the behavioural data carries real information. It does not
tell us how much survives once demographics and prior attempts are accounted
for — that is Phase 3's job.

Worth noticing that *active days* separates about as well as *total clicks*.
Consistency may matter as much as volume, which is a hypothesis Phase 2's
trajectory features are built to test.

---

## One data quirk

Nine enrolments have a non-null `date_unregistration` but a `final_result` of
`Fail` rather than `Withdrawn` (seven at day 0, two slightly before, one at day
166). Almost certainly administrative artefacts.

They do not need special handling: all nine are positives under our label either
way, and the eligibility filter removes them at any sensible cutoff. Noted so
that the count in the loader's consistency warning is not mistaken for a bug.

---

## Decisions carried into Phase 2

1. **Population**: students still enrolled at the cutoff.
   `labels.modelling_population()` applies this by default.
2. **Splitting**: `GroupKFold` on `id_student`. Confirmed necessary — 3,808
   repeat enrolments.
3. **Metric**: PR-AUC headline, precision@top-10% for the deployment question,
   ROC-AUC for comparability. Baseline to beat at day 28 is **0.441**.
4. **`imd_band` missingness** (3.41%) needs an explicit strategy — it is an
   ordinal variable, so it cannot simply be one-hot encoded with a "missing"
   category without losing the ordering.
5. **Revisit imbalance-motivated plans.** `scale_pos_weight` was on the Phase 3
   list; at a 44% base rate there may be nothing for it to correct.
