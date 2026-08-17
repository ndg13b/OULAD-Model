# Project design

What this project is trying to answer, what was decided, and why.

---

## 1. The question

> Using only information available in the first *N* weeks of a course, how well
> can we identify students who will fail or withdraw — and which behavioural
> signals carry that information?

The intent is an **early-warning system**: something that hands a tutor a
ranked list of students to contact while there is still time to act. That
framing, rather than "classify outcomes accurately", drives every decision
below.

Two consequences worth stating up front, because they rule out choices that
would otherwise look reasonable:

- **A prediction made too late is worthless even if it is correct.** Accuracy
  and actionability trade against each other, and measuring that trade-off is
  the point of the project, not a caveat on the side.
- **The output is a ranking, not a verdict.** Nobody acts on all 32,593
  students. A tutor has time for a handful. So the metric has to reward getting
  the *top of the list* right, and threshold-free ranking metrics are the ones
  that matter.

---

## 2. The dataset

**OULAD** — Open University Learning Analytics Dataset. 32,593 students across
22 module-presentations, 2013–2014. Seven linked CSVs, about 45 MB.

- Source: <https://analyse.kmi.open.ac.uk/open_dataset>, also UCI ML Repository
  dataset 349 (`pip install ucimlrepo`)
- Licence: CC BY 4.0
- Citation: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University
  Learning Analytics dataset. *Scientific Data*, 4, 170171.

| File | Grain | Role |
|---|---|---|
| `studentInfo` | student × module × presentation | Demographics + `final_result` |
| `studentVle` | student × site × day (~10.7M rows) | Clickstream — main feature source |
| `vle` | site | Maps site → `activity_type` |
| `studentAssessment` | student × assessment | Scores, submission dates |
| `assessments` | assessment | Type, weight, due date |
| `studentRegistration` | student × presentation | Registration / unregistration dates |
| `courses` | module × presentation | Module length |

All dates are integer days relative to the start of the presentation. Negative
values are before teaching began.

`src/oulad/schema.py` encodes this table as something the loader can check.

---

## 3. Decisions

### Target: binary, `{Fail, Withdrawn} = 1`

`final_result` has four values. We collapse to:

```
at_risk = 1   Fail, Withdrawn
at_risk = 0   Pass, Distinction
```

**Why binary and not four-class.** The intervention is binary — a tutor either
contacts a student or does not. A four-class model would have to be collapsed to
a decision anyway, and it splits the training signal across categories that the
use case does not distinguish. Four-class is a reasonable later extension.

**Why Fail and Withdrawn together.** Both describe someone who did not get the
qualification they enrolled for, and both warrant the same first response. They
are not the same phenomenon — withdrawal is an event with a date, failure is a
state discovered at the end — and separating them later is one of the more
interesting extensions.

**Alternatives considered and rejected for now:**

| Framing | Why not now |
|---|---|
| Predict final exam score (regression) | Only defined for students who sit the exam, which selects on the outcome |
| Withdrawal only | Ignores the enrolled-but-failing student, who is arguably the more helpable case |
| Time-to-withdrawal (survival analysis) | The best fit for the data's structure, and genuinely the specialist's answer — but it needs machinery beyond a first project. Flagged in the writeup rather than pretended away |
| Four-class | Later extension |

### Population: students still enrolled at the cutoff

At cutoff day *C*, the population is students who had **not** unregistered on or
before *C*.

This is the least obvious decision in the project and the most consequential.
Full reasoning in [`02-leakage.md`](02-leakage.md); the short version is that a
student who left on day 10 is not a prediction on day 28, and including them
teaches the model to detect people who have already gone — which scores
brilliantly and helps nobody.

### Feature extraction: one cutoff-aware function

Everything routes through `build_features(cutoff_day)`, which filters *all*
event tables to `date <= cutoff` before aggregating.

One function, one place to get it wrong, one place to test. Primary cutoff:
**day 28**.

### Splitting: grouped on `id_student`

`GroupKFold(n_splits=5)`. Students repeat across presentations; a random row
split puts the same person in train and test.

### Metric: PR-AUC, plus precision@top-10%

**Correction to an assumption worth flagging.** This project was planned on the
expectation of a ~1/3 positive rate and "imbalanced data" as the justification
for PR-AUC. The real number is **52.8%** — Fail (21.6%) plus Withdrawn (31.2%).
The positive class is the *majority*. After restricting to students still
enrolled at day 28 it falls to 44.1%, which is still roughly balanced.

So the imbalance argument does not apply, and any reasoning that depended on it
needs redoing. PR-AUC survives, but for a different reason:

- **PR-AUC**, because we act on the *top of the ranking*. Precision and recall
  both ignore true negatives, which is right here: correctly predicting that a
  passing student will pass earns a tutor nothing. ROC-AUC counts those, so it
  rewards work we do not care about.
- **Precision@top-10%**, because it is literally the deployment question: of the
  students we have time to contact, how many needed contacting?

Report ROC-AUC alongside for comparability with the published literature, which
mostly uses it.

Accuracy is still reported nowhere, but the usual reason is now the wrong one.
At a 52.8% base rate the majority-class trick barely beats a coin flip, so
accuracy is not *inflated* — it is just answering a question nobody asked. It
forces a threshold, and it treats "missed an at-risk student" and "wasted a
tutor's time" as equally bad, which they are not.

### Models: a ladder, on identical splits

1. **Majority-class baseline** — establishes the floor
2. **Logistic regression** — readable coefficients; the interpretability benchmark
3. **Random forest** — first non-linear model
4. **Gradient boosting (XGBoost)** — tuned

Each step must justify itself against the one below on the *same folds*. A model
that beats the previous one because it got a luckier split has demonstrated
nothing.

**No deep learning.** Gradient boosting outperforms neural networks on tabular
data at this scale. Understanding *why* is more valuable than reaching for a
bigger hammer.

### Pipelines from the start

`sklearn.pipeline.Pipeline` from day one, so preprocessing is fitted inside each
CV fold rather than before splitting.

### Ordinal encoding for ordinal variables

`imd_band` and `highest_education` have a meaningful order. Encoding them as
unordered categories throws that away; encoding them as ordered integers keeps
it. Orders are recorded in `schema.py`.

---

## 4. Phases

Worked in order. Each finishes before the next begins.

**Phase 1 — Load, inspect, label.** *(done)*
Load all seven files, validate shapes, document grain and missingness, build the
label, quantify the cutoff/population trade-off.
→ `scripts/phase1_inspect.py`

**Phase 2 — Feature engineering.** *(done — see `06-phase2-features.md`)*
`build_features(cutoff_day)`, returning one row per enrolment:
- *Volume*: total clicks, active days, clicks per active day
- *Trajectory*: clicks per week, week-over-week slope, days since last activity
- *Diversity*: clicks by `activity_type`, distinct sites touched
- *Timing*: `date_registration` relative to course start
- *Assessment*: first TMA score, submitted or not, days early/late
- *Demographics*: from `studentInfo`

Expect 40–80 columns.

**Phase 3 — The model ladder.** *(next)* The four models above, identical
splits, honest comparison.

**Phase 4 — The analysis that makes it a project.**
- **Cutoff sweep** — rerun everything at days 7, 14, 28, 56, 84. Plot
  performance against cutoff. This curve is the headline result.
- **Temporal generalisation** — train on 2013, test on 2014. Performance will
  drop; explaining why is a real finding.
- **SHAP** — global importance, dependence plots, individual explanations.
  These are associations, not mechanisms; the language stays disciplined.
- **Fairness** — does performance differ across `imd_band` and `age_band`? A
  model that systematically misses at-risk students from deprived areas is worse
  than no model, because it launders inaction as evidence.

---

## 5. What "done" looks like

A short writeup answering:

- How early can we usefully predict?
- What behaviour predicts risk?
- How much accuracy is traded for actionability?
- Where does the model fail, and on whom?

The last question is not a formality. A model deployed on students has to be
accountable for who it misses.
