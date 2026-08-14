# Leakage, and the population question

**Leakage** is when information reaches the model that would not have been
available at the moment the prediction is really made. It is the defining
failure mode of applied machine learning: it does not produce an error, it
produces an *excellent score*, and the score evaporates the moment the model
meets real data.

The reason it is so dangerous is that every instinct points the wrong way. A bug
that crashes gets fixed. A bug that makes your PR-AUC jump from 0.55 to 0.91
gets celebrated, written up, and deployed.

So the working rule for this project:

> **A result that looks too good is a bug report until proven otherwise.**
> At a day-28 cutoff, PR-AUC much above ~0.85 should send you hunting, not
> celebrating.

This document lists the specific traps in OULAD, in rough order of how easy they
are to fall into.

---

## Trap 1: `date_unregistration` is the answer written down

In `studentRegistration`, `date_unregistration` is non-null **if and only if**
the student withdrew. Since "Withdrawn" is part of our positive class, feeding
this column to a model hands it roughly half the label directly.

A model with this feature will score near-perfectly and has learned nothing.

**Rule:** `date_unregistration` may be used to *construct* the label and to
*define the population* (below). It must never appear in the feature matrix.

`src/oulad/labels.py` carries the column deliberately and documents this; the
feature builder must drop it.

---

## Trap 2: the population — the one this dataset punishes hardest

This is the trap that the obvious framing walks straight into, and it is worth
understanding properly because it changes what the project *is*.

Every date in OULAD is an integer offset from the start of the course. Suppose
we make our prediction on day 28. Now consider a student whose
`date_unregistration` is 10.

They left on day 10. We are "predicting" on day 28. There is nothing to predict —
they are already gone, and the university already knows it.

Two things follow, one obvious and one not.

**The obvious one:** it is not a prediction. Scoring it inflates every metric
with cases that were never uncertain.

**The one that actually does the damage:** that student generated no clicks
after day 10. So in a feature set built from the first 28 days, they show up as
someone with three weeks of total silence. Any model finds this instantly. "No
activity in the last fortnight" becomes the dominant rule, and it is right almost
every time — on students who have already withdrawn.

The model's headline number goes up. Its usefulness goes to zero. It has learned
to detect students who have already left, which a `WHERE date_unregistration IS
NOT NULL` query does perfectly, for free, and which helps nobody. Meanwhile the
signal we actually wanted — the still-enrolled student whose engagement is
quietly sliding — is a rounding error in the loss function and gets ignored.

### The fix

Define the population by the cutoff:

> Among students **still enrolled on day C**, who will go on to fail or withdraw?

`labels.eligible_at_cutoff(labels, C)` implements this. It excludes anyone whose
`date_unregistration <= C`.

Note the `<=`: someone who unregisters on the cutoff day itself is gone by the
time the model runs that evening.

### What this costs, and why it is worth it

Scores go *down*. The base rate falls, the easy cases are gone, and the remaining
problem is harder. That is the correct outcome — the earlier number was measuring
something we did not want.

It also makes the project more interesting rather than less. The cutoff sweep
stops being a simple "later is better" curve and becomes a real trade-off:

| Moving the cutoff later | Effect |
|---|---|
| More days of behaviour per student | Prediction gets easier |
| More students have already withdrawn | Fewer people left to help |
| Remaining students are the ambiguous ones | Prediction gets harder |
| Less time to act on the prediction | Less useful even when correct |

Only the first of those helps. The headline result of the project is where those
forces balance — and that is a genuinely useful thing to be able to tell a
university, in a way that "PR-AUC rises with more data" is not.

Phase 1 prints this table. Look at it before doing anything else.

### The honest alternative

Predicting withdrawal *timing* — a survival model, "how long until this student
leaves?" — handles the same structure differently and is arguably the more
natural fit for the data. It is out of scope for a first project, but it is the
right answer to "what would a specialist do here", and it is worth saying so in
the writeup rather than pretending the binary framing is the only option.

---

## Trap 3: any event dated after the cutoff

The general form of trap 2. If the cutoff is day 28, then:

- no `studentVle` row with `date > 28`
- no `studentAssessment` row with `date_submitted > 28`
- no assessment whose due date is after 28
- no aggregate computed over the full course and then filtered

The last one is the subtle version. Computing each student's total clicks across
the whole presentation and *then* restricting to day-28 rows does not undo the
leak: the aggregate already saw the future.

**Rule:** filtering happens first, at the event tables, inside
`build_features(cutoff_day)`. Everything routes through that one function so
there is exactly one place where this can go wrong, and it can be tested.

---

## Trap 4: random train/test splits

32,593 enrolments, but fewer distinct students — people repeat modules and take
several courses. A random split by row puts the same person on both sides.

The model then has an easier job than the one we are measuring: it can recognise
*this individual* rather than learn the general pattern, and the test score
stops estimating performance on students the model has never seen — which is
the only situation it will ever actually be used in.

**Rule:** `GroupKFold` grouped on `id_student`. Always.

---

## Trap 5: preprocessing fitted outside the fold

Scaling features by the mean and standard deviation of the *whole* dataset, then
cross-validating, leaks test-fold information into training. The effect is
usually small, but it is free to avoid.

**Rule:** every transform lives in a `sklearn.pipeline.Pipeline`, so that
`cross_val_score` refits it within each fold.

---

## Trap 6 (later): temporal leakage

OULAD spans 2013 and 2014. Training on a random mix of both and testing on the
same mix asks an easier question than reality does, because in deployment you
always train on the past and predict the future.

The honest test — train on 2013 presentations, predict 2014 — will score worse.
That drop is not a failure, it is the measurement. Explaining it (course content
changed? cohort changed? platform changed?) is one of the more interesting parts
of the project.

---

## The cheapest check you can run

Shuffle the labels and run the entire pipeline unchanged.

A correct pipeline scores at the base rate on shuffled labels, because there is
nothing left to learn. If it scores meaningfully better, information is reaching
the model through a route you did not intend, and the bug is in the code.

This takes one line and catches a whole category of mistakes. Run it whenever a
number surprises you.
