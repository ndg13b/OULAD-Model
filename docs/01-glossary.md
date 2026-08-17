# Glossary

Terms used in this repo, defined on first use and collected here. Nothing in the
code or docs should use a term that is not either defined inline or listed here.

Ordered roughly by when you meet them, not alphabetically.

---

## Data shape

**Row / observation / instance / example / sample** — five words for the same
thing: one unit you make a prediction about. Different books and libraries pick
different ones. In this project one row is one *enrolment*: a particular student
on a particular course in a particular term.

**Feature / predictor / independent variable / covariate / input / X** — again
all the same thing: a column used to make the prediction. "Feature" is the ML
word, "predictor" and "covariate" the statistics words. `scikit-learn` calls the
matrix of them `X`.

**Target / label / outcome / dependent variable / response / y** — the thing
being predicted. Here: whether the student ends up failing or withdrawing.

**Grain** — what one row of a table *means*. `studentInfo` has one row per
(student, course, term). `studentVle` has one row per (student, page, day). If
you join two tables whose grain differs without thinking about it, rows multiply
silently and every subsequent number is wrong. No error is raised. This is the
most common way to quietly ruin an analysis.

**Feature engineering** — turning raw records into columns a model can use.
10.7M clickstream rows are not a feature; "clicks in the first 14 days" is.
Most of the work and most of the judgement in a tabular project lives here.

---

## The prediction task

**Supervised learning** — you have examples where the answer is known, and you
want a rule that produces the answer for new cases. "Supervised" because the
known answers supervise the learning. The alternative, *unsupervised* learning,
looks for structure with no answer key (clustering, for instance).

**Classification vs regression** — classification predicts which category
something falls into; regression predicts a number. Predicting pass/fail is
classification. Predicting a final exam score would be regression.

**Binary classification** — classification with exactly two categories,
conventionally coded 1 and 0.

**Positive class** — the category coded 1. It does not mean "good"; it means
"the thing being detected". Here the positive class is *at risk*, because that
is what we want to find. Which class you call positive changes what precision
and recall mean, so it is worth stating explicitly.

**Base rate / prevalence / class prior** — the proportion of rows that are
positive. In OULAD 52.8% of enrolments end in Fail or Withdrawn, so the base
rate is 0.528. This is the number every model result must be compared against,
because you can achieve it without a model.

**Class imbalance** — when one class is much rarer than the other. It matters
because it makes accuracy meaningless (see below) and because some algorithms
under-fit the rare class. **This project does not actually have it**, which is
worth knowing, because a lot of advice about early-warning models assumes rare
positives and does not apply here.

---

## Evaluating a model

**Accuracy** — the fraction of predictions that are correct. Not used here, for
two reasons.

It forces you to pick a threshold, and the threshold is a deployment decision
(how many students can a tutor actually contact?) rather than a property of the
model. And it treats both mistakes as equally bad: missing an at-risk student
and wasting a tutor's time on a student who was fine are not the same error, but
accuracy counts them identically.

The textbook objection — that on rare-event data a model predicting "no" every
time scores brilliantly — does not apply here, because at a 52.8% base rate that
trick barely beats a coin flip. It is still the wrong metric; just not for the
usual reason.

**Confusion matrix** — the 2×2 table of what happened:

|                     | actually at risk | actually fine |
|---------------------|------------------|---------------|
| **flagged**         | true positive    | false positive |
| **not flagged**     | false negative   | true negative  |

Every classification metric is some ratio of these four cells.

**Precision** — of the students you flagged, what fraction really were at risk.
`TP / (TP + FP)`. Low precision means you are wasting tutor time on students who
were going to be fine.

**Recall / sensitivity / true positive rate** — of the students who really were
at risk, what fraction did you flag. `TP / (TP + FN)`. Low recall means you are
missing the people you exist to help.

**The precision/recall trade-off** — you can always raise one by lowering the
other, by moving the threshold at which you flag someone. Reporting one without
the other is meaningless.

**Threshold** — most models output a score between 0 and 1, not a decision. The
threshold is the cut you apply to turn the score into "flag" or "do not flag".
It is a *deployment* choice driven by how much tutor time exists, not a property
of the model, which is why we mostly evaluate the ranking rather than a
particular threshold.

**PR-AUC / average precision** — the area under the precision-recall curve: a
single number summarising performance across all possible thresholds. A useless
model scores about the base rate; a perfect one scores 1.0. This is our headline
metric because it focuses on the positive class and is not flattered by a large
easy negative class.

**ROC-AUC** — area under the receiver operating characteristic curve. Also
threshold-free, but a useless model scores 0.5 regardless of imbalance, which
makes it look reassuring on rare-event problems where PR-AUC would reveal the
model is weak. Worth reporting alongside; not worth leading with.

**Precision@k** — of the top *k* students by predicted risk, what fraction are
really at risk. This is the metric that matches how the system would actually be
used: a tutor has time for twenty students, so what matters is the quality of
the top twenty, not the whole curve.

**Calibration** — whether a predicted probability means what it says. If you
take every student the model scored 0.7 and 70% of them are at risk, the model
is calibrated. A model can rank perfectly and still be badly calibrated, which
matters as soon as anyone reads a score as a probability.

---

## Fitting and validating

**Training set / test set** — the data used to fit the model, and the held-out
data used to estimate how it performs on cases it has not seen.

**Cross-validation (CV)** — rather than one train/test split, cut the data into
*k* parts, train on k−1 and test on the remaining one, rotate, and average. Uses
all the data for both purposes and gives a sense of how much the estimate varies.

**Fold** — one of those *k* parts.

**Grouped splitting** — splitting so that all rows belonging to the same group
land on the same side. Here the group is `id_student`: the same person appears
in multiple enrolments, and if they appear in both training and test the model
can recognise the individual rather than learn the pattern. `GroupKFold` does
this.

**Overfitting** — learning the training data's noise rather than its signal.
Symptom: excellent training score, poor test score.

**Underfitting** — the model is too simple to capture the pattern. Symptom: both
scores poor.

**Hyperparameter** — a setting you choose before fitting (tree depth, learning
rate), as opposed to a parameter the model learns from data (a regression
coefficient).

**Validation set** — a third split, used for choosing hyperparameters. Necessary
because if you pick settings by looking at the test set, the test set has
informed the model and no longer gives an honest estimate.

---

## Leakage

**Leakage** — when information reaches the model that would not be available at
the moment the prediction is actually made. It produces excellent scores that
evaporate in deployment. It is the single most common serious error in applied
ML, and it is usually invisible unless you go looking. This project has three
specific traps; they have their own document, `02-leakage.md`.

**Target leakage** — a feature that is a consequence of the outcome rather than
a cause or correlate of it. `date_unregistration` is the example here: it is
non-null exactly when the student withdrew.

**Train/test contamination** — the test set influencing the training process.
Scaling features using the mean of the whole dataset before splitting is the
classic version: the training fold has now seen the test fold's mean.

**Pipeline** — in `scikit-learn`, an object chaining preprocessing steps and a
model into one estimator. It matters because it makes preprocessing get fitted
*inside* each cross-validation fold, which is what prevents contamination. This
is why the project uses one from the start rather than adding it later.

---

## Models

**Baseline** — the simplest thing that could work, fitted first so that every
later result has something to be compared against. Ours predicts the majority
class every time.

**Logistic regression** — despite the name, a classification model. Fits a
weighted sum of the features and squashes it to a probability. Its virtue is
that the weights are readable: you can say what the model thinks and check
whether it is plausible.

**Decision tree** — a flowchart of yes/no splits. Individually weak and prone to
overfitting; the building block for the two below.

**Random forest** — many decision trees, each fitted on a random subset of rows
and columns, predictions averaged. The randomness makes their errors partly
independent, so averaging cancels some of them out.

**Gradient boosting** — trees fitted in sequence, each one trained to correct
the errors the previous ones are still making. Usually the strongest option on
tabular data of this size, which is why the project ends there rather than with
a neural network.

**XGBoost** — a fast, widely used gradient boosting implementation.

**Regularisation** — a penalty on model complexity, used to combat overfitting.
In logistic regression it shrinks coefficients toward zero.

**SHAP (SHapley Additive exPlanations)** — a method for attributing a single
prediction to the features that drove it, borrowed from cooperative game theory.
It tells you what the *model* is doing. It does not tell you what causes what in
the world, and the distinction has to be maintained carefully in the writeup.
