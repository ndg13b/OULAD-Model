# What a tabular foundation model actually is

No code. Read this once, then run `10-first-contact.py` and watch it behave the
way described here.

---

## Start from what you already know

In the OULAD project you fitted a logistic regression. Concretely, that meant:

1. You had 27,538 labelled rows.
2. The model started with 45 coefficients set to arbitrary values.
3. It adjusted them, repeatedly, until its predictions on *your* rows were as
   close as it could get.
4. What you kept was those 45 numbers. They existed nowhere before you ran
   `fit()`, and they describe this dataset and nothing else.

Gradient boosting is the same story with more moving parts: hundreds of trees,
each one grown to fix the errors of the ones before, all of them shaped by your
data.

This is what "training a model" means, and it is what essentially every machine
learning tutorial teaches.

## Now the strange part

A tabular foundation model does none of that.

It is a neural network — a transformer, the same architecture behind language
models — whose weights were fixed before you ever saw it. They were set by
training on **millions of synthetic tabular datasets**, generated from random
structural causal models. Not your data. Not anybody's real data.

When you hand it your 2,000 labelled rows and ask for predictions on 500
unlabelled ones, it does not adjust a single weight. Instead it puts all 2,500
rows into the network at once — the labelled ones with their labels attached,
the unlabelled ones without — and reads the answers off the output.

The labelled rows function as **context**. They are much more like a prompt than
like a training set.

## In-context learning

This is the term, and it is borrowed from language models on purpose.

When you show a language model three examples of a task in the prompt and it
does the fourth correctly, no training happened. The weights were identical
before and after. Something that *behaves* like learning took place inside a
single forward pass, driven entirely by what was in the context window.

Tabular foundation models apply that to tables. "Learning from your data"
collapses into one forward pass, and the thing doing the work is a network
pretrained to be good at *inferring the structure of an unfamiliar table from
examples*.

Which raises the obvious question: how can a model trained on fake data be good
at your real data?

## Why synthetic pretraining is not obviously mad

The bet is that real tabular datasets have shared structure: a handful of
features matter and the rest are noise, effects are often monotone, interactions
exist but are usually low-order, outliers happen, missing values are not random.

If you can generate synthetic datasets with those properties, in enormous
variety, a network trained across all of them learns something like "how tabular
data tends to work" — a **prior**. Meeting your dataset, it applies that prior
to whatever examples you give it.

In Bayesian terms this is approximating the posterior predictive distribution
over functions, given your data, under that prior. The "PFN" in TabPFN stands
for **Prior-data Fitted Network**, which is exactly this idea.

Two consequences follow, and they explain most of the observed behaviour:

**It is very good where the prior fits.** Small clean tables with moderate
feature counts are close to the synthetic distribution, and performance there is
genuinely remarkable with zero tuning.

**You cannot fix it where the prior does not fit.** If your data has structure
the prior never generated — a strange dependency, an adversarial pattern,
something pathological — there is no tuning knob. The prior is baked in. Your
recourse is to go back to gradient boosting.

## What this changes in practice

| | Ordinary model | Foundation model |
|---|---|---|
| `fit()` | does the work | stores the context |
| `predict()` | cheap | **does the work** |
| Hyperparameters | many, worth tuning | few, mostly not worth tuning |
| Preprocessing | scale, encode, impute | **do none of it** |
| More data | monotonically helps | helps up to a hard cap, then refuses |
| Failure mode | underfits or overfits, diagnosable | silently mediocre, few levers |

The preprocessing row surprises people most. Standardising your features — the
reflex of every tutorial — makes things *worse*, because the model was
pretrained on particular distributions and does its own normalisation. Same for
one-hot encoding and imputation. You hand it the raw table.

## Where the claim is strong, and where it is not

**Strong.** A few hundred to a few thousand rows, a few dozen features, ordinary
tabular structure, no tuning budget. Here these models routinely match or beat
tuned gradient boosting while taking seconds of your attention instead of hours.
For exploratory work and small scientific datasets this is a real change.

**Weak or unproven.** Large data, where gradient boosting keeps improving and
the foundation model hits its ceiling. Heavy distribution shift, where the prior
has nothing to say. High-stakes settings needing interpretability, where "the
transformer decided" is not an answer. Anything needing low-latency single-row
prediction, where the inverted cost structure bites hard.

There is also a deeper open question, which the book is right to press on:
**nobody can fully characterise when the prior fits.** With gradient boosting,
poor performance is diagnosable — look at the trees, the learning curves, the
residuals. With a foundation model you mostly observe that it did worse and
cannot say why.

## The thing to hold on to

The question is not "are foundation models better than gradient boosting?"
That has no answer.

The question is "on *my* data, at *my* size, with *my* tuning budget, which
wins?" — and that is answerable in about ten minutes with a fixed split and a
baseline you trust.

`30-oulad.py` does exactly that on the OULAD data, against the gradient boosting
baseline built in Phase 2 of this repo. Running it is more informative than any
benchmark table, because it is your data.

---

Next: [`01-setup.md`](01-setup.md) for installation and the three practical
things that will block you, then `10-first-contact.py`.
