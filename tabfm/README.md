# Tabular foundation models

A hands-on companion to Christoph Molnar's *Tabular Foundation Models: A short
and opinionated guide*, using the OULAD data from the rest of this repo as the
real-world test case.

Read [`00-what-they-are.md`](00-what-they-are.md) first if the idea is new, then
work through the notebooks in order.

---

## The idea, in one page

Every model in the OULAD project so far was **trained**. You handed it 27,538
labelled rows, and it adjusted internal numbers until it predicted them well.
Logistic regression learned 45 coefficients. Gradient boosting grew a few
hundred trees. Training took seconds to minutes, and the result was a model that
existed only for this dataset.

A tabular foundation model does not do that.

It was trained **once, by someone else**, on millions of *synthetic* tabular
datasets — not on yours, and not on any real data at all. When you hand it your
data, it does not fit parameters to it. It reads your labelled rows as
**context**, the way a language model reads a prompt, and produces predictions
for your unlabelled rows in a single forward pass.

That is **in-context learning**: the model's weights never change, and the
"learning" happens inside one forward pass as a function of what you put in the
prompt. The idea is borrowed directly from language models, where showing a few
examples in the prompt changes the answer without any gradient step.

The practical consequences are strange if you are used to ordinary ML:

| | Ordinary model | Tabular foundation model |
|---|---|---|
| `fit()` | does the work — minutes | stores your data — milliseconds |
| `predict()` | cheap lookup | **does the work** — this is the expensive call |
| Hyperparameters | many, need tuning | few, mostly do not need tuning |
| More data | generally better | better *up to a hard ceiling*, then refuses |
| Where the knowledge lives | in the fitted parameters | in weights someone else trained |

The last row is the one worth sitting with. A tabular foundation model's prior —
everything it "believes" about how tabular data behaves — came from the
synthetic datasets its authors generated. When your data resembles that prior,
it is remarkably good with zero tuning. When it does not, you have limited
recourse, because there is nothing to tune.

## Why this is a genuinely new thing

For roughly fifteen years the consensus on tabular data was settled: gradient
boosting wins. Not neural networks, not anything clever — boosted trees, tuned
a bit. The OULAD project's own design doc says exactly this, and it was correct
when written.

Tabular foundation models are the first credible challenge to that, and they win
in a specific regime: **small to medium datasets, no tuning budget, tabular
structure**. On a few hundred to a few thousand rows they frequently beat a
tuned gradient boosting model while taking no tuning at all. That is a real
result and it is why the book exists.

What is *not* established is whether this holds at scale, on messy real data,
under distribution shift, or anywhere the synthetic prior does not reach. The
notebooks here test some of that on OULAD rather than taking it on faith.

## The two families

**TabPFN** (Prior Labs). The original and the one most people mean. PFN stands
for *Prior-data Fitted Network*. Versions have moved fast — see
[`01-setup.md`](01-setup.md) for what is current and what the limits are.

**TabICL** (INRIA and collaborators). Built for larger tables than TabPFN
originally handled, using a different architecture for encoding columns. Worth
comparing precisely because it makes different trade-offs.

## What is in here

| File | What it is |
|---|---|
| [`00-what-they-are.md`](00-what-they-are.md) | The concepts, with no code. Start here |
| [`01-setup.md`](01-setup.md) | Install, the licence, the GPU question, the access token |
| `10-first-contact.ipynb` | Run TabPFN on a small classic dataset, against real baselines |
| `20-what-fit-actually-does.ipynb` | Where the time goes, what the context window is, what breaks |
| `30-oulad-the-real-test.ipynb` | Against the tuned gradient boosting baseline on 45 real features |

## The honest summary before you start

Three things the marketing does not lead with, all covered properly in
[`01-setup.md`](01-setup.md):

1. **The good weights are non-commercial.** Fine for learning; a licensing
   conversation for anything else.
2. **It wants a GPU.** On CPU there is a hard row cap and it is slow.
3. **You need an account and a token** to download the weights at all.

None of these are reasons not to try it. They are reasons to know what you are
getting into before you build something on it.
