# Notebooks

Read these in order. Each one assumes the previous, and they build to a single
conclusion about how the problem has to be set up.

They're committed **with their outputs saved**, so you can read them on GitHub
without running anything. Every number and chart in them was produced from the
real OULAD data.

| | Notebook | What it settles |
|---|---|---|
| 01 | [Meeting the data](01-meet-the-data.ipynb) | The seven tables, what a row means in each, why joins go wrong, and how to read missing values |
| 02 | [What are we predicting?](02-the-target.ipynb) | The binary label, the base rate, why accuracy is out, why splits must group on student |
| 03 | [**Who are we predicting for?**](03-the-population-question.ipynb) | The population decision — argued, then **demonstrated with a real model** |
| 04 | [Exploratory data analysis](04-exploring-behaviour.ipynb) | Distributions, trajectories, activity types, demographics, assessments — and what they imply for feature engineering |

**If you read one, read 03.** It makes a design choice that *lowers* the
headline score, then fits the same model both ways to show that the
better-looking alternative is measuring something worthless.

## Running them

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

They need the data in `data/raw/` — see the main [README](../README.md).
Notebook 03 and 04 each take a minute or two, mostly aggregating the 10.6M-row
clickstream.

## How these relate to the rest of the repo

Notebooks are for **looking at things**. Pipeline logic lives in `src/oulad/`
and is covered by tests. The notebooks import from there rather than defining
their own versions, so there's one implementation of anything that matters.

Where a notebook writes throwaway code — the four quick features in 03, for
instance — it says so. Phase 2 builds the real version in `src/`.

## What they found

Things that came out of writing these, which weren't known beforehand:

- **The base rate is 52.8%**, not the ~33% the project was planned around. The
  positive class is the *majority* class, so this is not a rare-event problem
  and advice written for one doesn't transfer.
- **`imd_band` ships one category without its percent sign** (`10-20` where
  every other band has `10-20%`), affecting 3,516 enrolments — about 11% of the
  dataset. It fails silently: any code matching against a clean list of band
  names just drops those students. Now corrected at load time.
- **Withdrawal is not only an early-course phenomenon.** Median day 27, but the
  75th percentile is day 109.
- **The naive population framing scores PR-AUC 0.790 against 0.613** for the
  correct one — and leans 3× harder on "days since last click" to get there.
