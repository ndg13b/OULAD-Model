# OULAD Early-Warning Model

Predicting which Open University students will fail or withdraw, using only the
behavioural data available in the first few weeks of a course.

This repo is built to be **read as well as run**. Design decisions are written
down and argued for rather than assumed, terms are defined before they are used,
and every script prints an explanation of what it is doing and why.

| Document | What's in it |
|---|---|
| [**`docs/03-what-the-data-actually-is.md`**](docs/03-what-the-data-actually-is.md) | **Start here.** What the OU recorded, what we're building, and the one decision that determines whether it's useful. Assumes no ML background |
| [`docs/00-project-design.md`](docs/00-project-design.md) | The question, the decisions, and the alternatives that were rejected |
| [`docs/01-glossary.md`](docs/01-glossary.md) | Every term, defined. No jargon is used before it appears here |
| [`docs/02-leakage.md`](docs/02-leakage.md) | The ways this dataset will fool you, and the rules that prevent it |
| [`docs/04-phase1-findings.md`](docs/04-phase1-findings.md) | What the real data actually says — measured, not assumed |

**Status: Phase 1 complete** — data loading, schema validation, label
construction, and the cutoff/population analysis. Feature engineering is next.

---

## The problem in one paragraph

The Open University publishes anonymised records for 32,593 students: who they
are, what they clicked in the virtual learning environment, what they scored,
and how they finished (Pass, Fail, Withdrawn, Distinction). 52.8% fail or
withdraw — this is not a rare event, it is the majority outcome. If we could
spot them from their first few weeks of behaviour, a tutor could intervene while
it still mattered. The catch is that the signal that makes prediction easy — a
student who has stopped showing up — mostly arrives once it is too late to act.

Quantifying that trade-off is the project.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs the `oulad` package and deps
```

Phase 3 adds gradient boosting and model explanation:

```bash
pip install -e ".[models]"
```

## Getting the data

The CSVs are **not** in this repo — they are gitignored, as raw data should be.
Download them yourself:

- <https://analyse.kmi.open.ac.uk/open_dataset> (direct download), or
- UCI ML Repository dataset 349: `pip install ucimlrepo`

Then convert them to parquet:

```bash
python scripts/prepare_data.py --source ~/Downloads/oulad
```

**Why the conversion step.** `studentVle.csv` is ~10.7M rows and several hundred
MB. GitHub rejects any file over 100 MB, so the CSVs cannot be committed as-is.
Parquet stores data by column instead of by row and compresses it, which shrinks
the clickstream by roughly 10x — small enough to commit, and faster to load.
`oulad.load` prefers parquet automatically, so nothing else changes.

The script validates every file against `src/oulad/schema.py` before writing,
then reports which outputs are safe to commit:

```
data/raw/
├── assessments.parquet
├── courses.parquet
├── studentAssessment.parquet
├── studentInfo.parquet
├── studentRegistration.parquet
├── studentVle.parquet      # the big one, ~30 MB after conversion
└── vle.parquet
```

The `.csv` files stay gitignored; the `.parquet` files are committed.

Licence: CC BY 4.0. Cite Kuzilek, Hlosta & Zdrahal (2017), *Scientific Data* 4,
170171.

### Running without the real data

If you want to check the pipeline works before downloading anything:

```bash
python scripts/make_synthetic.py --out data/synthetic --students 4000
python scripts/phase1_inspect.py --data-dir data/synthetic
```

This generates a small fake dataset with the same schema, column vocabularies
and date conventions as the real thing. It exists to exercise the code and, more
usefully, to provide a case where the ground truth is known — see the module
docstring in `scripts/make_synthetic.py` for why that is worth having.

**No number produced from synthetic data is a finding about real students.**
Scripts print a warning when running against it.

---

## Running Phase 1

```bash
python scripts/phase1_inspect.py
```

Five sections:

1. **Load and validate** — every file checked against `src/oulad/schema.py`
2. **Grain and missingness** — what one row means in each table
3. **The outcome and the label** — the four-way outcome, collapsed to binary
4. **Who is still there** — the population question, and the table that shapes
   the rest of the project
5. **First look at signal** — do clicks separate the outcomes at all?

Section 4 is the one to read carefully. It is explained in
[`docs/02-leakage.md`](docs/02-leakage.md).

---

## Layout

```
├── src/oulad/           # library code — importable, testable
│   ├── paths.py         #   where things live on disk
│   ├── schema.py        #   expected shape of all seven tables
│   ├── load.py          #   readers with validation
│   └── labels.py        #   label construction + cutoff population
├── scripts/             # things you run
│   ├── phase1_inspect.py
│   └── make_synthetic.py
├── docs/                # the reasoning
├── tests/
├── data/raw/            # gitignored — put the CSVs here
├── data/interim/        # gitignored — cached aggregations
└── reports/             # figures and writeup
```

Logic lives in `src/`. Scripts and notebooks import from it and do not define
pipeline steps of their own, so anything worth keeping is testable.

---

## The one rule

> **A result that looks too good is a bug report until proven otherwise.**

At a day-28 cutoff, PR-AUC much above ~0.85 means go hunting for leakage, not
celebrating. [`docs/02-leakage.md`](docs/02-leakage.md) lists the six specific
ways this dataset will hand you a great score for nothing.
