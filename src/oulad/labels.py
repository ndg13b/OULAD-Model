"""Building the thing we are trying to predict, and deciding who we predict it for.

Two separate questions live in this file, and keeping them separate is the whole
point of the module:

1. **What is the label?** Which outcomes count as "at risk"?
2. **Who is in the population?** At a given moment in the course, which students
   are we actually making a prediction *about*?

Most introductions to machine learning only discuss question 1, because in the
textbook setting the population is just "all the rows". Here it is not, and
getting question 2 wrong produces a model that looks excellent and is useless.
The reasoning is written out in docs/02-leakage.md; the short version is below.

---

**The label.** `final_result` takes four values: Pass, Distinction, Fail,
Withdrawn. We collapse to binary:

    at_risk = 1  if final_result in {Fail, Withdrawn}
    at_risk = 0  if final_result in {Pass, Distinction}

Collapsing Fail and Withdrawn together is a choice, not a fact. It is the right
choice *for this use case*: the system exists to flag students a tutor should
reach out to, and both outcomes describe someone who did not get the
qualification they enrolled for. Splitting them apart is a natural later
extension, and they do behave differently in time.

---

**The population.** Every date in OULAD is a day offset from the start of the
course. If we make our prediction on day 28, then a student whose
`date_unregistration` is 10 has *already left* by the time we score them. Asking
a model to "predict" that is not prediction; the answer is already written down.

Worse, it is actively harmful to the model. A student who left on day 10 has no
clicks after day 10, so any model quickly discovers the rule "no recent activity
implies withdrawal". That rule scores wonderfully and tells a tutor nothing they
could not get from a database query -- and it crowds out the subtler signals
that would have flagged a student who is *still enrolled* and quietly drifting.

So the honest framing is:

    Among students still enrolled on day C, who will end up failing or
    withdrawing by the end of the course?

`eligible_at_cutoff` implements "still enrolled on day C". Everything downstream
filters through it.
"""

from __future__ import annotations

import pandas as pd

POSITIVE_OUTCOMES = ("Fail", "Withdrawn")
NEGATIVE_OUTCOMES = ("Pass", "Distinction")

# The columns that identify one student's enrolment on one course.
KEY = ["id_student", "code_module", "code_presentation"]


def build_labels(
    student_info: pd.DataFrame,
    student_registration: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per enrolment with the binary label and withdrawal timing.

    Output columns
    --------------
    id_student, code_module, code_presentation
        The enrolment key.
    final_result
        The original four-way outcome, kept for reference and for the
        later four-class extension.
    at_risk
        The binary label: 1 for Fail or Withdrawn, 0 for Pass or Distinction.
    withdrew
        1 if the outcome was specifically Withdrawn.
    date_unregistration
        Day the student left, or NaN if they never did. **This column must
        never be used as a model input** -- it is the answer in disguise. It is
        carried here so that `eligible_at_cutoff` can use it to define the
        population, which is a legitimate use.
    date_registration
        Day the student registered, relative to course start (usually negative).
        This one *is* safe as a feature: it is known before the course begins.
    """
    labels = student_info[KEY + ["final_result"]].copy()

    unknown = set(labels["final_result"].unique()) - set(
        POSITIVE_OUTCOMES + NEGATIVE_OUTCOMES
    )
    if unknown:
        raise ValueError(f"Unexpected final_result value(s): {sorted(unknown)}")

    labels["at_risk"] = labels["final_result"].isin(POSITIVE_OUTCOMES).astype("int8")
    labels["withdrew"] = (labels["final_result"] == "Withdrawn").astype("int8")

    reg = student_registration[KEY + ["date_registration", "date_unregistration"]]

    merged = labels.merge(reg, on=KEY, how="left", validate="one_to_one")

    # Consistency check: in OULAD, a non-null unregistration date should
    # correspond to a Withdrawn outcome. Where it does not, we want to know.
    inconsistent = int(
        (merged["date_unregistration"].notna() & (merged["withdrew"] == 0)).sum()
    )
    if inconsistent:
        # This is a warning, not an error: a handful of students unregister
        # after already having passed, which is odd but not corrupt.
        print(
            f"  note: {inconsistent:,} enrolments have an unregistration date "
            "but a non-Withdrawn outcome."
        )

    return merged


def eligible_at_cutoff(labels: pd.DataFrame, cutoff_day: int) -> pd.Series:
    """Boolean mask: was this student still enrolled on ``cutoff_day``?

    A student is *not* eligible if they had already unregistered on or before
    the cutoff. Their outcome is known by then, so scoring them is bookkeeping,
    not prediction, and including them inflates every metric.

    Note the ``<=``: a student who unregisters exactly on the cutoff day has
    already gone by the time we run the model that evening.
    """
    already_left = (
        labels["date_unregistration"].notna()
        & (labels["date_unregistration"] <= cutoff_day)
    )
    return ~already_left


def cutoff_population_summary(
    labels: pd.DataFrame, cutoffs: list[int]
) -> pd.DataFrame:
    """How the population and the base rate change as the cutoff moves later.

    This table is the single most useful thing to look at before modelling. It
    quantifies the central tension of an early-warning system: waiting longer
    gives you more behavioural data per student, but the students you most
    wanted to help have already gone, and the ones who remain are harder to
    separate.

    "Base rate" is the proportion of the population that is positive -- here,
    the share of still-enrolled students who go on to fail or withdraw. It is
    the score a model gets for free by guessing, and therefore the number every
    other result has to be compared against.
    """
    rows = []
    total = len(labels)
    for c in cutoffs:
        mask = eligible_at_cutoff(labels, c)
        pop = labels.loc[mask]
        rows.append(
            {
                "cutoff_day": c,
                "n_eligible": len(pop),
                "pct_of_all": 100 * len(pop) / total,
                "n_already_left": total - len(pop),
                "base_rate_at_risk": pop["at_risk"].mean(),
                "base_rate_withdraw": pop["withdrew"].mean(),
            }
        )
    return pd.DataFrame(rows)
