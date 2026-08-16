"""Turning raw event tables into one row per enrolment.

Everything routes through :func:`build_features`. That is deliberate: the cutoff
rule -- *no information dated after the cutoff may be used* -- is easy to violate
by accident and impossible to detect afterwards from the numbers alone. One
function means one place where it can go wrong, and one place to test.

The rule, stated once:

    Every event table is filtered to ``date <= cutoff_day`` **before** any
    aggregation. Not after. Aggregating first and filtering second leaves the
    aggregate contaminated with the future, and nothing downstream will notice.

Feature families, and why each is here
--------------------------------------

**Volume** -- how much did they do? The obvious signal, and the weakest: notebook
04 shows the distributions overlap heavily.

**Consistency** -- how *evenly* did they do it? Someone with 200 clicks across 18
days is behaving differently from someone with 200 clicks in one burst before
going quiet. Notebook 04 found ``active_days`` separates about as well as total
clicks, which is what motivates measuring this properly rather than by proxy.

**Trajectory** -- which direction are they moving? A single total is
direction-blind. Weekly buckets and a slope are not.

**Diversity** -- what kinds of thing did they touch? Activity types differ in how
strongly they separate outcomes, so splitting clicks by type may carry more than
the sum does.

**Assessment** -- did they submit, how well, how promptly? The strongest single
signal in the EDA, and the one with the most awkward missingness. See
``_assessment_features`` for how "nothing was due yet" is kept distinct from
"something was due and they didn't do it".

**Demographics and context** -- who are they and what did they sign up for.
Modest effects, and deliberately not the centre of the model: flagging students
for who they are rather than what they do is both weaker and harder to defend.

Missing values are left as ``NaN`` rather than filled here. Imputation is a
modelling decision and belongs inside the cross-validation pipeline, where it
gets fitted on training folds only. Filling here would leak test-fold
information into training through the fill value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import labels as lab
from .schema import AGE_ORDER, EDUCATION_ORDER, IMD_ORDER

KEY = lab.KEY

# Activity types kept as their own columns. The rest are summed into "other".
# Chosen as the types carrying meaningful volume; a long tail of rare types
# would add columns that are almost always zero.
TRACKED_ACTIVITY_TYPES = (
    "oucontent",
    "homepage",
    "subpage",
    "resource",
    "url",
    "quiz",
    "forumng",
    "oucollaborate",
    "externalquiz",
    "ouwiki",
)

# Columns that must never reach a model. `date_unregistration` is non-null
# exactly when a student withdrew, so it is half the label written down.
FORBIDDEN = ("date_unregistration", "final_result", "at_risk", "withdrew")


def _ordinal(series: pd.Series, order: tuple[str, ...]) -> pd.Series:
    """Map an ordered category to integers, preserving NaN.

    `imd_band` and `highest_education` have a real order -- more deprived to
    less, less educated to more. Treating them as unordered categories throws
    that away and forces a model to rediscover it from data. Integers keep it.

    Unknown values become NaN rather than silently mapping to a number.
    """
    lookup = {v: i for i, v in enumerate(order)}
    return series.map(lookup).astype("float64")


def _clickstream_features(
    student_vle: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """Volume, consistency, trajectory and diversity from the clickstream."""
    early = student_vle[student_vle["date"] <= cutoff_day]

    # --- volume -----------------------------------------------------------
    base = (
        early.groupby(KEY)
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            distinct_sites=("id_site", "nunique"),
            first_active_day=("date", "min"),
            last_active_day=("date", "max"),
        )
        .reset_index()
    )
    base["clicks_per_active_day"] = base["total_clicks"] / base["active_days"]
    base["days_since_last_activity"] = cutoff_day - base["last_active_day"]

    # --- consistency ------------------------------------------------------
    base = base.merge(_gap_features(early), on=KEY, how="left")

    # --- trajectory -------------------------------------------------------
    base = base.merge(_weekly_features(early, cutoff_day), on=KEY, how="left")

    # --- diversity --------------------------------------------------------
    return base


def _gap_features(early: pd.DataFrame) -> pd.DataFrame:
    """Regularity of engagement, measured from the spacing of active days.

    The idea is borrowed from work on schedule regularity in habit formation:
    *how evenly* someone engages may matter separately from *how much*. Two
    students with twelve active days behave differently if one is spread evenly
    and the other is two clusters with a fortnight of silence between.

    ``gap_cv`` is the coefficient of variation of the gaps between consecutive
    active days -- the standard deviation divided by the mean. It is scale-free
    by construction, which is the point: **low means regular, high means
    bursty**, and it does not simply restate how often they showed up.

    That independence is the thing to check rather than assume. A well-known
    regularity index (the Sleep Regularity Index) turns out to correlate about
    -0.94 with log session count on real data -- it is largely a session counter
    with a regularity label. On OULAD at day 28, ``gap_cv`` correlates -0.24
    with log active days, so it is carrying mostly different information. A
    weekday-concentration measure was tested alongside and *did* show the
    confound (+0.86), so it is deliberately not included here.

    Undefined below three active days: with fewer than two gaps a standard
    deviation is not meaningful. Those enrolments get NaN, not zero -- "too
    little history to say" is different from "perfectly regular".

    **Measured result: these features add nothing.** Gradient boosting with and
    without ``gap_cv``, ``max_gap`` and ``mean_gap``, five-fold grouped CV:

    ========  =============  ==========  ========
    cutoff    without        with        delta
    ========  =============  ==========  ========
    day 28    0.7653         0.7657      +0.0004
    day 84    0.8264         0.8261      -0.0002
    day 168   0.9038         0.9040      +0.0003
    ========  =============  ==========  ========

    That is noise at every cutoff. It is not a shortage of history either: by
    day 168 the median student has 47 active days, well past the ~30 that a
    regularity score is usually said to need.

    The honest reading is that at *daily* resolution, once you already know how
    much and how recently someone engaged, how evenly they spaced it carries no
    additional information about this outcome. Whether that also holds at the
    sub-hour resolution the construct was designed for is untestable here --
    OULAD records no time of day at all. See docs/05-proposal-regularity.md.

    Kept in the feature set because a measured null is worth more than a
    silently deleted branch, and because it costs nothing. Drop it if the
    column count starts to matter.
    """
    day_lists = (
        early.groupby(KEY)["date"]
        .apply(lambda s: np.sort(s.unique()))
        .reset_index(name="days")
    )

    rows = []
    for days in day_lists["days"]:
        n = len(days)
        if n < 3:
            rows.append((np.nan, np.nan, np.nan))
            continue
        gaps = np.diff(days)
        mean_gap = gaps.mean()
        rows.append(
            (
                gaps.std() / mean_gap if mean_gap > 0 else np.nan,  # gap_cv
                float(gaps.max()),                                  # longest silence
                float(mean_gap),
            )
        )

    out = pd.DataFrame(rows, columns=["gap_cv", "max_gap", "mean_gap"])
    return pd.concat([day_lists[KEY], out], axis=1)


def _weekly_features(early: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Clicks per week, plus a trend across those weeks.

    A single total cannot tell rising from falling. `clicks_week_0`,
    `clicks_week_1`, ... let a model see the shape, and `click_slope` summarises
    it as one number: the least-squares gradient across the weekly counts.
    Negative means fading.
    """
    n_weeks = max(1, (cutoff_day // 7) + 1)

    w = early.copy()
    w["week"] = (w["date"].clip(lower=0) // 7).clip(upper=n_weeks - 1)

    weekly = (
        w.groupby(KEY + ["week"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(n_weeks), fill_value=0)
    )
    weekly.columns = [f"clicks_week_{i}" for i in range(n_weeks)]

    counts = weekly.to_numpy(dtype="float64")
    if n_weeks >= 2:
        x = np.arange(n_weeks, dtype="float64")
        x_centred = x - x.mean()
        denom = (x_centred**2).sum()
        slope = (counts * x_centred).sum(axis=1) / denom
    else:
        slope = np.zeros(len(counts))
    weekly["click_slope"] = slope

    # Share of all clicks that fall in the final week -- a scale-free way of
    # asking "are they still here?" that does not depend on how much they click.
    last = counts[:, -1]
    total = counts.sum(axis=1)
    weekly["last_week_share"] = np.where(total > 0, last / total, np.nan)

    return weekly.reset_index()


def _activity_type_features(
    student_vle: pd.DataFrame, vle: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    """Clicks split by what kind of page was being clicked."""
    early = student_vle[student_vle["date"] <= cutoff_day]
    typed = early.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")

    typed["activity_type"] = typed["activity_type"].where(
        typed["activity_type"].isin(TRACKED_ACTIVITY_TYPES), "other"
    )

    wide = (
        typed.groupby(KEY + ["activity_type"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
    )
    keep = list(TRACKED_ACTIVITY_TYPES) + ["other"]
    wide = wide.reindex(columns=keep, fill_value=0)
    wide.columns = [f"clicks_{c}" for c in wide.columns]

    wide["n_activity_types"] = (wide > 0).sum(axis=1)
    return wide.reset_index()


def _assessment_features(
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    cutoff_day: int,
) -> pd.DataFrame:
    """Submission behaviour up to the cutoff.

    The missingness here needs care, because three situations look similar in a
    naive encoding and mean completely different things:

    1. The student's course had **nothing due yet** at the cutoff.
    2. Something was due and they **submitted** it.
    3. Something was due and they **did not**.

    Collapsing 1 and 3 into "no submission" would tell a model that students on
    slow-starting modules resemble students who missed a deadline. They do not.

    So ``assessment_due`` is a separate flag, and ``submitted`` is NaN when
    nothing was due -- genuinely unknown rather than false. Downstream, tree
    models handle the NaN directly and linear models get an imputer plus a
    missingness indicator inside the pipeline.
    """
    due = assessments[
        (assessments["assessment_type"] != "Exam")
        & assessments["date"].notna()
        & (assessments["date"] <= cutoff_day)
    ]

    # Which module-presentations had anything due by now.
    courses_with_due = due[["code_module", "code_presentation"]].drop_duplicates()
    courses_with_due = courses_with_due.assign(assessment_due=1)

    subs = student_assessment.merge(
        due[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="inner",
    )
    # Only submissions that had actually happened by the cutoff.
    subs = subs[subs["date_submitted"] <= cutoff_day]

    per_student = (
        subs.sort_values("date_submitted")
        .groupby(KEY, as_index=False)
        .agg(
            n_submitted=("id_assessment", "nunique"),
            first_score=("score", "first"),
            mean_score=("score", "mean"),
            first_due=("date", "first"),
            first_submitted_on=("date_submitted", "first"),
        )
    )
    per_student["days_before_deadline"] = (
        per_student["first_due"] - per_student["first_submitted_on"]
    )
    per_student = per_student.drop(columns=["first_due", "first_submitted_on"])

    return courses_with_due, per_student


def _demographic_features(student_info: pd.DataFrame) -> pd.DataFrame:
    """Demographics, with ordered categories encoded as ordered integers."""
    out = student_info[KEY].copy()

    out["imd_band_ord"] = _ordinal(student_info["imd_band"], IMD_ORDER)
    out["education_ord"] = _ordinal(student_info["highest_education"], EDUCATION_ORDER)
    out["age_ord"] = _ordinal(student_info["age_band"], AGE_ORDER)

    out["is_female"] = (student_info["gender"] == "F").astype("int8")
    out["has_disability"] = (student_info["disability"] == "Y").astype("int8")
    out["num_of_prev_attempts"] = student_info["num_of_prev_attempts"]
    out["studied_credits"] = student_info["studied_credits"]
    out["region"] = student_info["region"]
    return out


def build_features(
    tables: dict[str, pd.DataFrame],
    cutoff_day: int,
    *,
    include_already_left: bool = False,
) -> pd.DataFrame:
    """One row per enrolment, using only information available at ``cutoff_day``.

    Parameters
    ----------
    tables
        The seven OULAD tables, as returned by :func:`oulad.load.load_all`.
    cutoff_day
        The day the prediction is made. No event dated later may influence any
        returned value.
    include_already_left
        Passed through to :func:`labels.modelling_population`. Off by default,
        so the population is students still enrolled at the cutoff. See
        docs/02-leakage.md for why.

    Returns
    -------
    DataFrame with the enrolment key, the label columns (``at_risk``,
    ``withdrew``, ``final_result``), and the feature columns. Use
    :func:`feature_columns` to get the model-input subset -- the label columns
    are carried for convenience and must not be fed to a model.
    """
    labels = lab.build_labels(tables["studentInfo"], tables["studentRegistration"])
    population = lab.modelling_population(
        labels, cutoff_day, include_already_left=include_already_left
    )

    df = population[KEY + ["final_result", "at_risk", "withdrew", "date_registration"]].copy()

    df = df.merge(_clickstream_features(tables["studentVle"], cutoff_day),
                  on=KEY, how="left")
    df = df.merge(
        _activity_type_features(tables["studentVle"], tables["vle"], cutoff_day),
        on=KEY, how="left",
    )

    courses_with_due, per_student = _assessment_features(
        tables["assessments"], tables["studentAssessment"], cutoff_day
    )
    df = df.merge(courses_with_due, on=["code_module", "code_presentation"], how="left")
    df["assessment_due"] = df["assessment_due"].fillna(0).astype("int8")
    df = df.merge(per_student, on=KEY, how="left")

    # Submitted is only a meaningful question where something was due.
    submitted = df["n_submitted"].notna() & (df["n_submitted"] > 0)
    df["submitted"] = np.where(df["assessment_due"] == 1, submitted.astype(float), np.nan)
    df["n_submitted"] = np.where(df["assessment_due"] == 1,
                                 df["n_submitted"].fillna(0), np.nan)

    df = df.merge(_demographic_features(tables["studentInfo"]), on=KEY, how="left")

    df = df.merge(
        tables["courses"], on=["code_module", "code_presentation"], how="left"
    )
    df["cutoff_fraction"] = cutoff_day / df["module_presentation_length"]

    # A student with no clickstream rows genuinely made zero clicks. That is an
    # observation, not a missing value, so these specific columns are filled.
    zero_means_zero = (
        ["total_clicks", "active_days", "distinct_sites", "n_activity_types"]
        + [c for c in df.columns if c.startswith("clicks_")]
    )
    for col in zero_means_zero:
        df[col] = df[col].fillna(0)

    _assert_no_leakage(df)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns that may be fed to a model.

    Excludes the enrolment key, the label columns, and anything on the
    forbidden list. Keeping this as a function rather than a hardcoded list
    means adding a feature does not require remembering to update two places.
    """
    exclude = set(KEY) | set(FORBIDDEN) | {"module_presentation_length"}
    return [c for c in df.columns if c not in exclude]


def _assert_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if a forbidden column survived into the feature frame."""
    present = [c for c in FORBIDDEN if c in feature_columns(df)]
    if present:
        raise AssertionError(
            f"Forbidden column(s) reached the feature set: {present}. "
            "See docs/02-leakage.md."
        )
