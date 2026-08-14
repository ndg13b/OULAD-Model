"""What each OULAD table is supposed to contain.

This module is documentation that the computer can check. Writing the expected
shape down means a corrupted or half-downloaded file fails loudly at load time
instead of silently producing a wrong answer twenty minutes later.

Two ideas that recur throughout this project:

**Grain** -- the grain of a table is what one row *means*. `studentInfo` has one
row per (student, module, presentation): if you see the same `id_student` twice,
that is not a duplicate, it is the same person taking a different course. Most
data bugs in this project come from joining two tables whose grain differs and
silently multiplying rows.

**Days relative to module start** -- every date column is an integer offset from
day 0 of the presentation, not a calendar date. Negative means before teaching
began (e.g. registering three weeks early is `date_registration = -21`).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    """The expected shape of one OULAD CSV."""

    filename: str
    grain: tuple[str, ...]  # columns that together uniquely identify a row
    columns: tuple[str, ...]  # every column that must be present
    description: str
    # Columns that are allowed to contain nulls. Anything else being null is a
    # signal that something is wrong with the file or our understanding of it.
    nullable: tuple[str, ...] = field(default_factory=tuple)


TABLES: dict[str, TableSpec] = {
    "studentInfo": TableSpec(
        filename="studentInfo.csv",
        grain=("id_student", "code_module", "code_presentation"),
        columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ),
        nullable=("imd_band",),
        description=(
            "One row per student enrolment. Holds demographics and the outcome "
            "column `final_result`, which is where our label comes from."
        ),
    ),
    "studentRegistration": TableSpec(
        filename="studentRegistration.csv",
        grain=("id_student", "code_module", "code_presentation"),
        columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "date_registration",
            "date_unregistration",
        ),
        nullable=("date_registration", "date_unregistration"),
        description=(
            "When each student registered and, if they left, when they "
            "unregistered. `date_unregistration` is non-null if and only if the "
            "student withdrew -- which makes it both the key to the timing of "
            "withdrawal and a serious leakage hazard. See docs/02-leakage.md."
        ),
    ),
    "studentVle": TableSpec(
        filename="studentVle.csv",
        grain=("id_student", "code_module", "code_presentation", "id_site", "date"),
        columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ),
        description=(
            "The clickstream, and the largest table by far (~10.7M rows). One "
            "row per student per VLE page per day, with the number of clicks. "
            "This is the main source of behavioural signal."
        ),
    ),
    "vle": TableSpec(
        filename="vle.csv",
        grain=("id_site",),
        columns=(
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        ),
        nullable=("week_from", "week_to"),
        description=(
            "Lookup table mapping a VLE page (`id_site`) to what kind of thing "
            "it is (`activity_type`: forum, quiz, content page, ...). Join this "
            "onto studentVle to turn raw clicks into interpretable categories."
        ),
    ),
    "assessments": TableSpec(
        filename="assessments.csv",
        grain=("id_assessment",),
        columns=(
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ),
        nullable=("date",),
        description=(
            "One row per assessment. `assessment_type` is TMA (tutor-marked), "
            "CMA (computer-marked) or Exam. `date` is the due date; `weight` is "
            "the percentage contribution to the final mark."
        ),
    ),
    "studentAssessment": TableSpec(
        filename="studentAssessment.csv",
        grain=("id_assessment", "id_student"),
        columns=(
            "id_assessment",
            "id_student",
            "date_submitted",
            "is_banked",
            "score",
        ),
        nullable=("score",),
        description=(
            "One row per submitted assessment. Note: only *submissions* appear "
            "here. A student who never submitted has no row at all, so a "
            "missing row is itself informative -- absence of evidence is, in "
            "this one case, evidence of absence."
        ),
    ),
    "courses": TableSpec(
        filename="courses.csv",
        grain=("code_module", "code_presentation"),
        columns=("code_module", "code_presentation", "module_presentation_length"),
        description=(
            "One row per module-presentation, giving its length in days. Useful "
            "for turning an absolute cutoff day into a fraction of the course."
        ),
    ),
}

# The four possible values of studentInfo.final_result.
FINAL_RESULTS = ("Pass", "Fail", "Withdrawn", "Distinction")

# `highest_education` and `imd_band` are *ordinal*: their categories have a
# meaningful order (more deprived -> less deprived; less education -> more).
# Encoding them as unordered categories throws that ordering away, so we record
# the intended order here and use it when we build features.
EDUCATION_ORDER = (
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
)

# IMD = Index of Multiple Deprivation, the UK government's official measure of
# relative deprivation for small areas. "0-10%" means the student's home area is
# among the 10% most deprived in the country; "90-100%" the least deprived.
IMD_ORDER = (
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
)

AGE_ORDER = ("0-35", "35-55", "55<=")
