"""Generate a small, fake OULAD-shaped dataset.

WHY THIS EXISTS
---------------
This is **not** a substitute for the real data and no result produced from it is
a finding about real students. It exists for two reasons, both of which are
ordinary practice:

1. **Develop the pipeline without waiting on data.** The code can be written and
   tested against something with the right shape.

2. **Know the ground truth.** In real data you never find out whether your model
   found genuine signal or an artefact of your own pipeline. Here we *plant* the
   signal ourselves, so we can ask a question that real data cannot answer: does
   the pipeline recover what we put in, and does it correctly find nothing when
   we put nothing in? That second test -- run the whole pipeline against pure
   noise and confirm it scores at chance -- is the cheapest and most effective
   leakage detector there is. If a model scores well on shuffled labels, the
   bug is in the code, not the data.

The generated files match the real schema exactly: same filenames, same columns,
same value vocabularies, same "days relative to course start" convention. They
are deliberately much smaller than the real thing (a few thousand students, not
32,593) so that iteration is fast.

Usage:
    python scripts/make_synthetic.py --out data/synthetic --students 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from oulad.schema import AGE_ORDER, EDUCATION_ORDER, IMD_ORDER

RNG_DEFAULT = 20140101

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]

REGIONS = [
    "East Anglian Region", "Scotland", "North Western Region", "South East Region",
    "West Midlands Region", "Wales", "North Region", "South Region",
    "Ireland", "South West Region", "East Midlands Region", "Yorkshire Region",
    "London Region",
]

ACTIVITY_TYPES = [
    "resource", "oucontent", "url", "homepage", "subpage", "glossary",
    "forumng", "oucollaborate", "dataplus", "quiz", "ouelluminate",
    "sharedsubpage", "questionnaire", "page", "externalquiz", "ouwiki",
    "dualpane", "repeatactivity", "folder", "htmlactivity",
]


def _make_course_calendar(rng: np.random.Generator) -> pd.DataFrame:
    """22 module-presentations, each with a length in days."""
    rows = []
    for m in MODULES:
        for p in PRESENTATIONS:
            # Not every module runs in every presentation; drop a few to land
            # near the real count of 22.
            if rng.random() < 0.22:
                continue
            rows.append(
                {
                    "code_module": m,
                    "code_presentation": p,
                    "module_presentation_length": int(rng.integers(234, 270)),
                }
            )
    return pd.DataFrame(rows)


def _make_vle(courses: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The catalogue of VLE pages, one row per site."""
    rows = []
    site_id = 500000
    for _, c in courses.iterrows():
        n_sites = int(rng.integers(60, 200))
        for _ in range(n_sites):
            site_id += 1
            has_week = rng.random() < 0.35
            wk = int(rng.integers(1, 30)) if has_week else None
            rows.append(
                {
                    "id_site": site_id,
                    "code_module": c["code_module"],
                    "code_presentation": c["code_presentation"],
                    "activity_type": rng.choice(ACTIVITY_TYPES),
                    "week_from": wk,
                    "week_to": (wk if has_week else None),
                }
            )
    return pd.DataFrame(rows)


def _make_students(
    courses: pd.DataFrame, n_students: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Demographics plus a hidden 'engagement' variable that drives everything.

    `z` is a latent trait we invent: higher z means a student who engages more
    and is more likely to pass. It is *not* written to the CSVs -- it is the
    ground truth the model is supposed to rediscover indirectly, through the
    behavioural traces it causes.
    """
    ids = rng.choice(np.arange(100000, 100000 + n_students * 3), size=n_students,
                     replace=False)
    rows = []
    for sid in ids:
        # Some students take more than one course, as in the real data.
        n_enrol = 1 + (rng.random() < 0.12) + (rng.random() < 0.03)
        picks = courses.sample(n=min(n_enrol, len(courses)), random_state=int(rng.integers(1e9)))

        # Student-level traits, stable across their enrolments.
        z_student = rng.normal(0, 1)
        edu_idx = int(np.clip(rng.normal(1.8, 1.0), 0, len(EDUCATION_ORDER) - 1))
        imd_idx = int(np.clip(rng.normal(4.7, 2.6), 0, len(IMD_ORDER) - 1))
        age_idx = int(rng.choice([0, 1, 2], p=[0.70, 0.28, 0.02]))
        gender = rng.choice(["M", "F"], p=[0.55, 0.45])
        disability = rng.choice(["N", "Y"], p=[0.905, 0.095])
        region = rng.choice(REGIONS)

        for _, c in picks.iterrows():
            prev = int(rng.choice([0, 1, 2, 3], p=[0.79, 0.14, 0.05, 0.02]))
            credits = int(rng.choice([30, 60, 90, 120, 150], p=[.28, .48, .12, .09, .03]))

            # Enrolment-level engagement: student trait, nudged by education,
            # deprivation, repeat attempts and course load. These coefficients
            # are the "true model" the pipeline will try to recover.
            z = (
                0.80 * z_student
                + 0.16 * (edu_idx - 1.8)
                + 0.10 * (imd_idx - 4.7) / 2.6
                - 0.34 * prev
                - 0.004 * (credits - 60)
                + rng.normal(0, 0.55)
            )
            rows.append(
                {
                    "code_module": c["code_module"],
                    "code_presentation": c["code_presentation"],
                    "id_student": int(sid),
                    "gender": gender,
                    "region": region,
                    "highest_education": EDUCATION_ORDER[edu_idx],
                    # imd_band is missing for a small share of students, as in
                    # the real file.
                    "imd_band": (None if rng.random() < 0.031 else IMD_ORDER[imd_idx]),
                    "age_band": AGE_ORDER[age_idx],
                    "num_of_prev_attempts": prev,
                    "studied_credits": credits,
                    "disability": disability,
                    "module_length": c["module_presentation_length"],
                    "z_latent": z,
                }
            )
    return pd.DataFrame(rows)


def _assign_outcomes(students: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Turn latent engagement into outcomes, withdrawal dates and registration.

    Target marginals, roughly matching the published OULAD distribution:
    Pass 38%, Withdrawn 31%, Fail 22%, Distinction 9%.
    """
    df = students.copy()
    z = df["z_latent"].to_numpy()
    n = len(df)

    # Withdrawal propensity falls as engagement rises. The intercepts here were
    # tuned so the marginal outcome mix lands near the published OULAD one.
    p_withdraw = 1 / (1 + np.exp(-(-1.25 - 1.05 * z)))
    withdrew = rng.random(n) < p_withdraw

    # Among those who stay, engagement decides fail / pass / distinction.
    u = rng.random(n)
    p_fail_given_stay = 1 / (1 + np.exp(-(-0.80 - 1.25 * z)))
    p_dist_given_stay = 1 / (1 + np.exp(-(-2.05 + 1.15 * z)))

    final = np.empty(n, dtype=object)
    final[withdrew] = "Withdrawn"
    stay = ~withdrew
    fail = stay & (u < p_fail_given_stay)
    final[fail] = "Fail"
    rest = stay & ~fail
    dist = rest & (rng.random(n) < p_dist_given_stay)
    final[dist] = "Distinction"
    final[rest & ~dist] = "Pass"
    df["final_result"] = final

    # Registration: mostly a few weeks before the course starts.
    df["date_registration"] = np.where(
        rng.random(n) < 0.006,
        np.nan,
        -np.abs(rng.normal(60, 45, n)).round(),
    )

    # Withdrawal timing. This is the part that matters most for the project:
    # withdrawals are front-loaded, so a large share happen in the first weeks.
    # Lower engagement means leaving sooner.
    length = df["module_length"].to_numpy()
    early_pull = np.clip(1.4 - 0.5 * z, 0.35, 3.0)
    raw = rng.gamma(shape=1.6, scale=38.0 / early_pull, size=n) - 12
    unreg = np.where(withdrew, np.clip(raw.round(), -25, length - 1), np.nan)
    df["date_unregistration"] = unreg

    return df


def _make_clicks(
    students: pd.DataFrame, vle: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """The clickstream: the observable trace of the latent engagement.

    Activity ramps up at course start, decays gently, spikes near assessment
    deadlines, and stops entirely once a student unregisters.
    """
    sites_by_course = {
        k: g["id_site"].to_numpy() for k, g in vle.groupby(["code_module", "code_presentation"])
    }

    frames = []
    for row in students.itertuples(index=False):
        key = (row.code_module, row.code_presentation)
        sites = sites_by_course.get(key)
        if sites is None or len(sites) == 0:
            continue

        length = int(row.module_length)
        # Last day the student is present.
        stop = length if np.isnan(row.date_unregistration) else int(row.date_unregistration)
        if stop <= 0:
            continue  # left before the course began: no clicks at all

        days = np.arange(0, stop)
        # Baseline daily intensity driven by engagement.
        intensity = np.exp(0.85 * row.z_latent - 0.9) * 2.4
        # Weekly rhythm and a slow decay over the presentation.
        seasonal = 1.0 + 0.30 * np.sin(2 * np.pi * days / 7.0)
        decay = np.exp(-days / (length * 0.9))
        # Disengagement ramp: students who are about to leave taper off first.
        if not np.isnan(row.date_unregistration):
            taper = np.clip((stop - days) / 14.0, 0.05, 1.0)
        else:
            taper = 1.0
        lam = np.clip(intensity * seasonal * decay * taper, 0, None)

        n_events = rng.poisson(lam)
        active = days[n_events > 0]
        if len(active) == 0:
            continue
        counts = n_events[n_events > 0]

        day_col = np.repeat(active, counts)
        total = len(day_col)
        frames.append(
            pd.DataFrame(
                {
                    "code_module": row.code_module,
                    "code_presentation": row.code_presentation,
                    "id_student": row.id_student,
                    "id_site": rng.choice(sites, size=total),
                    "date": day_col,
                    "sum_click": 1 + rng.poisson(2.2, size=total),
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["code_module", "code_presentation", "id_student",
                     "id_site", "date", "sum_click"]
        )
    clicks = pd.concat(frames, ignore_index=True)
    # Collapse to the real grain: one row per student/site/day.
    return (
        clicks.groupby(
            ["code_module", "code_presentation", "id_student", "id_site", "date"],
            as_index=False,
        )["sum_click"].sum()
    )


def _make_assessments(
    courses: pd.DataFrame, students: pd.DataFrame, rng: np.random.Generator
):
    """Assessment definitions and the submissions students made against them."""
    a_rows = []
    aid = 1000
    for _, c in courses.iterrows():
        length = c["module_presentation_length"]
        n_tma = int(rng.integers(3, 7))
        due = np.linspace(20, length - 40, n_tma).round().astype(int)
        weights = np.full(n_tma, 100.0 / n_tma)
        for d, w in zip(due, weights):
            aid += 1
            a_rows.append({
                "code_module": c["code_module"],
                "code_presentation": c["code_presentation"],
                "id_assessment": aid,
                "assessment_type": rng.choice(["TMA", "CMA"], p=[0.7, 0.3]),
                "date": int(d),
                "weight": round(float(w), 1),
            })
        aid += 1
        a_rows.append({
            "code_module": c["code_module"],
            "code_presentation": c["code_presentation"],
            "id_assessment": aid,
            "assessment_type": "Exam",
            "date": int(length),
            "weight": 100.0,
        })
    assessments = pd.DataFrame(a_rows)

    by_course = {
        k: g for k, g in assessments[assessments.assessment_type != "Exam"]
        .groupby(["code_module", "code_presentation"])
    }

    s_rows = []
    for row in students.itertuples(index=False):
        g = by_course.get((row.code_module, row.code_presentation))
        if g is None:
            continue
        stop = (row.module_length if np.isnan(row.date_unregistration)
                else row.date_unregistration)
        for a in g.itertuples(index=False):
            if a.date > stop:
                continue
            # Higher engagement -> more likely to submit at all.
            if rng.random() > 1 / (1 + np.exp(-(1.1 + 1.15 * row.z_latent))):
                continue
            offset = int(round(rng.normal(-2.5 - 1.2 * row.z_latent, 6)))
            score = float(np.clip(rng.normal(66 + 9.5 * row.z_latent, 15), 0, 100).round())
            s_rows.append({
                "id_assessment": a.id_assessment,
                "id_student": row.id_student,
                "date_submitted": int(a.date + offset),
                "is_banked": 0,
                "score": score,
            })
    student_assessment = pd.DataFrame(s_rows).drop_duplicates(
        subset=["id_assessment", "id_student"]
    )
    return assessments, student_assessment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic", type=Path)
    ap.add_argument("--students", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic OULAD-shaped data")
    print(f"  seed={args.seed}  target students={args.students:,}")

    courses = _make_course_calendar(rng)
    print(f"  courses            {len(courses):>10,} module-presentations")

    vle = _make_vle(courses, rng)
    print(f"  vle                {len(vle):>10,} sites")

    students = _make_students(courses, args.students, rng)
    students = _assign_outcomes(students, rng)
    print(f"  studentInfo        {len(students):>10,} enrolments")

    clicks = _make_clicks(students, vle, rng)
    print(f"  studentVle         {len(clicks):>10,} rows")

    assessments, student_assessment = _make_assessments(courses, students, rng)
    print(f"  assessments        {len(assessments):>10,} rows")
    print(f"  studentAssessment  {len(student_assessment):>10,} rows")

    # Split the working frame into the two real files it represents.
    student_info = students[[
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "final_result",
    ]]
    student_registration = students[[
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ]]

    courses.to_csv(out / "courses.csv", index=False)
    vle.to_csv(out / "vle.csv", index=False)
    student_info.to_csv(out / "studentInfo.csv", index=False)
    student_registration.to_csv(out / "studentRegistration.csv", index=False)
    clicks.to_csv(out / "studentVle.csv", index=False)
    assessments.to_csv(out / "assessments.csv", index=False)
    student_assessment.to_csv(out / "studentAssessment.csv", index=False)

    dist = student_info["final_result"].value_counts(normalize=True).mul(100).round(1)
    print("\n  final_result mix (%):")
    for k, v in dist.items():
        print(f"    {k:12s} {v:5.1f}")
    print(f"\nWrote 7 CSVs to {out}/")
    print("NOTE: synthetic data. No number derived from it is a real finding.")


if __name__ == "__main__":
    main()
