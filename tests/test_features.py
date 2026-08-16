"""Tests for feature building.

The most important test in this file is `test_events_after_cutoff_change_nothing`.
Everything else checks that a feature is computed correctly; that one checks the
property the whole project depends on, and it is the only one whose failure
would invalidate every result rather than just one column.
"""

import numpy as np
import pandas as pd
import pytest

from oulad import features
from oulad.features import KEY

MOD, PRES = "AAA", "2013J"


def _info(rows):
    """rows: (id_student, final_result, imd_band, education, age)"""
    return pd.DataFrame(
        [
            {
                "code_module": MOD, "code_presentation": PRES, "id_student": sid,
                "gender": "F", "region": "Scotland",
                "highest_education": edu, "imd_band": imd, "age_band": age,
                "num_of_prev_attempts": 0, "studied_credits": 60,
                "disability": "N", "final_result": res,
            }
            for sid, res, imd, edu, age in rows
        ]
    )


def _reg(rows):
    """rows: (id_student, date_unregistration)"""
    return pd.DataFrame(
        [
            {
                "code_module": MOD, "code_presentation": PRES, "id_student": sid,
                "date_registration": -30.0, "date_unregistration": unreg,
            }
            for sid, unreg in rows
        ]
    )


def _vle(rows):
    """rows: (id_student, id_site, date, sum_click)"""
    return pd.DataFrame(
        [
            {
                "code_module": MOD, "code_presentation": PRES, "id_student": sid,
                "id_site": site, "date": date, "sum_click": clicks,
            }
            for sid, site, date, clicks in rows
        ]
    )


def _tables(info, reg, vle_rows, assessments=None, submissions=None):
    sites = sorted({r["id_site"] for _, r in vle_rows.iterrows()}) or [1]
    return {
        "studentInfo": info,
        "studentRegistration": reg,
        "studentVle": vle_rows,
        "vle": pd.DataFrame(
            [
                {"id_site": s, "code_module": MOD, "code_presentation": PRES,
                 "activity_type": "oucontent", "week_from": None, "week_to": None}
                for s in sites
            ]
        ),
        "assessments": assessments if assessments is not None else pd.DataFrame(
            columns=["code_module", "code_presentation", "id_assessment",
                     "assessment_type", "date", "weight"]
        ),
        "studentAssessment": submissions if submissions is not None else pd.DataFrame(
            columns=["id_assessment", "id_student", "date_submitted",
                     "is_banked", "score"]
        ),
        "courses": pd.DataFrame(
            [{"code_module": MOD, "code_presentation": PRES,
              "module_presentation_length": 260}]
        ),
    }


@pytest.fixture
def simple():
    info = _info([
        (1, "Pass", "20-30%", "A Level or Equivalent", "0-35"),
        (2, "Fail", "0-10%", "No Formal quals", "35-55"),
    ])
    reg = _reg([(1, None), (2, None)])
    vle = _vle([
        # Student 1: active on days 0, 7, 14, 21 -- perfectly even spacing.
        (1, 100, 0, 5), (1, 100, 7, 5), (1, 100, 14, 5), (1, 100, 21, 5),
        # Student 2: same four visits, all bunched at the start.
        (2, 100, 0, 5), (2, 100, 1, 5), (2, 100, 2, 5), (2, 100, 21, 5),
    ])
    return _tables(info, reg, vle)


# --------------------------------------------------------------------------
# The one that matters
# --------------------------------------------------------------------------

def test_events_after_cutoff_change_nothing(simple):
    """Adding events after the cutoff must not move a single feature value.

    This is the property the entire project rests on. If it fails, every score
    reported anywhere is meaningless, because the model saw the future.
    """
    before = features.build_features(simple, 28)

    corrupted = {k: v.copy() for k, v in simple.items()}
    # Enormous, obviously-detectable activity, all strictly after the cutoff.
    future = _vle([(1, 100, d, 9999) for d in range(29, 200)]
                  + [(2, 100, d, 9999) for d in range(29, 200)])
    corrupted["studentVle"] = pd.concat([simple["studentVle"], future],
                                        ignore_index=True)

    after = features.build_features(corrupted, 28)

    cols = features.feature_columns(before)
    pd.testing.assert_frame_equal(
        before[KEY + cols].sort_values(KEY).reset_index(drop=True),
        after[KEY + cols].sort_values(KEY).reset_index(drop=True),
    )


def test_late_submissions_are_excluded(simple):
    """An assessment submitted after the cutoff has not happened yet."""
    assessments = pd.DataFrame([{
        "code_module": MOD, "code_presentation": PRES, "id_assessment": 500,
        "assessment_type": "TMA", "date": 20.0, "weight": 30.0,
    }])
    # Student 1 submits before the cutoff, student 2 after it.
    submissions = pd.DataFrame([
        {"id_assessment": 500, "id_student": 1, "date_submitted": 18,
         "is_banked": 0, "score": 70.0},
        {"id_assessment": 500, "id_student": 2, "date_submitted": 40,
         "is_banked": 0, "score": 65.0},
    ])
    tables = _tables(simple["studentInfo"], simple["studentRegistration"],
                     simple["studentVle"], assessments, submissions)

    out = features.build_features(tables, 28).set_index("id_student")

    assert out.loc[1, "submitted"] == 1.0
    assert out.loc[1, "first_score"] == 70.0
    assert out.loc[2, "submitted"] == 0.0, "day-40 submission must not count at day 28"
    assert pd.isna(out.loc[2, "first_score"])


def test_no_forbidden_columns_reach_the_feature_set(simple):
    out = features.build_features(simple, 28)
    cols = set(features.feature_columns(out))
    for bad in features.FORBIDDEN:
        assert bad not in cols, f"{bad} must never be a feature"


# --------------------------------------------------------------------------
# Consistency features
# --------------------------------------------------------------------------

def test_gap_cv_distinguishes_even_from_bursty(simple):
    """Same number of active days, different spacing -> different gap_cv."""
    out = features.build_features(simple, 28).set_index("id_student")

    assert out.loc[1, "active_days"] == out.loc[2, "active_days"] == 4
    assert out.loc[1, "total_clicks"] == out.loc[2, "total_clicks"]

    # Student 1's gaps are 7,7,7 -> no variation at all.
    assert out.loc[1, "gap_cv"] == pytest.approx(0.0)
    # Student 2's gaps are 1,1,19 -> highly variable.
    assert out.loc[2, "gap_cv"] > 1.0

    assert out.loc[1, "max_gap"] == 7
    assert out.loc[2, "max_gap"] == 19


def test_gap_cv_is_nan_with_too_little_history():
    """Below three active days there are not enough gaps to say anything."""
    info = _info([(1, "Pass", "20-30%", "HE Qualification", "0-35")])
    reg = _reg([(1, None)])
    vle = _vle([(1, 100, 3, 5), (1, 100, 9, 5)])  # only two active days

    out = features.build_features(_tables(info, reg, vle), 28).set_index("id_student")

    assert out.loc[1, "active_days"] == 2
    assert pd.isna(out.loc[1, "gap_cv"]), "must be NaN, not 0 -- unknown != regular"


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------

def test_click_slope_sign_follows_direction():
    info = _info([
        (1, "Pass", "20-30%", "HE Qualification", "0-35"),
        (2, "Fail", "20-30%", "HE Qualification", "0-35"),
    ])
    reg = _reg([(1, None), (2, None)])
    vle = _vle(
        # Student 1 ramps up week on week; student 2 fades away.
        [(1, 100, 1, 1), (1, 100, 8, 5), (1, 100, 15, 20), (1, 100, 22, 50)]
        + [(2, 100, 1, 50), (2, 100, 8, 20), (2, 100, 15, 5), (2, 100, 22, 1)]
    )

    out = features.build_features(_tables(info, reg, vle), 28).set_index("id_student")

    assert out.loc[1, "click_slope"] > 0
    assert out.loc[2, "click_slope"] < 0


def test_weekly_columns_track_the_cutoff():
    info = _info([(1, "Pass", "20-30%", "HE Qualification", "0-35")])
    reg = _reg([(1, None)])
    vle = _vle([(1, 100, 1, 5)])
    tables = _tables(info, reg, vle)

    at_14 = features.build_features(tables, 14)
    at_28 = features.build_features(tables, 28)

    assert "clicks_week_2" in at_14.columns
    assert "clicks_week_3" not in at_14.columns
    assert "clicks_week_4" in at_28.columns


# --------------------------------------------------------------------------
# Missingness semantics
# --------------------------------------------------------------------------

def test_nothing_due_is_distinct_from_due_and_not_submitted(simple):
    """The distinction that a naive encoding destroys."""
    # No assessments at all -> nothing was due.
    nothing_due = features.build_features(simple, 28).set_index("id_student")
    assert (nothing_due["assessment_due"] == 0).all()
    assert nothing_due["submitted"].isna().all(), (
        "with nothing due, 'submitted' is unknown, not False"
    )

    # Now something is due and neither student submits.
    assessments = pd.DataFrame([{
        "code_module": MOD, "code_presentation": PRES, "id_assessment": 500,
        "assessment_type": "TMA", "date": 20.0, "weight": 30.0,
    }])
    tables = _tables(simple["studentInfo"], simple["studentRegistration"],
                     simple["studentVle"], assessments)
    due_not_done = features.build_features(tables, 28).set_index("id_student")

    assert (due_not_done["assessment_due"] == 1).all()
    assert (due_not_done["submitted"] == 0.0).all(), (
        "with something due and nothing handed in, 'submitted' is False, not unknown"
    )


def test_zero_clicks_is_zero_not_missing():
    """A student who never clicked made zero clicks -- that is an observation."""
    info = _info([(1, "Fail", "20-30%", "HE Qualification", "0-35")])
    reg = _reg([(1, None)])
    # Another student generates the only clickstream rows.
    info = pd.concat([info, _info([(2, "Pass", "20-30%", "HE Qualification", "0-35")])])
    reg = pd.concat([reg, _reg([(2, None)])])
    vle = _vle([(2, 100, 5, 3)])

    out = features.build_features(_tables(info, reg, vle), 28).set_index("id_student")

    assert out.loc[1, "total_clicks"] == 0
    assert out.loc[1, "active_days"] == 0
    assert out.loc[1, "clicks_oucontent"] == 0
    # But "when were they last active" genuinely has no answer.
    assert pd.isna(out.loc[1, "days_since_last_activity"])


def test_ordinal_encoding_preserves_order_and_missingness():
    info = _info([
        (1, "Pass", "0-10%", "No Formal quals", "0-35"),
        (2, "Pass", "90-100%", "Post Graduate Qualification", "55<="),
        (3, "Pass", None, "HE Qualification", "35-55"),
    ])
    reg = _reg([(1, None), (2, None), (3, None)])
    vle = _vle([(1, 100, 1, 1), (2, 100, 1, 1), (3, 100, 1, 1)])

    out = features.build_features(_tables(info, reg, vle), 28).set_index("id_student")

    assert out.loc[1, "imd_band_ord"] < out.loc[2, "imd_band_ord"]
    assert out.loc[1, "education_ord"] < out.loc[2, "education_ord"]
    assert out.loc[1, "age_ord"] < out.loc[2, "age_ord"]
    assert pd.isna(out.loc[3, "imd_band_ord"]), "missing band stays missing"


# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------

def test_already_withdrawn_students_are_excluded_by_default():
    info = _info([
        (1, "Withdrawn", "20-30%", "HE Qualification", "0-35"),
        (2, "Pass", "20-30%", "HE Qualification", "0-35"),
    ])
    reg = _reg([(1, 10.0), (2, None)])
    vle = _vle([(1, 100, 1, 5), (2, 100, 1, 5)])
    tables = _tables(info, reg, vle)

    default = features.build_features(tables, 28)
    assert set(default["id_student"]) == {2}

    everyone = features.build_features(tables, 28, include_already_left=True)
    assert set(everyone["id_student"]) == {1, 2}
