"""Tests for label construction and the cutoff population.

These matter more than they look. The population logic is the piece most likely
to be quietly wrong, and a quiet error there does not crash -- it just makes
every downstream score too high.
"""

import pandas as pd
import pytest

from oulad import labels as lab

KEY = ["id_student", "code_module", "code_presentation"]


def make_info(rows):
    return pd.DataFrame(
        [
            {
                "id_student": sid,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "final_result": result,
            }
            for sid, result in rows
        ]
    )


def make_reg(rows):
    return pd.DataFrame(
        [
            {
                "id_student": sid,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "date_registration": -30.0,
                "date_unregistration": unreg,
            }
            for sid, unreg in rows
        ]
    )


def test_binary_label_mapping():
    info = make_info([(1, "Pass"), (2, "Fail"), (3, "Withdrawn"), (4, "Distinction")])
    reg = make_reg([(1, None), (2, None), (3, 40.0), (4, None)])

    out = lab.build_labels(info, reg).set_index("id_student")

    assert out.loc[1, "at_risk"] == 0  # Pass
    assert out.loc[2, "at_risk"] == 1  # Fail
    assert out.loc[3, "at_risk"] == 1  # Withdrawn
    assert out.loc[4, "at_risk"] == 0  # Distinction

    # `withdrew` isolates only the Withdrawn outcome, not Fail.
    assert out.loc[2, "withdrew"] == 0
    assert out.loc[3, "withdrew"] == 1


def test_unexpected_outcome_value_raises():
    info = make_info([(1, "Deferred")])
    reg = make_reg([(1, None)])
    with pytest.raises(ValueError, match="Unexpected final_result"):
        lab.build_labels(info, reg)


def test_eligibility_excludes_students_already_gone():
    """A student who left before the cutoff is not a prediction."""
    info = make_info([(1, "Withdrawn"), (2, "Withdrawn"), (3, "Pass")])
    reg = make_reg([(1, 10.0), (2, 50.0), (3, None)])
    out = lab.build_labels(info, reg)

    elig = lab.eligible_at_cutoff(out, 28)
    kept = set(out.loc[elig, "id_student"])

    assert 1 not in kept, "student who left on day 10 must be excluded at day 28"
    assert 2 in kept, "student who leaves later is still a genuine prediction"
    assert 3 in kept, "student who never leaves is always eligible"


def test_eligibility_boundary_is_inclusive():
    """Leaving exactly on the cutoff day counts as already gone."""
    info = make_info([(1, "Withdrawn")])
    reg = make_reg([(1, 28.0)])
    out = lab.build_labels(info, reg)

    assert not lab.eligible_at_cutoff(out, 28).iloc[0]
    assert lab.eligible_at_cutoff(out, 27).iloc[0]


def test_base_rate_falls_as_cutoff_moves_later():
    """The characteristic pattern: excluding early leavers lowers the base rate.

    Early withdrawals are all positives, so removing them removes positives
    faster than negatives. A pipeline where this does *not* happen is a signal
    that the population filter is not being applied.
    """
    rows_info, rows_reg = [], []
    for i in range(50):  # withdraw early
        rows_info.append((i, "Withdrawn"))
        rows_reg.append((i, float(i % 20)))
    for i in range(50, 150):  # pass, never leave
        rows_info.append((i, "Pass"))
        rows_reg.append((i, None))

    out = lab.build_labels(make_info(rows_info), make_reg(rows_reg))
    summary = lab.cutoff_population_summary(out, [0, 7, 14, 28])

    rates = summary["base_rate_at_risk"].tolist()
    assert rates == sorted(rates, reverse=True), (
        f"base rate should fall monotonically as the cutoff moves later, got {rates}"
    )
    # By day 28 every early withdrawal is gone, leaving only the passes.
    assert summary.iloc[-1]["base_rate_at_risk"] == 0.0
    assert summary.iloc[-1]["n_eligible"] == 100


def test_population_summary_accounting_adds_up():
    info = make_info([(1, "Withdrawn"), (2, "Pass"), (3, "Fail")])
    reg = make_reg([(1, 5.0), (2, None), (3, None)])
    out = lab.build_labels(info, reg)

    summary = lab.cutoff_population_summary(out, [14])
    row = summary.iloc[0]
    assert row["n_eligible"] + row["n_already_left"] == len(out)
