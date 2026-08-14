"""Tests for schema validation on load.

The point of these is that the loader should fail loudly and specifically. A
missing column noticed at load time names the file; the same problem noticed
three joins later surfaces as a KeyError somewhere unrelated.
"""

import pandas as pd
import pytest

from oulad import load
from oulad.schema import TABLES


def write_csv(directory, name, df):
    df.to_csv(directory / name, index=False)


def valid_student_info():
    return pd.DataFrame(
        {
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2013J"],
            "id_student": [1, 2],
            "gender": ["M", "F"],
            "region": ["Scotland", "Wales"],
            "highest_education": ["A Level or Equivalent", "HE Qualification"],
            "imd_band": ["20-30%", None],  # nullable by spec
            "age_band": ["0-35", "35-55"],
            "num_of_prev_attempts": [0, 1],
            "studied_credits": [60, 60],
            "disability": ["N", "N"],
            "final_result": ["Pass", "Fail"],
        }
    )


def test_valid_file_loads(tmp_path):
    write_csv(tmp_path, "studentInfo.csv", valid_student_info())
    df = load.load_table("studentInfo", data_dir=tmp_path)
    assert len(df) == 2


def test_missing_column_is_reported_by_name(tmp_path):
    df = valid_student_info().drop(columns=["final_result"])
    write_csv(tmp_path, "studentInfo.csv", df)

    with pytest.raises(load.SchemaError, match="final_result"):
        load.load_table("studentInfo", data_dir=tmp_path)


def test_unexpected_null_is_rejected(tmp_path):
    """imd_band may be null; final_result may not."""
    df = valid_student_info()
    df.loc[0, "final_result"] = None
    write_csv(tmp_path, "studentInfo.csv", df)

    with pytest.raises(load.SchemaError, match="unexpected nulls"):
        load.load_table("studentInfo", data_dir=tmp_path)


def test_duplicate_grain_is_rejected(tmp_path):
    """Two rows for the same student on the same course would multiply on join."""
    df = valid_student_info()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    write_csv(tmp_path, "studentInfo.csv", df)

    with pytest.raises(load.SchemaError, match="duplicate the declared grain"):
        load.load_table("studentInfo", data_dir=tmp_path)


def test_missing_file_points_at_the_readme(tmp_path):
    with pytest.raises(FileNotFoundError, match="Getting the data"):
        load.load_table("studentInfo", data_dir=tmp_path)


def test_unknown_table_name(tmp_path):
    with pytest.raises(KeyError):
        load.load_table("studentGrades", data_dir=tmp_path)


def test_data_available_requires_all_seven(tmp_path):
    assert not load.data_available(tmp_path)
    for spec in TABLES.values():
        (tmp_path / spec.filename).touch()
    assert load.data_available(tmp_path)
