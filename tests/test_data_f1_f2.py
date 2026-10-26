"""Tests for the F1 and F2 loaders (data.load_f1, data.load_f2), added in M5.

Each test checks one fact the M5 analysis relies on: the right columns are found by their title
(not their position), blanks stay NaN and are never zero, F1's cash rate target follows the A2
effective-date convention (data.target_in_effect), OIS runs from 4 Jan 2011 to 1 Dec 2022, and F2
holds the four nominal maturities without the indexed bond.

Dates and counts are from the 20 Oct 2026 download (checksums in data/MANIFEST.csv). F1 and F2
are gitignored, so a fresh clone may not have them: the real-data tests are then skipped. Nothing
here writes F1 or F2 data anywhere (the last test writes a small made-up file to a temp folder).
"""

import numpy as np
import pandas as pd
import pytest

import data

F1_COLUMNS = ["cash_target", "cash_rate", "ois_1m", "ois_3m", "ois_6m"]
OIS = ["ois_1m", "ois_3m", "ois_6m"]


@pytest.fixture(scope="module")
def f1():
    """data.load_f1(), loaded once for the module; skips if the gitignored file is missing."""
    if not (data.RAW_DIR / data.F1_FILE).exists():
        pytest.skip("F1 is gitignored and not on disk")
    return data.load_f1()


@pytest.fixture(scope="module")
def f2():
    """data.load_f2(), loaded once for the module; skips if the gitignored file is missing."""
    if not (data.RAW_DIR / data.F2_FILE).exists():
        pytest.skip("F2 is gitignored and not on disk")
    return data.load_f2()


def test_f1_columns_and_first_row(f1):
    """F1 has exactly the five float columns, a clean date index, and the first raw row lands right.

    The raw row for 04-Jan-2011 is: target 4.75, change blank, cash rate 4.75, four blank cash
    market columns, index 100, bank bills 4.83 / 4.97 / 5.14, OIS 4.75 / 4.79 / 4.86, Treasury
    notes 4.81 / 4.84. Picking the OIS values out from between the bank bills and the Treasury
    notes shows the columns were found by title.
    """
    assert list(f1.columns) == F1_COLUMNS
    assert all(dtype == np.float64 for dtype in f1.dtypes)
    assert isinstance(f1.index, pd.DatetimeIndex) and f1.index.name == "date"
    assert f1.index.is_monotonic_increasing and not f1.index.has_duplicates
    first = f1.loc[pd.Timestamp("2011-01-04")]
    assert first.to_dict() == {"cash_target": 4.75, "cash_rate": 4.75,
                               "ois_1m": 4.75, "ois_3m": 4.79, "ois_6m": 4.86}


def test_f2_maturities(f2):
    """F2 has the 2, 3, 5 and 10-year nominal yields only; the indexed bond is left out.

    On 14-Oct-2026 the raw row is 4.919, 4.909, 4.976, 5.314 and then 2.983 for the indexed bond,
    so 2.983 must not appear. Start dates: the 10-year from 20 May 2013, the others from 2 Sep 2013.
    """
    assert list(f2.columns) == [2.0, 3.0, 5.0, 10.0]
    assert f2.columns.name == "maturity_years"
    assert all(dtype == np.float64 for dtype in f2.dtypes)
    assert f2.index.is_monotonic_increasing and not f2.index.has_duplicates
    assert f2.loc[pd.Timestamp("2026-10-14")].to_dict() == {2.0: 4.919, 3.0: 4.909, 5.0: 4.976, 10.0: 5.314}
    starts = {m: f2[m].first_valid_index() for m in f2.columns}
    assert starts == {2.0: pd.Timestamp("2013-09-02"), 3.0: pd.Timestamp("2013-09-02"),
                      5.0: pd.Timestamp("2013-09-02"), 10.0: pd.Timestamp("2013-05-20")}
    assert all(f2[m].last_valid_index() == pd.Timestamp("2026-10-14") for m in f2.columns)


def test_blanks_stay_nan_never_zero(f1, f2):
    """Blank cells become NaN, never 0; short rows are padded; all-blank dates are dropped.

    - No kept value is exactly 0 (the only 0.000 in either file is the F2 indexed bond on
      18 Feb 2022, which is not loaded), so no blank was filled with zero.
    - 1 Aug 2011 (NSW bank holiday): F1 has the target but OIS is blank.
    - 1 Mar 2023: the raw F1 row stops after the bank bills, so the OIS cells are missing; they
      must be NaN, and the target must still be read.
    - 3 Jan 2014: F2 has only the 10-year; 2, 3 and 5 years are NaN.
    - All-blank dates are not in the index: F1 20 Oct 2026 (only the total return index) and
      F2 4 Aug 2014 (NSW bank holiday, date only).
    """
    assert not (f1 == 0).any().any()
    assert not (f2 == 0).any().any()
    assert f1.isna().sum().to_dict() == {"cash_target": 0, "cash_rate": 0, "ois_1m": 985, "ois_3m": 985, "ois_6m": 985}
    assert f2.isna().sum().to_dict() == {2.0: 78, 3.0: 78, 5.0: 78, 10.0: 0}

    holiday = f1.loc[pd.Timestamp("2011-08-01")]
    assert holiday["cash_target"] == 4.75 and holiday[OIS].isna().all()
    short_row = f1.loc[pd.Timestamp("2023-03-01")]
    assert short_row[["cash_target", "cash_rate"]].notna().all() and short_row[OIS].isna().all()
    ten_only = f2.loc[pd.Timestamp("2014-01-03")]
    assert ten_only[[2.0, 3.0, 5.0]].isna().all() and ten_only[10.0] == 4.307

    assert pd.Timestamp("2026-10-20") not in f1.index and f1.index[-1] == pd.Timestamp("2026-10-19")
    assert pd.Timestamp("2014-08-04") not in f2.index


def test_f1_cash_target_is_target_in_effect(f1):
    """F1's target on day d equals data.target_in_effect(d), not target_known_at_close(d).

    Compared on every F1 date the A2 file covers (4 Jan 2011 to 30 Sep 2026, 3,985 dates): no
    difference at all. Against target_known_at_close the two differ on exactly the Board
    (announcement) days of the A2 changes in that span (38): on those days F1 still shows the
    old target, because the new one takes effect the next day.
    """
    changes = data.load_a2()
    covered = f1.index[f1.index <= changes["covers_until"].iloc[0]]
    assert len(covered) == 3985
    f1_target = f1.loc[covered, "cash_target"].to_numpy()

    in_effect = data.target_in_effect(covered, changes).to_numpy()
    np.testing.assert_allclose(f1_target, in_effect, rtol=0, atol=1e-9)

    known = data.target_known_at_close(covered, changes).to_numpy()
    differs = covered[np.isfinite(known) & (np.abs(f1_target - known) > 1e-9)]
    in_span = changes[(changes["effective_date"] > covered[0]) & (changes["effective_date"] <= covered[-1])]
    assert (in_span["change_pp"] != 0).all() and len(in_span) == 38
    assert set(differs) == set(in_span["announcement_date"])


def test_ois_start_and_end(f1):
    """OIS starts on the first F1 date (4 Jan 2011) and ends on 1 Dec 2022, for all three maturities.

    After 1 Dec 2022 every OIS cell is blank. Inside the span only four NSW bank holidays are
    blank, the same dates for 1, 3 and 6 months. This sets the sample of the M5 OIS benchmark.
    """
    for col in OIS:
        assert f1[col].first_valid_index() == pd.Timestamp("2011-01-04") == f1.index[0]
        assert f1[col].last_valid_index() == pd.Timestamp("2022-12-01")
    assert f1.loc["2022-12-02":, OIS].isna().all().all()
    span = f1.loc[:"2022-12-01", OIS]
    blank_dates = span.index[span.isna().any(axis=1)]
    assert list(blank_dates.strftime("%Y-%m-%d")) == ["2011-08-01", "2011-10-03", "2012-08-06", "2013-08-05"]
    assert span.loc[blank_dates].isna().all().all()  # all three maturities blank together


def test_f1_title_lookup_survives_reordered_columns(tmp_path, monkeypatch):
    """A made-up F1 with shuffled columns, look-alike titles and ragged rows still loads correctly.

    'Highest Interbank Overnight Cash Rate' and 'Change in the Cash Rate Target' sit before the
    columns we want, '3- month OIS' has the spaced hyphen F1 uses for Treasury notes, the second
    row omits its trailing cell and holds 'N/A' in a column we do not read, and the third row
    has no value in any kept column (so it is dropped).
    """
    lines = [
        "F1 TEST TABLE",
        "Title,Highest Interbank Overnight Cash Rate,6-month OIS,Change in the Cash Rate Target,"
        "Interbank Overnight Cash Rate,3- month OIS,1-month OIS,Cash Rate Target,Volume",
        "Units,Per cent,Per cent,Per cent,Per cent,Per cent,Per cent,Per cent,$m",
        "Series ID,FIRMMCRIH,FIRMMOIS6D,FIRMMCCRT,FIRMMCRID,FIRMMOIS3D,FIRMMOIS1D,FIRMMCRTD,FIRMMCRIV",
        "01-Feb-2022,9.99,0.36,,0.06,0.20,0.10,0.10,500",
        "02-Feb-2022,N/A,,,0.07,,,0.10",
        "03-Feb-2022,,,,,,,,123",
    ]
    (tmp_path / data.F1_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)

    got = data.load_f1()
    expected = pd.DataFrame(
        [[0.10, 0.06, 0.10, 0.20, 0.36], [0.10, 0.07, np.nan, np.nan, np.nan]],
        index=pd.DatetimeIndex(["2022-02-01", "2022-02-02"], name="date"),
        columns=F1_COLUMNS,
    )
    pd.testing.assert_frame_equal(got, expected)
