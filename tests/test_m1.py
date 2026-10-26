"""Tests for the M1 loaders and helpers (src/data.py, src/m1_validate.py).

Each test checks one rule from the README that later milestones rely on. The
F17 and A2 tests read the real files in data/raw/ (read only). The two
tests for checks 11-12 (added in M5) use made-up F2 data and the real F1
file; the F1 test is skipped if that gitignored file is not on disk.
"""

import numpy as np
import pandas as pd
import pytest

import data
import m1_validate


@pytest.fixture(scope="module")
def f17():
    """Both F17 files, loaded once for all tests in this file."""
    return {"yields": data.load_f17("yields"), "forwards": data.load_f17("forwards")}


@pytest.fixture(scope="module")
def a2():
    """The A2 change list, loaded once."""
    return data.load_a2()


def test_f17_shape_grid_and_dates(f17):
    """2,448 rows, the 41-point 0-10 year grid, increasing dates, same dates in both files."""
    for frame in f17.values():
        assert frame.shape == (2448, 41)
        assert list(frame.columns) == [0.25 * k for k in range(41)]
        assert frame.index.is_monotonic_increasing and frame.index.is_unique
        assert frame.index[0] == pd.Timestamp("2017-01-03")
        assert frame.index[-1] == pd.Timestamp("2026-09-30")
    assert f17["yields"].index.equals(f17["forwards"].index)


def test_f17_has_no_blanks_and_is_numeric(f17):
    """Every cell is a float and none is missing (blanks would be NaN, never 0)."""
    for frame in f17.values():
        assert (frame.dtypes == "float64").all()
        assert int(frame.isna().sum().sum()) == 0


def test_a2_range_strings_stay_missing(a2):
    """Ranges such as '-0.50 to -1.00' do not crash the loader and do not become numbers."""
    # The helper on its own: text range -> (NaN, the text kept).
    value, text = data._a2_number("-0.50 to -1.00")
    assert np.isnan(value) and text == "-0.50 to -1.00"
    assert data._a2_number(-0.25) == (-0.25, None)

    # In the real file: exactly three rows hold text, all in early 1990, all ranges.
    text_rows = a2[a2["has_text"]]
    assert len(text_rows) == 3
    assert (text_rows["effective_date"].dt.year == 1990).all()
    first = a2.iloc[0]
    assert first["effective_date"] == pd.Timestamp("1990-01-23")
    assert np.isnan(first["change_pp"]) and np.isnan(first["new_target"])
    assert first["change_text"] == "-0.50 to -1.00"
    # Every row after the text rows has real numbers.
    later = a2[a2["effective_date"] > pd.Timestamp("1990-04-04")]
    assert later["change_pp"].notna().all() and later["new_target"].notna().all()


def test_26_changes_in_f17_sample(a2, f17):
    """README and the README: 26 changes inside the curve sample, 9 cuts and 17 hikes."""
    inside = m1_validate.changes_in_sample(a2, f17["yields"])
    assert len(inside) == 26
    assert int((inside["change_pp"] < 0).sum()) == 9
    assert int((inside["change_pp"] > 0).sum()) == 17


def test_target_known_at_close_is_one_day_ahead(a2):
    """A decision announced on d is in target_known_at_close(d) but not in target_in_effect(d)."""
    # Real data: hike to 4.60 announced 29 Sep 2026, effective 30 Sep 2026.
    days = pd.DatetimeIndex(["2026-09-28", "2026-09-29", "2026-09-30"])
    assert data.target_in_effect(days, a2).tolist() == [4.35, 4.35, 4.60]
    known = data.target_known_at_close(days, a2).tolist()
    assert known[:2] == [4.35, 4.60]
    # 30 Sep 2026 is the last day A2 covers (published 29 Sep), so the target
    # known at that close would need 1 Oct 2026: unknown, not "no change".
    assert np.isnan(known[2])

    # Made-up table: no target before the first change; boundaries are exact.
    toy = pd.DataFrame({"effective_date": pd.to_datetime(["2020-01-10", "2020-02-10"]),
                        "new_target": [1.0, 2.0]})
    days = pd.DatetimeIndex(["2020-01-08", "2020-01-09", "2020-01-10", "2020-02-09", "2020-02-10"])
    in_effect = data.target_in_effect(days, toy).tolist()
    known = data.target_known_at_close(days, toy).tolist()
    assert np.isnan(in_effect[0]) and in_effect[2:] == [1.0, 1.0, 2.0]
    assert np.isnan(known[0]) and known[1:] == [1.0, 1.0, 2.0, 2.0]


def test_2020_timing_cases(a2):
    """The off-cycle cut of Thu 19 Mar 2020 and the cut of Tue 3 Nov 2020, both effective the next day."""
    days = pd.DatetimeIndex(["2020-03-18", "2020-03-19", "2020-03-20", "2020-11-03", "2020-11-04"])
    assert data.target_in_effect(days, a2).tolist() == [0.50, 0.50, 0.25, 0.25, 0.10]
    assert data.target_known_at_close(days, a2).tolist() == [0.50, 0.25, 0.25, 0.10, 0.10]
    row = a2.loc[a2["effective_date"] == "2020-03-20"]
    assert row["announcement_date"].iloc[0] == pd.Timestamp("2020-03-19")


def test_target_is_missing_after_a2_coverage(a2):
    """After the A2 publication date + 1 day the target is unknown (NaN), never an assumed 'no change'."""
    assert (a2["covers_until"] == pd.Timestamp("2026-09-30")).all()
    values = data.target_in_effect(["2026-09-30", "2026-10-01", "2030-01-01"], a2).tolist()
    assert values[0] == 4.60 and np.isnan(values[1]) and np.isnan(values[2])


def test_target_known_at_close_refuses_pre_2008_dates(a2):
    """Before Feb 2008 announcement and effect were the same day, so d + 1 would look ahead."""
    with pytest.raises(ValueError):
        data.target_known_at_close(["2007-11-06"], a2)


def test_curve_maths_on_a_known_curve():
    """Forward helpers are exact where they should be.

    For a straight-line yield curve y(m) = a + b m, m y(m) is quadratic, so the
    central difference gives the instantaneous forward a + 2 b m exactly.
    For a straight-line forward curve the trapezoid rule is exact, so the
    average forward over [0, m] gives back y(m) = a + b m.
    """
    m = np.array([0.25 * k for k in range(41)])
    a, b = 1.5, 0.3
    yields = pd.DataFrame([a + b * m], columns=list(m))
    forwards = pd.DataFrame([a + 2 * b * m], columns=list(m))

    inst = m1_validate.instantaneous_forwards(yields).to_numpy()[0]
    assert np.allclose(inst[:-1], (a + 2 * b * m)[:-1])  # last point left as NaN by design
    assert np.isnan(inst[-1])

    rebuilt = m1_validate.average_forward(forwards, "trapezoid").to_numpy()[0]
    assert np.allclose(rebuilt[1:], (a + b * m)[1:])


def test_period_forward_helpers():
    """The competing readings that check 9 rejects are built correctly.

    On y(m) = a + b m, m y(m) = a m + b m^2, so the 3-month period forward
    F(m - h, m) = a + b (2m - h) and F(m, m + h) = a + b (2m + h).
    Round trip on any curve: a period forward is flat over its 3 months, so
    averaging 'starting' forwards with the left sum (or 'ending' forwards
    with the right sum) gives the yields back exactly.
    """
    m = np.array([0.25 * k for k in range(41)])
    h, a, b = 0.25, 1.5, 0.3
    line = pd.DataFrame([a + b * m], columns=list(m))
    ending = m1_validate.period_forwards(line, "ending").to_numpy()[0]
    starting = m1_validate.period_forwards(line, "starting").to_numpy()[0]
    assert np.isnan(ending[0]) and np.allclose(ending[1:], (a + b * (2 * m - h))[1:])
    assert np.isnan(starting[-1]) and np.allclose(starting[:-1], (a + b * (2 * m + h))[:-1])

    # A curved, made-up yield curve (per cent) for the round trip.
    curve = pd.DataFrame([1.0 + 2.0 * (1 - np.exp(-m / 3)) + 0.1 * np.sin(m)], columns=list(m))
    from_left = m1_validate.average_forward(m1_validate.period_forwards(curve, "starting"), "left")
    from_right = m1_validate.average_forward(m1_validate.period_forwards(curve, "ending"), "right")
    assert np.allclose(from_left.to_numpy()[0, 1:], curve.to_numpy()[0, 1:])
    assert np.allclose(from_right.to_numpy()[0, 1:], curve.to_numpy()[0, 1:])


def test_blank_cells_are_missing_and_text_is_an_error():
    """the README: blanks are missing data, never zero; text in F17 stops the loader."""
    assert np.isnan(data._to_float(" ", "x"))
    assert data._to_float("1.25", "x") == 1.25
    with pytest.raises(ValueError):
        data._to_float("abc", "x")
    # The report formatter shows every kind of missing value as an empty cell.
    assert [m1_validate.fmt_cell(v) for v in (None, pd.NA, pd.NaT, np.nan)] == ["", "", "", ""]


def test_processed_files_read_back(f17):
    """data/processed/ (written by m1_validate.py) reads back to exactly what the loaders give."""
    path = data.PROCESSED_DIR / "f17_yields.csv"
    if not path.exists():
        pytest.skip("run src/m1_validate.py first to write data/processed/")
    back = pd.read_csv(path, index_col="date", parse_dates=["date"])
    back.columns = [float(c) for c in back.columns]
    pd.testing.assert_frame_equal(back, f17["yields"], check_freq=False, check_names=False)


def test_sha256_ignores_crlf_only_when_asked(tmp_path):
    """Same text with LF and CRLF endings: different raw hashes, equal normalised hashes."""
    unix, windows = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    unix.write_bytes(b"a,b\n1,2\n")
    windows.write_bytes(b"a,b\r\n1,2\r\n")
    assert data.sha256_of_file(unix) != data.sha256_of_file(windows)
    assert data.sha256_of_file(unix, normalise_crlf=True) == data.sha256_of_file(windows, normalise_crlf=True)


# Checks 11 and 12 (added in M5): F2 and F1 cross-checks

def test_f2_comparison_uses_common_dates():
    """Check 11 lines F17 and F2 up on common dates before taking daily changes.

    Made-up data: F2 = F17 - 5 bp on every common date. F2 also has a row on
    Sat 4 Jan (standing in for a bank holiday F17 skips) holding a wild
    value, and is blank (NaN, as load_f2 gives) on 7 Jan, a date F17 has.
    If changes were taken on each file's own dates, the wild value would
    enter F2's changes. On common dates (2, 3, 6, 8 Jan) the two changes are
    identical and the gap is -5 bp everywhere.
    """
    f17_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
    yields = pd.DataFrame({2.0: 0.5, 10.0: [1.00, 1.10, 1.05, 1.20, 1.15]}, index=f17_dates)
    f2_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-06", "2020-01-07",
                               "2020-01-08"])
    f2 = pd.DataFrame({2.0: 0.5, 3.0: 0.6, 5.0: 0.8,
                       10.0: [0.95, 1.05, 9.99, 1.00, np.nan, 1.10]}, index=f2_dates)

    levels, changes = m1_validate.f2_10y_comparison(yields, f2)
    common = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08"])
    assert levels.index.equals(common)
    assert np.allclose(levels["gap_bp"], -5.0)
    # One change per common date after the first; the one dated 8 Jan runs from 6 Jan.
    assert changes.index.equals(common[1:])
    assert np.allclose(changes["f17"], [10.0, -5.0, 10.0])
    assert np.allclose(changes["f2"], changes["f17"])


def test_zero_year_point_is_f1_cash_rate(f17, a2):
    """Check 12 on the real files: the 0-year point equals F1's same-day actual cash rate on every F17 day.

    This confirms check 8's reading (the 0-year point is the interbank
    overnight cash rate, not the target). Skipped if the gitignored F1 file
    is not on disk.
    """
    if not (data.RAW_DIR / data.F1_FILE).exists():
        pytest.skip("F1 is gitignored and not on disk; re-download it from data/MANIFEST.csv")
    cash = data.load_f1()["cash_rate"].reindex(f17["yields"].index)
    assert cash.notna().all()  # F1 has a cash rate on every F17 date
    assert ((f17["yields"][0.0] - cash).abs() < m1_validate.TOL).all()

    outcome = m1_validate.check_f1_cash_rate(f17["yields"], a2)
    assert outcome["status"] == "PASS" and "100.0% of 2,448 F17 days" in outcome["line"]
