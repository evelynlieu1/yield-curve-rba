"""Tests for M5 (src/m5_robustness.py).

Each test checks one rule the robustness report relies on: the OIS outcome averages the nights in
(t, t + h months] and no forecast looks past t; an OIS quote is put on the same basis as an average of
daily rates; the F17 zero yield used as a forecast is the average of F17 forwards; the M4 variant run
with default settings reproduces the M4 headline (so the new m4 arguments change nothing); the F2
duration hedge has the planned weights; and the M2 table's main rows are M2's own results.
F1 and F2 are gitignored, so tests that read them are skipped if the files are missing. Nothing here
writes F1 or F2 data anywhere.
"""

import numpy as np
import pandas as pd
import pytest

import data
import m2_expectations as m2
import m4_hedging as m4
import m5_robustness as m5


def synthetic_f1(seed=1):
    """A made-up F1 frame on 2020 business days: a random-walk cash rate and OIS columns near it."""
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-12-31"), name="date")
    rate = 0.25 + np.round(np.random.default_rng(seed).normal(0, 0.01, len(dates)).cumsum(), 4)
    return pd.DataFrame({"cash_target": 0.25, "cash_rate": rate, "ois_1m": rate + 0.01, "ois_3m": rate + 0.02,
                         "ois_6m": rate + 0.03}, index=dates)


def test_ois_outcome_window_has_no_look_ahead():
    """The outcome for t = Thu 30 Apr 2020, h = 1 is the mean over the 30 calendar days 1-30 May, each day
    carrying the rate of the latest business day on or before it (hand loop, not ffill). Changing the
    rate on t or after the window leaves it unchanged; changing 1 May (Fri, carried to Sat and Sun) moves
    it by 3/30 of the change. ois_panel's forecasts on t ignore every F1 and F17 row after t."""
    f1 = synthetic_f1()
    t = pd.Timestamp("2020-04-30")
    rate = f1["cash_rate"]
    days = pd.date_range("2020-05-01", "2020-05-30")
    expected = np.mean([rate[rate.index <= d].iloc[-1] for d in days])
    value, n_days = m5.realised_average(m5.calendar_cash_rate(rate), t, 1)
    assert n_days == 30 and value == pytest.approx(expected, abs=1e-12)

    def outcome_after(shocked_dates, size=1.0):
        r = rate.copy()
        r[r.index.isin(pd.DatetimeIndex(shocked_dates))] += size
        return m5.realised_average(m5.calendar_cash_rate(r), t, 1)[0]

    assert outcome_after([t]) == pytest.approx(value, abs=1e-12)                          # the night of t is out
    assert outcome_after(rate.index[rate.index >= "2020-06-01"]) == pytest.approx(value, abs=1e-12)
    assert outcome_after(["2020-05-01"]) == pytest.approx(value + 3 / 30, abs=1e-12)     # Fri, Sat, Sun

    grid = data.MATURITY_GRID
    yields = pd.DataFrame(0.3 + 0.1 * np.outer(np.ones(len(f1)), grid), index=f1.index, columns=grid)
    dates = m2.month_end_dates(f1.index)
    base = m5.ois_panel(f1, yields, dates, 3).set_index("t")
    f1_later, y_later = f1.copy(), yields.copy()
    f1_later.loc[f1_later.index > t] += 5.0
    y_later.loc[y_later.index > t] += 5.0
    moved = m5.ois_panel(f1_later, y_later, dates, 3).set_index("t")
    cols = ["ois_quote", "ois", "f17_zero", "no_change"]
    pd.testing.assert_frame_equal(moved.loc[:t, cols], base.loc[:t, cols])
    assert moved.loc[t, "outcome"] == pytest.approx(base.loc[t, "outcome"] + 5.0)       # only the outcome moves
    assert base.loc[t, "no_change"] == rate[t] and base.loc[t, "ois_quote"] == f1.loc[t, "ois_3m"]


def test_ois_average_basis_matches_daily_compounding():
    """An OIS quote q satisfies 1 + q n / 365 = product of (1 + r_d / 365). Built from known daily rates,
    the quote converted by ois_average_basis gives back their plain average (within 0.02 bp), while the
    raw quote sits above it (by about 2.2 bp for a flat 3% over 182 days)."""
    flat_quote = ((1 + 0.03 / 365) ** 182 - 1) * 365 / 182 * 100
    assert m5.ois_average_basis(flat_quote, 182) == pytest.approx(3.0, abs=2e-4)
    assert 2.0 < (flat_quote - 3.0) * 100 < 2.5
    r = np.random.default_rng(2).uniform(0.0, 4.0, 91)                    # 91 daily rates, per cent
    quote = (np.prod(1 + r / 100 / 365) - 1) * 365 / 91 * 100
    assert m5.ois_average_basis(quote, 91) == pytest.approx(r.mean(), abs=2e-4)


def test_f17_zero_is_forward_average():
    """Synthetic curve: forward f(s) = a + b s up to 0.25y, curved beyond. Then y(m) = average of f over
    [0, m], and f17_zero_at gives exactly a + b / 24 at 1 month (linear interpolation of y between 0 and
    0.25y is exact for a linear forward) and the grid yields at 3 and 6 months. Real F17: at the OIS
    month-ends the 0.25 and 0.5-year zero yields are within 2.5 bp (median) of the trapezoid average of
    F17's own forwards, M1's finding that y is the average of instantaneous forwards."""
    a, b, c = 1.0, 0.8, 3.0
    grid = np.array(data.MATURITY_GRID)
    tail = np.clip(grid - 0.25, 0, None)
    y = np.where(grid > 0, a + b * grid / 2 + c * tail ** 3 / (3 * np.where(grid > 0, grid, 1)), a)
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-28"])
    yields = pd.DataFrame([y, y], index=dates, columns=list(grid))
    assert m5.f17_zero_at(yields, dates, 1).to_numpy() == pytest.approx(a + b / 24, abs=1e-12)
    assert (m5.f17_zero_at(yields, dates, 3) == yields[0.25]).all()
    assert (m5.f17_zero_at(yields, dates, 6) == yields[0.5]).all()

    real_y, real_f = data.load_f17("yields"), data.load_f17("forwards")
    t = m2.month_end_dates(real_y.index)
    t = t[t <= "2022-11-30"]
    for m in (0.25, 0.5):
        pts = [g for g in data.MATURITY_GRID if g <= m]
        fwd = np.column_stack([real_y.loc[t, 0.0].to_numpy(), real_f.loc[t, pts[1:]].to_numpy()])  # f(0) = y(0)
        average = np.trapezoid(fwd, dx=0.25, axis=1) / m
        gap_bp = (real_y.loc[t, m].to_numpy() - average) * 100
        assert np.median(np.abs(gap_bp)) <= 2.5                  # M1: trapezoid MAD 2.0 bp at 0.25y
        assert (m5.f17_zero_at(real_y, t, int(m * 12)).to_numpy() == real_y.loc[t, m].to_numpy()).all()


def test_default_variant_reproduces_m4_headline():
    """run_hedges with no options is M4's main run: 33 blocks, 1,943 days, residual sd 5.46 / 0.93 / 0.74 /
    1.04 bp/day, t = +0.77 / +1.85 at Newey-West lag 7 and block-clustered p = 0.762 / 0.476 (M4 report).
    Passing target and factor_hedges explicitly (the new arguments) gives identical frames."""
    yields = data.load_f17("yields")
    clean = data.daily_changes_bp(yields, m5.MATURITIES, drop_suspect=True)
    every = data.daily_changes_bp(yields, m5.MATURITIES, drop_suspect=False)
    resid, blocks = m5.run_hedges(clean, every)
    row = m5.hedge_row("planned", resid, blocks)
    assert (row["n blocks"], row["n days"]) == (33, 1943)
    assert [round(row[f"sd {h}"], 2) for h in m4.HEDGES] == [5.46, 0.93, 0.74, 1.04]
    assert (round(row["t H3 vs H1"], 2), round(row["t H3 vs H2"], 2)) == (0.77, 1.85)
    assert (round(row["p cl H3 vs H1"], 3), round(row["p cl H3 vs H2"], 3)) == (0.762, 0.476)
    r_explicit, b_explicit = m4.rolling_hedges(clean, every, target=m4.TARGET, factor_hedges=m4.FACTOR_HEDGES)
    r_default, b_default = m4.rolling_hedges(clean, every)
    pd.testing.assert_frame_equal(r_explicit, r_default)
    pd.testing.assert_frame_equal(b_explicit, b_default)


def test_f2_h1_weights():
    """The 5-year duration hedge is 0.625 x 2y + 0.375 x 10y (DV01-neutral, maturity-weighted to 5y). On F2
    every block uses those weights, H1's residual equals dy5 - 0.625 dy2 - 0.375 dy10 recomputed from F2,
    and H3 (2y/3y/10y) has no in-sample PC1-3 exposure."""
    w = m4.h1_weights(m5.FIVE_YEAR)
    assert w.to_dict() == {2.0: 0.625, 10.0: 0.375}
    assert w.sum() == pytest.approx(1.0) and float(w.index.to_numpy() @ w.to_numpy()) == pytest.approx(5.0)
    if not (data.RAW_DIR / data.F2_FILE).exists():
        pytest.skip("F2 is gitignored and not on disk")
    f2, yields = data.load_f2(), data.load_f17("yields")
    _, runs = m5.f2_check(f2, yields)
    resid, blocks = runs["F2 bond yields"]
    assert (blocks["H1 w2"] == 0.625).all() and (blocks["H1 w10"] == 0.375).all()
    assert blocks["max |H3 exposure|"].max() < 1e-10
    levels = f2[data.F2_MATURITIES].dropna()
    levels = levels.loc[levels.index.intersection(yields[data.F2_MATURITIES].dropna().index)]
    dy = data.daily_changes_bp(levels, data.F2_MATURITIES).loc[resid.index]
    np.testing.assert_allclose(resid["H1"], dy[5.0] - 0.625 * dy[2.0] - 0.375 * dy[10.0], atol=1e-9)


def test_m2_table_matches_m2():
    """The M2 table's main rows are M2's own forward-vs-no-change DM results, and '2022 onwards' keeps
    exactly the forecasts made from Jan 2022 (54 at 3 months, outcome dates up to 30 Sep 2026)."""
    forwards, yields, changes = data.load_f17("forwards"), data.load_f17("yields"), data.load_a2()
    table = m5.m2_robustness(forwards, yields, changes)
    panels = {h: m2.build_panel(forwards, yields, changes, m2.month_end_dates(forwards.index), h)
              for h in m2.ALL_HORIZONS}
    dm, _ = m2.tests_tables(panels)
    own = dm[dm["model vs no change"] == "forward"].set_index("h")
    main = table[table["variant"].str.startswith("main")].set_index("h")
    np.testing.assert_allclose(main["DM t"], own.loc[main.index, "t"])
    np.testing.assert_allclose(main["DM p"], own.loc[main.index, "p"])
    recent = table[table["variant"].str.startswith("2022")].set_index("h")
    for h, p in panels.items():
        assert recent.loc[h, "n"] == int((p["t"] >= "2022-01-01").sum())
    assert recent.loc[3, "n"] == 54
