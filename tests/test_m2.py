"""Tests for the M2 expectations test (src/m2_expectations.py).

Each test pins one M2 rule that the results depend on: no
look-ahead, announcement-day timing, the forecast and outcome dates and the
sample rule, the gap adjustment and error sign, the Newey-West lag, the
Harvey-Leybourne-Newbold DM, the Diebold-Mariano sign, the Fama regression
and the episodes. The data
tests read the real files in data/raw/ (read only); the statistics tests use
small made-up series so the right answer is known.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import data
import m2_expectations as m2


@pytest.fixture(scope="module")
def inputs():
    """F17 forwards, F17 yields and the A2 change list, loaded once."""
    return data.load_f17("forwards"), data.load_f17("yields"), data.load_a2()


def test_no_look_ahead(inputs):
    """Damaging every F17 row after t0 and dropping every A2 decision announced after t0 leaves forecasts made on or before t0 unchanged.

    t0 = Mon 31 Oct 2022, and the next decision was announced the next day
    (Tue 1 Nov 2022, effective Wed 2 Nov). So if build_panel read the curve
    after t, or the target even 2 days after t, the clean and the damaged
    inputs would give different forecasts for t <= t0.
    """
    forwards, yields, changes = inputs
    t0 = pd.Timestamp("2022-10-31")
    bad_forwards, bad_yields = forwards.copy(), yields.copy()
    bad_forwards.loc[bad_forwards.index > t0] = 99.0
    bad_yields.loc[bad_yields.index > t0] = 99.0
    bad_changes = changes[changes["announcement_date"] <= t0]
    assert changes["announcement_date"].gt(t0).any() and not bad_changes["announcement_date"].gt(t0).any()
    dates = m2.month_end_dates(forwards.index)
    forecast_cols = ["t", "forward", "y0", "known", "gap_adj", "no_change"]

    for h in m2.ALL_HORIZONS:
        clean = m2.build_panel(forwards, yields, changes, dates, h)
        damaged = m2.build_panel(bad_forwards, bad_yields, bad_changes, dates, h)
        early_clean = clean.loc[clean["t"] <= t0, forecast_cols].reset_index(drop=True)
        early_damaged = damaged.loc[damaged["t"] <= t0, forecast_cols].reset_index(drop=True)
        assert len(early_clean) == 70
        pd.testing.assert_frame_equal(early_clean, early_damaged)
        # The damage does reach later forecasts, in the curve and in 'known', so the comparison has teeth.
        late_clean, late_damaged = clean[clean["t"] > t0], damaged[damaged["t"] > t0]
        assert (late_damaged["forward"] == 99.0).all()
        assert (late_damaged["known"].to_numpy() != late_clean["known"].to_numpy()).any()


def test_announcement_day_timing(inputs):
    """'known' switches to a new target on the announcement day, never before: all 26 decisions in the sample.

    For each decision announced on day d (A2 effective date minus 1 day):
      - build_panel refuses t = d (known - y0 would add the decision to a
        forward that already prices it);
      - on the F17 day before d, known is the old target (nothing leaks from d);
      - on the F17 day after d, known is the new target.
    Expected targets come straight from the A2 rows, not from target_in_effect.
    On every day other than d, target_in_effect(t) and target_in_effect(t + 1)
    agree, so refusing d leaves no room for an off-by-one the other way.
    """
    forwards, yields, changes = inputs
    moves = changes[(changes["announcement_date"] >= forwards.index[0]) & (changes["change_pp"] != 0)]
    assert len(moves) == 26
    checked = 0
    for i, row in moves.iterrows():
        d = row["announcement_date"]
        with pytest.raises(ValueError, match="announcement day"):
            m2.build_panel(forwards, yields, changes, [d], 3)
        before = forwards.index[forwards.index < d][-1]
        after = forwards.index[forwards.index > d][0]
        known = m2.build_panel(forwards, yields, changes, [before, after], 3).set_index("t")["known"]
        # The last decision (29 Sep 2026) has no 3-month outcome, so its rows drop out of the sample.
        if before in known.index:
            assert known[before] == pytest.approx(changes.loc[i - 1, "new_target"])
            checked += 1
        if after in known.index:
            assert known[after] == pytest.approx(row["new_target"])
            checked += 1
    assert checked == 50


def test_gap_adjustment_and_error_sign(inputs):
    """28 Feb 2022, h = 6: gap = known - y0 = 5 bp; gap_adj = forward + gap; error = outcome - forecast (bp)."""
    forwards, yields, changes = inputs
    p = m2.build_panel(forwards, yields, changes, m2.month_end_dates(forwards.index), 6)
    r = p[p["t"] == "2022-02-28"].iloc[0]
    assert (r["forward"], r["y0"], r["known"], r["outcome"]) == pytest.approx((0.54, 0.05, 0.10, 1.85))
    assert r["gap_bp"] == pytest.approx(5) and r["gap_adj"] == pytest.approx(0.59)
    assert (r["err_forward"], r["err_gap_adj"], r["err_no_change"]) == pytest.approx((131, 126, 175))


def test_forecast_dates_and_outcome_dates(inputs):
    """Month-end dates, forward read at h/12, outcome = t + h calendar months, sample ends 30 Sep 2026."""
    forwards, yields, changes = inputs
    dates = m2.month_end_dates(forwards.index)
    assert len(dates) == 117
    assert dates[0] == pd.Timestamp("2017-01-31") and dates[-1] == pd.Timestamp("2026-09-30")

    sizes = {}
    for h in m2.ALL_HORIZONS:
        p = m2.build_panel(forwards, yields, changes, dates, h)
        sizes[h] = len(p)
        np.testing.assert_allclose(p["forward"], forwards.loc[p["t"], h / 12].to_numpy())
        assert (p["outcome_date"] == p["t"] + pd.DateOffset(months=h)).all()
        assert p["outcome_date"].max() == pd.Timestamp("2026-09-30")
    assert sizes == {3: 114, 6: 111, 12: 105}

    # The sample rule is "outcome date on or before 30 Sep 2026". If A2 stopped earlier, an outcome inside
    # the sample would be missing: build_panel must stop, not silently drop the row.
    short_a2 = changes.assign(covers_until=pd.Timestamp("2026-06-30"))
    with pytest.raises(ValueError, match="missing outcome"):
        m2.build_panel(forwards, yields, short_a2, dates, 3)

    # Month-end clipping: 30 Nov 2017 + 3 months is 28 Feb 2018.
    p3 = m2.build_panel(forwards, yields, changes, dates, 3)
    assert p3.loc[p3["t"] == "2017-11-30", "outcome_date"].iloc[0] == pd.Timestamp("2018-02-28")
    # The case M1 flagged: t = 30 Jun 2026, h = 3. The 29 Sep 2026 hike takes effect on 30 Sep.
    row = p3[p3["t"] == "2026-06-30"].iloc[0]
    assert row["outcome_date"] == pd.Timestamp("2026-09-30") and row["outcome"] == pytest.approx(4.60)
    by_days = m2.build_panel(forwards, yields, changes, dates, 3, rule="days")
    row = by_days[by_days["t"] == "2026-06-30"].iloc[0]
    assert row["outcome_date"] == pd.Timestamp("2026-09-29") and row["outcome"] == pytest.approx(4.35)

    # Mid-month dates skip the M1 suspect day 13 Dec 2019 (falls back to 12 Dec).
    mid = m2.mid_month_dates(forwards.index, skip=pd.DatetimeIndex(sorted(data.SUSPECT_DAYS)))
    assert pd.Timestamp("2019-12-12") in mid and pd.Timestamp("2019-12-13") not in mid
    assert (mid.day <= 15).all()


def test_newey_west_uses_lag_h():
    """The DM standard error equals a hand-computed Newey-West (Bartlett, lag h) standard error.

    Hand formula for the mean of d (n values, u = d - mean):
        gamma_j = sum_t u_t u_(t-j) / n,  S = gamma_0 + 2 sum_(j=1..h) (1 - j / (h + 1)) gamma_j,
        se = sqrt(S / n), p-value from t(n - 1).
    """
    rng = np.random.default_rng(7)
    shocks = rng.normal(size=83)
    e_model = np.convolve(shocks, np.ones(4), mode="valid")  # overlapping sums: serially correlated
    e_bench = e_model + rng.normal(size=80)
    h = 4
    out = m2.diebold_mariano(e_model, e_bench, h)

    def hand_se(lag):
        d = e_model ** 2 - e_bench ** 2
        u, n = d - d.mean(), len(d)
        gamma = [u[j:] @ u[:n - j] / n for j in range(lag + 1)]
        s = gamma[0] + 2 * sum((1 - j / (lag + 1)) * gamma[j] for j in range(1, lag + 1))
        return np.sqrt(s / n)

    assert out["se"] == pytest.approx(hand_se(h), rel=1e-10)
    assert out["se"] != pytest.approx(hand_se(h - 1), rel=1e-3)  # the lag matters here
    assert out["p"] == pytest.approx(2 * stats.t.sf(abs(out["mean_d"] / out["se"]), 80 - 1))


def test_hln_matches_hand_formula():
    """Harvey-Leybourne-Newbold DM: at h = 1 it is the one-sample t-test of d; at h = 4 it matches the docstring formula."""
    rng = np.random.default_rng(5)
    e_model, e_bench = rng.normal(0, 1, 90), rng.normal(0, 1.3, 90)
    d = e_model ** 2 - e_bench ** 2
    one = m2.diebold_mariano(e_model, e_bench, 1)
    t_test = stats.ttest_1samp(d, 0.0)
    assert one["stat_hln"] == pytest.approx(t_test.statistic) and one["p_hln"] == pytest.approx(t_test.pvalue)

    h, n, u = 4, len(d), d - d.mean()
    lrv = (u @ u + 2 * sum(u[j:] @ u[:n - j] for j in range(1, h))) / n  # equal weights on lags 1..h-1
    hand = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    out = m2.diebold_mariano(e_model, e_bench, h)
    assert out["stat_hln"] == pytest.approx(hand)
    assert out["p_hln"] == pytest.approx(2 * stats.t.sf(abs(hand), n - 1))


def test_dm_sign_convention():
    """d = e_model^2 - e_bench^2: a model with smaller errors gets a negative statistic."""
    rng = np.random.default_rng(3)
    small, large = rng.normal(0, 1, 120), rng.normal(0, 5, 120)
    better = m2.diebold_mariano(small, large, 3)
    assert better["mean_d"] < 0 and better["stat"] < 0 and better["p"] < 0.01
    worse = m2.diebold_mariano(large, small, 3)
    assert worse["stat"] == pytest.approx(-better["stat"])


def test_fama_regression_recovers_known_coefficients():
    """On made-up data with a = 5 bp and b = 0.5 the regression finds them, and p_b1 matches statsmodels."""
    rng = np.random.default_rng(11)
    n = 200
    known = np.full(n, 1.0)                                # per cent
    priced_bp = rng.normal(0, 50, n)                       # forecast - known, bp
    actual_bp = 5 + 0.5 * priced_bp + rng.normal(0, 2, n)  # outcome - known, bp
    out = m2.fama_regression(known + actual_bp / 100, known, known + priced_bp / 100, h=3)
    assert out["a"] == pytest.approx(5, abs=1.0)
    assert out["b"] == pytest.approx(0.5, abs=0.02)
    assert out["p_b1"] < 0.001 and out["p_b0"] < 0.001

    res = m2.hac_ols(actual_bp, np.column_stack([np.ones(n), priced_bp]), 3)
    t_test = res.t_test((np.array([[0.0, 1.0]]), np.array([1.0])))
    assert out["p_b1"] == pytest.approx(float(t_test.pvalue))


def test_episodes_cover_sample():
    """The 7 episodes cover Jan 2017 - Sep 2026 month by month, without gaps or overlaps."""
    for (_, _, end), (_, start, _) in zip(m2.EPISODES, m2.EPISODES[1:]):
        assert pd.Period(start, "M") == pd.Period(end, "M") + 1
    months = pd.period_range("2017-01", "2026-09", freq="M").to_timestamp(how="end").normalize()
    names = m2.episode_of(months)
    assert len(set(names)) == 7
    assert m2.episode_of([pd.Timestamp("2020-12-31")])[0] == "Floor and YCC"
    assert m2.episode_of([pd.Timestamp("2022-04-29")])[0] == "Tightening"
    with pytest.raises(ValueError):
        m2.episode_of([pd.Timestamp("2016-12-30")])
