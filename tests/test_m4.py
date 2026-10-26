"""Tests for M4 (src/m4_hedging.py).

Each test checks one rule the hedging test relies on: the duration hedge is DV01-neutral, a factor
hedge has no in-sample exposure to the factors it was built on, the rolling loop never uses data
dated after t, H2 hedges a two-factor world perfectly, the paired test (Newey-West and block-clustered)
matches a hand calculation, the rounding floor and the fallback rule do what their docstrings say, and
the real-data run treats the suspect F17 days the way the README says.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import data
import m4_hedging as m4

MATS = [0.25 * k for k in range(1, 41)]  # 0.25 to 10 years, as in M3 and M4
PCS = ["PC1", "PC2", "PC3"]


def random_changes(n_days, seed):
    """Random daily changes (bp) at MATS on business days from 1 Jan 2015.

    Each day is a random walk across maturities (cumulative sum of independent steps) plus a little
    noise, so neighbouring maturities move together and PCA finds distinct PC1, PC2, PC3.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_days, len(MATS))).cumsum(axis=1) + rng.normal(0, 0.3, (n_days, len(MATS)))
    return pd.DataFrame(values, index=pd.bdate_range("2015-01-01", periods=n_days), columns=MATS)


def test_h1_weights_are_dv01_neutral():
    """H1 = 0.375 x 2y + 0.625 x 10y: weights sum to 1, hedge parallel and linear moves, not curvature."""
    w = m4.h1_weights()
    assert w.to_dict() == {2.0: 0.375, 10.0: 0.625}
    assert w.sum() == pytest.approx(1.0)                                    # DV01-neutral
    assert float(w.index.to_numpy() @ w.to_numpy()) == pytest.approx(7.0)  # interpolation in maturity
    m = np.array(MATS)
    days = pd.DataFrame([np.full(len(m), 3.0), 0.5 * m - 2.0, (m - 5.0) ** 2], columns=MATS)
    r = m4.residuals(days, w)
    assert r.iloc[0] == pytest.approx(0.0, abs=1e-12)   # parallel 3 bp move
    assert r.iloc[1] == pytest.approx(0.0, abs=1e-12)   # change linear in maturity
    assert r.iloc[2] == pytest.approx(4.0 - (0.375 * 9.0 + 0.625 * 25.0))  # a hump is not hedged
    assert m4.residuals(days, pd.Series(dtype=float)).equals(days[7.0])   # H0: no weights, r = dy7


def test_h3_residual_has_no_pc1_3_exposure_in_sample():
    """On real 2022-2023 changes, H3 (H2) zeroes the exposure to PC1-3 (PC1-2), so its in-sample residual is
    uncorrelated with those factor scores; the exposure to PC4 is left, so the check is not empty."""
    changes = data.daily_changes_bp(data.load_f17("yields"), MATS).loc["2022-01-01":"2023-12-31"]
    eig, load, _ = m4.pca_loadings(changes)
    w = m4.hedge_weights(load)
    np.testing.assert_allclose(m4.factor_exposure(load, w["H3"], PCS), 0.0, atol=1e-12)
    np.testing.assert_allclose(m4.factor_exposure(load, w["H2"], PCS[:2]), 0.0, atol=1e-12)
    r = m4.residuals(changes, w["H3"])
    scores = (changes - changes.mean()) @ load[PCS + ["PC4"]]       # factor scores f_k = demeaned changes . L_k
    cov = np.array([np.cov(r, scores[pc])[0, 1] for pc in PCS + ["PC4"]])
    # cov(r, f_k) = eigenvalue_k x exposure_k: zero for PC1-3, not for PC4.
    np.testing.assert_allclose(cov[:3], 0.0, atol=1e-8)
    expected_pc4 = eig["PC4"] * m4.factor_exposure(load, w["H3"], ["PC4"]).iloc[0]
    assert cov[3] == pytest.approx(expected_pc4) and abs(cov[3]) > 1e-3


def test_rolling_loop_has_no_look_ahead():
    """Windows end at t, evaluation starts after t, blocks tile the later days, and changing data after a
    block's t leaves that block's weights (and earlier residuals) unchanged."""
    changes = random_changes(800, seed=3)
    resid, blocks = m4.rolling_hedges(changes, changes, window=500, step=60)
    for b in m4.block_schedule(changes.index, changes.index, window=500, step=60):
        assert len(b["est"]) == 500 and b["est"].max() == b["t"] < b["eval"].min()
    assert list(resid.index) == list(changes.index[500:])     # first t = 500th day; every later day once
    assert blocks["n eval"].tolist() == [60] * 5

    t1 = blocks["t"].iloc[1]
    shocked = changes.copy()
    later = shocked.index > t1
    shocked.loc[later] = 10 * random_changes(800, seed=4).to_numpy()[later]
    resid2, blocks2 = m4.rolling_hedges(shocked, shocked, window=500, step=60)
    pd.testing.assert_frame_equal(blocks2.loc[:1, m4.WEIGHT_COLS], blocks.loc[:1, m4.WEIGHT_COLS])
    pd.testing.assert_frame_equal(resid2[resid2["block"] == 0], resid[resid["block"] == 0])
    assert not np.allclose(blocks2.loc[2:, m4.WEIGHT_COLS], blocks.loc[2:, m4.WEIGHT_COLS])


def test_two_factor_world_h2_hedges_perfectly():
    """If curves move only by level and a curved slope, H2 leaves no residual out of sample; H1 does,
    because the slope is not linear in maturity."""
    rng = np.random.default_rng(5)
    m, n = np.array(MATS), 700
    level, slope = np.ones_like(m), 1.0 - np.exp(-m / 2.0)
    values = np.outer(rng.normal(0, 5, n), level) + np.outer(rng.normal(0, 3, n), slope)
    changes = pd.DataFrame(values, index=pd.bdate_range("2015-01-01", periods=n), columns=MATS)
    resid, blocks = m4.rolling_hedges(changes, changes, window=500, step=60)
    assert len(resid) == 200 and len(blocks) == 4                 # blocks of 60, 60, 60, 20 days
    assert resid["H2"].abs().max() < 1e-8
    assert resid["H1"].std() > 0.1


def test_dm_test_matches_hand_calculation():
    """dm_test = mean of d = r_model^2 - r_bench^2 over its Bartlett (Newey-West) standard error, t(n-1)
    p-value; the hedge with smaller residuals gets a negative t. Also pins the lag rule and the
    block-clustered standard error."""
    rng = np.random.default_rng(6)
    n, lags = 300, 4
    r_model, r_bench = rng.normal(0, 1.0, n), rng.normal(0, 1.2, n)
    out = m4.dm_test(r_model, r_bench, lags)
    d = r_model ** 2 - r_bench ** 2
    u = d - d.mean()
    gamma = [u[j:] @ u[:n - j] / n for j in range(lags + 1)]
    lrv = gamma[0] + 2 * sum((1 - j / (lags + 1)) * gamma[j] for j in range(1, lags + 1))
    t = d.mean() / np.sqrt(lrv / n)
    assert out["mean d"] == pytest.approx(d.mean())
    assert out["t"] == pytest.approx(t)
    assert out["p"] == pytest.approx(2 * stats.t.sf(abs(t), n - 1))
    assert out["t"] < 0
    assert (m4.nw_lags(100), m4.nw_lags(1943)) == (4, 7)

    # Block-clustered version: var(mean d) = G / (G - 1) x sum_g (sum of u in block g)^2 / n^2, p from t(G - 1).
    groups = np.repeat(np.arange(10), 30)                                   # 10 blocks of 30 days
    out_cl = m4.dm_test(r_model, r_bench, groups=groups)
    sums = np.array([u[groups == g].sum() for g in range(10)])
    t_cl = d.mean() / np.sqrt(10 / 9 * (sums ** 2).sum() / n ** 2)
    assert out_cl["se type"] == "cluster, 10 blocks" and out_cl["t"] == pytest.approx(t_cl)
    assert out_cl["p"] == pytest.approx(2 * stats.t.sf(abs(t_cl), 9))


def test_summary_floor_and_fallback():
    """Rounding floor = sqrt((1 + sum w^2) / 6); variance removed vs H0; the fallback swaps H3 for H2
    only in blocks with cond > threshold and leaves its input unchanged."""
    resid = pd.DataFrame({"block": [0, 0, 1, 1], "H0": [2.0, -2.0, 4.0, -4.0], "H1": [1.0, -1.0, 1.0, -1.0],
                          "H2": [0.5, -0.5, 0.5, -0.5], "H3": [0.1, -0.1, 9.0, -9.0]})
    blocks = pd.DataFrame({"block": [0, 1], "cond (2/5/10y)": [5.0, 50.0], **{c: 0.0 for c in m4.WEIGHT_COLS}})
    blocks["H1 w2"], blocks["H1 w10"] = 0.375, 0.625
    s = m4.summary_table(resid, blocks).set_index("hedge")
    assert s.loc["H0", "rounding floor sd"] == pytest.approx(np.sqrt(1 / 6))
    assert s.loc["H1", "rounding floor sd"] == pytest.approx(np.sqrt((1 + 0.375 ** 2 + 0.625 ** 2) / 6))
    assert s.loc["H1", "variance removed %"] == pytest.approx(100 * (1 - resid["H1"].var() / resid["H0"].var()))
    fb = m4.with_fallback(resid, blocks, threshold=10)
    assert fb["H3"].tolist() == [0.1, -0.1, 0.5, -0.5] and resid["H3"].iloc[2] == 9.0


def test_real_data_run():
    """On F17: first t is 31 Dec 2018; no estimation window holds a suspect-day change; the four suspect-day
    changes are still evaluated; the keep-suspect run evaluates the same days; H3's in-sample exposure is 0."""
    yields = data.load_f17("yields")
    clean = data.daily_changes_bp(yields, MATS)
    every = data.daily_changes_bp(yields, MATS, drop_suspect=False)
    suspect = every.index.difference(clean.index)
    resid, blocks = m4.rolling_hedges(clean, every)
    first_t = blocks["t"].iloc[0]
    assert first_t == pd.Timestamp("2018-12-31") and len(suspect) == 4
    assert not any(b["est"].isin(suspect).any() for b in m4.block_schedule(clean.index, every.index))
    assert resid.index.equals(every.index[every.index > first_t]) and suspect.isin(resid.index).all()
    resid_k, _ = m4.rolling_hedges(every, every, first_t=first_t)
    assert resid_k.index.equals(resid.index)
    assert blocks["max |H3 exposure|"].max() < 1e-12
