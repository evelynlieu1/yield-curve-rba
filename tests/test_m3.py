"""Tests for M3 (src/m3_factors.py) and the shared helper data.daily_changes_bp.

Each test checks one rule M3 and M4 rely on: the PCA sign rules, that the
variance shares add up, that PCA finds factors we planted ourselves, that a
swap of PC3 and PC4 is caught, the period bounds, the report's number
formatting, and that the suspect F17 days are dropped the way the README says.
"""

import numpy as np
import pandas as pd
import pytest

import data
import m1_validate
import m3_factors as m3

MATS = [0.25 * k for k in range(1, 41)]  # 0.25 to 10 years, as in M3


def synthetic_changes(n_days=2000, seed=0, noise_bp=0.2):
    """Daily changes (bp) built from known level, slope and curvature factors.

    level - the same move at every maturity; slope - a move that rises
    linearly with maturity; hump - the parabola -(m - 3)^2 with its level
    and slope parts removed (so it is orthogonal to both; the hump then
    peaks near 5y and is positive at 5y). All scaled to length 1, like PCA
    loadings. Factor sizes: level sd 6 bp, slope sd 2 bp, hump sd 1 bp, plus
    independent noise of sd noise_bp at each maturity.
    Returns (changes DataFrame, level, slope, hump Series).
    """
    rng = np.random.default_rng(seed)
    m = np.array(MATS)
    level = np.ones_like(m) / np.linalg.norm(np.ones_like(m))
    slope = (m - m.mean()) / np.linalg.norm(m - m.mean())
    hump = -(m - 3.0) ** 2
    hump = hump - level * (level @ hump) - slope * (slope @ hump)   # Gram-Schmidt against level and slope
    hump = hump / np.linalg.norm(hump)
    values = (np.outer(rng.normal(0, 6, n_days), level) + np.outer(rng.normal(0, 2, n_days), slope)
              + np.outer(rng.normal(0, 1, n_days), hump) + rng.normal(0, noise_bp, (n_days, len(m))))
    return (pd.DataFrame(values, columns=MATS), pd.Series(level, index=MATS),
            pd.Series(slope, index=MATS), pd.Series(hump, index=MATS))


def test_sign_rules():
    """PC1 sums > 0, PC2 > 0 at 10y, PC3 > 0 at 5y; negated loadings are flipped back."""
    rng = np.random.default_rng(1)
    changes = pd.DataFrame(rng.normal(size=(500, len(MATS))), columns=MATS).cumsum(axis=1)  # smooth-ish curves
    _, load, _ = m3.pca_loadings(changes)
    assert load["PC1"].sum() > 0
    assert load.loc[10.0, "PC2"] > 0
    assert load.loc[5.0, "PC3"] > 0
    # Every sign wrong on purpose: the rules must restore the original.
    pd.testing.assert_frame_equal(m3.apply_sign_rules(-load[["PC1", "PC2", "PC3"]]), load[["PC1", "PC2", "PC3"]])
    # Without a 10y or 5y column the nearest maturity is used (0.5-10 robustness keeps both; this checks
    # the fallback on a grid that has neither).
    short = load.loc[[0.25, 0.5, 1.0, 4.75, 9.75], ["PC1", "PC2", "PC3"]]
    signed = m3.apply_sign_rules(-short)
    assert signed.loc[9.75, "PC2"] > 0 and signed.loc[4.75, "PC3"] > 0


def test_variance_shares_sum_to_one_and_vectors_are_orthonormal():
    """Shares sum to 1, eigenvalues are largest first and add to the total variance; loadings are orthonormal."""
    changes = synthetic_changes()[0]
    eig, load, share = m3.pca_loadings(changes)
    assert share.sum() == pytest.approx(1.0)
    assert (np.diff(eig.to_numpy()) <= 1e-12).all()
    assert eig.sum() == pytest.approx(np.trace(changes.cov().to_numpy()))
    np.testing.assert_allclose(load.to_numpy().T @ load.to_numpy(), np.eye(len(MATS)), atol=1e-10)
    assert list(load.index) == MATS and list(load.columns[:3]) == ["PC1", "PC2", "PC3"]
    # Correlation-matrix PCA: each maturity has variance 1, so the eigenvalues add to 40.
    eig_c, _, share_c = m3.pca_loadings(changes, standardise=True)
    assert eig_c.sum() == pytest.approx(len(MATS)) and share_c.sum() == pytest.approx(1.0)
    # Missing data must not be dropped silently.
    with pytest.raises(ValueError):
        m3.pca_loadings(changes.mask(changes.index == 0))


def test_synthetic_level_slope_and_curvature_are_recovered():
    """PCA on changes built from known level, slope and curvature returns them as PC1, PC2, PC3, signed."""
    changes, level, slope, hump = synthetic_changes()
    assert hump.loc[5.0] > 0                     # so the PC3 sign rule (> 0 at 5y) must give the planted sign
    _, load, share = m3.pca_loadings(changes)
    # Signed dot products: the sign rules must also give the planted signs (level sums > 0, slope > 0 at 10y).
    assert float(load["PC1"] @ level) > 0.999
    assert float(load["PC2"] @ slope) > 0.999
    assert float(load["PC3"] @ hump) > 0.99
    # Planted variances: level 36, slope 4, hump 1, noise 0.04 x 40 = 1.6 (bp^2/day). Shares follow.
    assert share["PC1"] == pytest.approx(36 / 42.6, abs=0.02)
    assert share["PC3"] == pytest.approx(1 / 42.6, abs=0.005)
    assert share.iloc[:3].sum() > 0.95


def test_best_match_catches_a_swap():
    """If a period's PC3 and PC4 trade places, best_match and compare_pcs say so instead of 'PC1 PC2 PC3'."""
    _, ref, _ = m3.pca_loadings(synthetic_changes()[0])
    swapped = ref.rename(columns={"PC3": "PC4", "PC4": "PC3"})[ref.columns]
    assert m3.compare_pcs(ref, ref)["best match"] == "PC1 PC2 PC3"
    assert m3.compare_pcs(swapped, ref)["best match"] == "PC1 PC2 PC4"
    assert m3.best_match(ref[["PC3"]], swapped)[0][0] == "PC4"   # the 'main PC3 found as' column
    # A sign flip alone is not a swap: cosine 1, flagged in 'sign flipped'.
    row = m3.compare_pcs(ref.assign(PC3=-ref["PC3"]), ref)
    assert row["best match"] == "PC1 PC2 PC3" and row["sign flipped"] == "PC3"
    assert row["cos PC3"] == pytest.approx(1.0)


def test_period_rules():
    """Bounds are inclusive calendar dates on the date a change ends; the YCC regime windows start on 20 Mar 2020."""
    changes = data.daily_changes_bp(data.load_f17("yields"), MATS)
    ycc = m3.slice_period(changes, m3.PERIODS[m3.YCC_LABEL])
    assert (ycc.index[0], ycc.index[-1], len(ycc)) == (pd.Timestamp("2020-03-02"), pd.Timestamp("2021-11-30"), 443)
    regime = m3.slice_period(changes, list(m3.YCC_VARIANTS.values())[0])
    assert (regime.index[0], regime.index[-1], len(regime)) == (pd.Timestamp("2020-03-20"),
                                                                pd.Timestamp("2021-11-01"), 408)
    calm = m3.slice_period(changes, list(m3.YCC_VARIANTS.values())[1])
    assert calm.index[-1] == pd.Timestamp("2021-10-26") and len(calm) == 404

    def periods_of(day):
        """Labels of the (non full-sample) periods that hold the change dated 'day'."""
        return [k for k, b in m3.PERIODS.items() if k != "Full sample" and day in m3.slice_period(changes, b).index]

    for day, expected in [("2019-12-31", ["2017-2019"]), ("2020-01-02", []), ("2021-12-01", []),
                          ("2020-03-02", [m3.YCC_LABEL])]:
        assert pd.Timestamp(day) in changes.index
        assert periods_of(pd.Timestamp(day)) == expected


def test_report_text_helpers():
    """to_text never prints '-0.00' and keeps signs; zero_crossings finds each sign change of a loading."""
    frame = pd.DataFrame({"x": [-0.0004, 0.123, -0.126], "y": [1.0, -2.0, 0.04]})
    out = m3.to_text(frame, {"x": 2, "y": 1}, signed=["x"])
    assert list(out["x"]) == ["+0.00", "+0.12", "-0.13"] and list(out["y"]) == ["1.0", "-2.0", "0.0"]
    assert frame["x"].iloc[0] == -0.0004                 # the input (and so the CSV) keeps the float
    s = pd.Series([-1.0, -0.5, 0.5, 1.0, -1.0], index=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert m3.zero_crossings(s) == [(2.0, 3.0), (4.0, 5.0)]


def test_aligned_cosine():
    """Cosine is 1 for the same shape whatever its sign, says when a flip was needed, and uses common maturities."""
    a = pd.Series([0.1, 0.2, 0.3, 0.4], index=[0.25, 0.5, 1.0, 2.0])
    assert m3.aligned_cosine(a, -2 * a) == (pytest.approx(1.0), True)
    assert m3.aligned_cosine(a, a) == (pytest.approx(1.0), False)
    b = pd.Series([0.2, 0.3, 0.4], index=[0.5, 1.0, 2.0])  # no 0.25 point, as in the 0.5-10 variant
    assert m3.aligned_cosine(a, b)[0] == pytest.approx(1.0)
    c = pd.Series([1.0, -1.0, 0.0, 0.0], index=a.index)
    d = pd.Series([1.0, 1.0, 0.0, 0.0], index=a.index)
    assert m3.aligned_cosine(c, d)[0] == pytest.approx(0.0)


def test_suspect_day_changes_are_dropped():
    """daily_changes_bp drops the change into and out of each suspect day, and gives bp."""
    dates = pd.to_datetime(["2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-18"])
    yields = pd.DataFrame({0.25: [1.50, 1.51, 1.36, 1.51, 1.52], 10.0: [2.00, 2.00, 2.01, 2.02, 2.02]},
                          index=dates)
    kept = data.daily_changes_bp(yields, [0.25, 10.0], drop_suspect=False)
    assert list(kept.index) == list(dates[1:])
    assert kept.loc["2019-02-14", 0.25] == -15.0 and kept.loc["2019-02-15", 0.25] == 15.0
    dropped = data.daily_changes_bp(yields, [0.25, 10.0], drop_suspect=True)
    # Into 14 Feb (row 14 Feb) and out of it (row 15 Feb, the next F17 day) are gone; 13 and 18 Feb stay.
    assert list(dropped.index) == [pd.Timestamp("2019-02-13"), pd.Timestamp("2019-02-18")]
    assert dropped.loc["2019-02-18", 0.25] == 1.0


def test_suspect_days_on_real_data_and_shared_with_m1():
    """On F17 exactly 4 changes go (14-15 Feb 2019, 13-16 Dec 2019), and M1 uses the same list."""
    yields = data.load_f17("yields")
    every = data.daily_changes_bp(yields, MATS, drop_suspect=False)
    main = data.daily_changes_bp(yields, MATS)
    gone = every.index.difference(main.index)
    assert list(gone) == list(pd.to_datetime(["2019-02-14", "2019-02-15", "2019-12-13", "2019-12-16"]))
    assert len(every) == len(yields) - 1 and not main.isna().any().any()
    assert m1_validate.SUSPECT_DAYS is data.SUSPECT_DAYS
