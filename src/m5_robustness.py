"""M5 - robustness: which results survive? the README, "M5 - Robustness and write-up".

Four parts, each reusing the M2 / M4 functions by importing them (nothing is copied): 1. the OIS
benchmark (OIS, the F17 zero yield and no change as forecasts of the average actual cash rate over
the next 1, 3, 6 months); 2. the M2 variants in one table, plus 2022 onwards; 3. H0-H3 rerun with the
M4 code, one change at a time; 4. a 5-year bond hedged on F2 bond yields, and the same test on F17.
The planned designs (M2, M4) stay the headline; every variant here is robustness. The M4 PCA on
1-10y was found after seeing the M4 results and is labelled that way wherever it appears.
Licensing (data.load_f1, data.load_f2): F1's OIS columns are FENICS data and F2's licence is
unchecked. Nothing here writes a series from either file, or a per-date series built from them
(forecasts, errors, daily residuals), to outputs/: only statistics and charts of statistics.
Writes outputs/m5/ (robustness_report.md, two PNGs, a CSV per table).
Run from the project folder:  python src/m5_robustness.py
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no screen needed
import matplotlib.pyplot as plt

# Make "import data" work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    PROJECT_DIR, SUSPECT_DAYS, F2_MATURITIES, load_f17, load_a2, load_f1, load_f2, daily_changes_bp,
)
from m1_validate import COLORS, YCC_REGIMES, md_table, style_axes  # noqa: E402  (M1 style, YCC dates)
import data  # noqa: E402  (file names for the missing-file check)
import m2_expectations as m2  # noqa: E402  (panels, DM / HAC tests, episodes)
import m4_hedging as m4  # noqa: E402  (rolling hedge test, paired tests)
from m3_factors import MATURITIES, to_text, pca_loadings  # noqa: E402

# Settings

OUT_DIR = PROJECT_DIR / "outputs" / "m5"
REPORT_PATH = OUT_DIR / "robustness_report.md"

OIS_HORIZONS = [1, 3, 6]                                  # months (the README M5)
OIS_COLUMN = {1: "ois_1m", 3: "ois_3m", 6: "ois_6m"}      # data.load_f1 column for each horizon
OIS_FORECASTS = {"ois": "OIS", "f17_zero": "F17 zero", "no_change": "no change"}
OIS_PAIRS = [("ois", "f17_zero"), ("ois", "no_change"), ("f17_zero", "no_change")]  # model vs bench
DAYS_PER_YEAR = 365.0          # AUD money-market day count (actual/365)
MAX_F1_GAP_DAYS = 7            # longest gap between F1 dates the calendar fill accepts (Easter 2011: 6)

LEFT_OUT = {"2020-2021": ("2020-01-01", "2021-12-31"), "2022-2023": ("2022-01-01", "2023-12-31")}
FIVE_YEAR = 5.0                # target of the 5-year variant and of the F2 check
FIVE_YEAR_HEDGES = {"H2": ([2.0, 10.0], ["PC1", "PC2"]), "H3": ([2.0, 3.0, 10.0], ["PC1", "PC2", "PC3"])}
ALL_PAIRS = m4.PAIRS + [("H2", "H1")]   # M4's two paired tests, plus level + slope vs duration
EX_POST = "found after seeing the M4 results"            # label of the 1-10y PCA, used everywhere
EX_POST_M2 = "added after seeing the M2 results"         # label of M2's 'excluding Tightening' row
BIG_WEIGHT = 5.0               # F2 ex-post split: blocks with an H3 weight above this in size (judgement)
HEDGE_COLORS = {"H1": COLORS["blue"], "H2": COLORS["orange"], "H3": COLORS["aqua"]}
FORECAST_COLORS = {"ois": COLORS["blue"], "f17_zero": COLORS["orange"], "no_change": COLORS["aqua"]}


# 1. OIS benchmark

def calendar_cash_rate(cash_rate):
    """The actual cash rate on every calendar day.

    Input:  cash_rate - Series data.load_f1()['cash_rate'] (per cent, one value per F1 date).
    Output: Series indexed by every calendar day from the first to the last F1 date. A day with no
            F1 row (weekend, public holiday) carries the rate of the latest F1 date before it: an
            overnight loan made on a Friday runs to Monday, so Friday's rate applies to 3 nights.
    Raises: ValueError if a rate is blank or two F1 dates are more than MAX_F1_GAP_DAYS apart, because
            the fill would then invent data.
    """
    if cash_rate.isna().any():
        raise ValueError("calendar_cash_rate: blank cash rate in F1")
    gaps = cash_rate.index.to_series().diff().dt.days
    if gaps.max() > MAX_F1_GAP_DAYS:
        raise ValueError(f"calendar_cash_rate: {gaps.max():.0f}-day gap between F1 dates")
    days = pd.date_range(cash_rate.index[0], cash_rate.index[-1], freq="D")
    return cash_rate.reindex(days).ffill()


def realised_average(daily, t, h):
    """Average actual cash rate over the calendar days in (t, t + h months].

    Inputs: daily - calendar_cash_rate output; t - forecast date; h - months.
    Output: (average in per cent, n_days). The window is day t + 1 up to and including t + h months
            (pandas DateOffset, as in M2), so n_days = (t + h months - t) in days. The night of t is
            not in it: its rate is known at the close on t (it is the no-change forecast). If t is a
            Friday, Saturday and Sunday still carry Friday's rate, because those nights come after t.
            NaN if the window runs past the last day of 'daily'.
    """
    end = pd.Timestamp(t) + pd.DateOffset(months=h)
    n_days = (end - pd.Timestamp(t)).days
    if end > daily.index[-1]:
        return np.nan, n_days
    return float(daily.loc[pd.Timestamp(t) + pd.Timedelta(days=1):end].mean()), n_days


def ois_average_basis(quote, n_days):
    """Turn an OIS quote into the average daily rate it implies (per cent in, per cent out).

    An OIS rate q is quoted as simple interest on actual/365 against daily compounding:
    1 + q n / 365 = product over the n days of (1 + r_d / 365). Taking logs, the right side is
    sum ln(1 + r_d / 365), which equals sum r_d / 365 to within 0.02 bp at these rates, so the
    average daily rate the quote implies is ln(1 + q n / 365) x 365 / n. That is a little below q
    (by about q^2 n / 730: about 2.7 bp for a 6-month quote at 3.3%, under 0.6 bp at 1.5%). The
    outcome (a plain average) and the F17 zero yield (continuously compounded) are on this basis.
    """
    tau = np.asarray(n_days, dtype=float) / DAYS_PER_YEAR
    return 100.0 * np.log1p(np.asarray(quote, dtype=float) / 100.0 * tau) / tau


def f17_zero_at(yields, dates, h):
    """F17 zero-coupon yield at h/12 years on each date (per cent).

    For h = 3 and 6 the maturity is on the 0.25-year grid and np.interp returns the grid value
    itself. For h = 1 (1/12 year) it interpolates linearly between the 0 and 0.25-year points.
    Why that is the forward average: under continuous compounding y(m) is the average of the
    instantaneous forwards over [0, m] (M1, check 9). If the forward is linear on [0, 0.25],
    f(s) = a + b s, then y(m) = a + b m / 2 is linear too, so linear interpolation is exact.
    """
    rows = yields.loc[pd.DatetimeIndex(dates)]
    grid = np.asarray(rows.columns, dtype=float)
    return pd.Series([np.interp(h / 12, grid, r) for r in rows.to_numpy()], index=rows.index)


def ois_panel(f1, yields, dates, h):
    """One row per forecast date t: the three forecasts, the outcome and the errors.

    Inputs: f1 - data.load_f1(); yields - load_f17('yields'); dates - month-end F17 dates
            (m2.month_end_dates); h - months.
    Output: DataFrame (kept in memory only, never written: it holds F1 values) with t, h, episode
            (m2.episode_of), n_days, ois_quote (F1 h-month OIS on t, NaN after 1 Dec 2022), ois
            (ois_average_basis), f17_zero (f17_zero_at), no_change (F1 cash rate on t), outcome
            (realised_average) and err_ois, err_f17_zero, err_no_change = (outcome - forecast) in bp.
            Rows kept: the whole outcome window lies inside F1.
    No look-ahead: every forecast reads only the F1 and F17 rows dated t; only the outcome reads
    later rows, and it averages the nights after t only.
    Raises ValueError if a date is missing from F1 or F17.
    """
    t = pd.DatetimeIndex(dates)
    if not t.isin(f1.index).all() or not t.isin(yields.index).all():
        raise ValueError("ois_panel: a forecast date is missing from F1 or F17")
    daily = calendar_cash_rate(f1["cash_rate"])
    outcome = [realised_average(daily, d, h) for d in t]
    panel = pd.DataFrame({"t": t, "h": h, "episode": m2.episode_of(t),
                          "n_days": [n for _, n in outcome], "outcome": [v for v, _ in outcome]})
    panel["ois_quote"] = f1.loc[t, OIS_COLUMN[h]].to_numpy()
    panel["ois"] = ois_average_basis(panel["ois_quote"], panel["n_days"])
    panel["f17_zero"] = f17_zero_at(yields, t, h).to_numpy()
    panel["no_change"] = f1.loc[t, "cash_rate"].to_numpy()
    for name in OIS_FORECASTS:
        panel[f"err_{name}"] = (panel["outcome"] - panel[name]) * m2.BP
    return panel[panel["outcome"].notna()].reset_index(drop=True)


def forecast_tables(panels, forecasts, pairs):
    """Accuracy, Diebold-Mariano and by-episode tables for a set of forecasts.

    Inputs: panels - dict h -> panel (ois_panel rows to use); forecasts - panel names, e.g.
            ['ois', 'f17_zero', 'no_change']; pairs - (model, bench) for the paired tests.
    Output: (accuracy, dm, episodes) DataFrames, floats unrounded.
      accuracy - per h and forecast: n, bias (mean error, bp), 'bias p' (errors on a constant,
                 Newey-West lag h, m2.hac_ols), MAE, RMSE (bp; m2.error_stats).
      dm       - per h and pair: m2.diebold_mariano on squared errors, d = e_model^2 - e_bench^2
                 (negative = model better): mean d, Newey-West t and p (lag h), HLN p.
      episodes - per h and the README episode: n, then bias and RMSE of each forecast.
    """
    acc, dm, ep = [], [], []
    for h, p in panels.items():
        for name in forecasts:
            e = p[f"err_{name}"].to_numpy()
            s = m2.error_stats(e)
            acc.append({"h": h, "forecast": OIS_FORECASTS[name], "n": s["n"], "bias (bp)": s["bias"],
                        "bias p": m2.hac_ols(e, np.ones((len(e), 1)), h).pvalues[0],
                        "MAE (bp)": s["mae"], "RMSE (bp)": s["rmse"]})
        for a, b in pairs:
            r = m2.diebold_mariano(p[f"err_{a}"], p[f"err_{b}"], h)
            dm.append({"h": h, "test": f"{OIS_FORECASTS[a]} vs {OIS_FORECASTS[b]}", "n": r["n"],
                       "mean d (bp^2)": r["mean_d"], "t": r["stat"], "p": r["p"], "HLN p": r["p_hln"]})
        for name, _, _ in m2.EPISODES:
            part = p[p["episode"] == name]
            if part.empty:
                continue
            row = {"h": h, "episode": name, "n": len(part)}
            for f in forecasts:
                s = m2.error_stats(part[f"err_{f}"])
                row.update({f"bias {OIS_FORECASTS[f]}": s["bias"], f"RMSE {OIS_FORECASTS[f]}": s["rmse"]})
            ep.append(row)
    return pd.DataFrame(acc), pd.DataFrame(dm), pd.DataFrame(ep)


def plot_ois_episodes(episodes, path, period):
    """ois_rmse_by_episode.png: RMSE (bp) of OIS, F17 zero and no change by episode, one panel per
    horizon, on the common sample (forecast dates with all three). Statistics only: no per-date
    values from F1 are drawn. Inputs: episodes - the 'episodes' table of forecast_tables; period -
    text for the title, e.g. 'Jan 2017 - Nov 2022' (computed in main from the sample, not typed)."""
    names = list(OIS_FORECASTS)
    fig, axes = plt.subplots(1, len(OIS_HORIZONS), figsize=(11, 4.6), sharey=True, facecolor=COLORS["surface"])
    for ax, h in zip(axes, OIS_HORIZONS):
        part = episodes[episodes["h"] == h].reset_index(drop=True)
        x, width = np.arange(len(part)), 0.26
        for k, f in enumerate(names):
            ax.bar(x + (k - 1) * width, part[f"RMSE {OIS_FORECASTS[f]}"], width=width - 0.03,
                   color=FORECAST_COLORS[f], label=OIS_FORECASTS[f])
        labels = [f"{textwrap.fill(e, 10)}\n(n = {n})" for e, n in zip(part["episode"], part["n"])]
        ax.set_xticks(x, labels, fontsize=7)
        style_axes(ax, f"h = {h} month{'s' if h > 1 else ''}", "RMSE (bp)" if h == OIS_HORIZONS[0] else "")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=COLORS["ink"], loc="upper left")
    fig.suptitle(f"Error in forecasting the average actual cash rate, by episode (forecasts {period})",
                 fontsize=10, color=COLORS["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 2. M2 robustness

def m2_robustness(forwards, yields, changes):
    """The M2 variants in one table, plus forecasts made from Jan 2022 on.

    Inputs: load_f17('forwards'), load_f17('yields'), load_a2().
    Output: one m2.robustness_row per (variant, h) for h in m2.ALL_HORIZONS (3 and 6 main, 12 the
            robustness horizon): forward bias, RMSEs, DM t / p (forward vs no change) and Fama b.
    Panels come from m2.build_panel exactly as in M2's main(); the subsets are rows of the main panel.
    """
    month_ends = m2.month_end_dates(forwards.index)
    mid = m2.mid_month_dates(forwards.index, skip=pd.DatetimeIndex(sorted(SUSPECT_DAYS)))
    rows = []
    for h in m2.ALL_HORIZONS:
        main = m2.build_panel(forwards, yields, changes, month_ends, h)
        variants = {"main (month-end, t + h months)": main,
                    "outcome date t + days": m2.build_panel(forwards, yields, changes, month_ends, h, rule="days"),
                    "mid-month forecast dates": m2.build_panel(forwards, yields, changes, mid, h),
                    "excluding Floor and YCC": main[main["episode"] != "Floor and YCC"],
                    f"excluding Tightening ({EX_POST_M2})": main[main["episode"] != "Tightening"],
                    "2022 onwards (new in M5)": main[main["t"] >= pd.Timestamp("2022-01-01")]}
        rows += [m2.robustness_row(name, p, h) for name, p in variants.items()]
    return pd.DataFrame(rows)


# 3. M4 robustness

def run_hedges(clean, every, **kwargs):
    """One rolling hedge run the M4 way (m4.rolling_hedges): estimate on 'clean' (suspect-day
    changes out), evaluate on 'every', then drop the suspect-day changes from the days the
    statistics use, as M4's main run does. kwargs: window, step, first_t, target, factor_hedges.
    Output: (resid on the statistics days, blocks)."""
    resid, blocks = m4.rolling_hedges(clean, every, **kwargs)
    return resid[resid.index.isin(clean.index)], blocks


def hedge_row(label, resid, blocks):
    """One row of the M4 and F2 tables.

    m4.variant_row (sd per hedge, H3's variance removed, paired t at Newey-West lag nw_lags(n) and
    block-clustered, H3 vs H1 and H3 vs H2), plus for each pair in ALL_PAIRS the block-clustered p
    ('p cl'; t(G - 1), M4's conservative headline) and t, 'H3 < H1 blocks' (blocks where H3's mean
    squared residual is below H1's) and 'max abs H3 w' (largest H3 weight in any block used; large =
    an ill-conditioned hedge). d = r_model^2 - r_bench^2, negative t = the first hedge is better.
    """
    row = m4.variant_row(label, resid)
    for a, b in ALL_PAIRS:
        r = m4.dm_test(resid[a], resid[b], groups=resid["block"])
        row[f"t cl {a} vs {b}"], row[f"p cl {a} vs {b}"] = r["t"], r["p"]
    msq = (resid[["H1", "H3"]] ** 2).groupby(resid["block"]).mean()
    row["H3 < H1 blocks"] = int((msq["H3"] < msq["H1"]).sum())
    used = blocks[blocks["block"].isin(resid["block"])]
    row["max abs H3 w"] = float(h3_size(used).max())
    return row


def h3_size(blocks):
    """Largest |H3 weight| of each block (Series aligned with 'blocks'). The H3 weight columns are
    'H3 w2', 'H3 w5', ... (m4.rolling_hedges); 'H3 total' is not a weight and is left out."""
    return blocks[[c for c in blocks.columns if c.startswith("H3 w")]].abs().max(axis=1)


def worst_block_table(runs):
    """For each run (label -> (resid, blocks)): the block where H3's residual sd is highest, with its
    evaluation dates, 'cond (2/5/10y)', 'cond (H3 legs)' (only where computed: the F2 check, whose H3
    is on 2y/3y/10y; NaN elsewhere), largest |H3 weight| and the block sds of H1-H3 (m4.with_block_sd
    on the run's statistics days). Shows what breaks H3 in each run."""
    rows = []
    for label, (resid, blocks) in runs.items():
        sized = m4.with_block_sd(blocks[blocks["block"].isin(resid["block"])], resid)
        k = sized["block sd H3"].idxmax()
        rows.append({"rows": label, "eval first": sized.loc[k, "eval first"], "eval last": sized.loc[k, "eval last"],
                     "cond (2/5/10y)": sized.loc[k, "cond (2/5/10y)"],
                     "cond (H3 legs)": sized.loc[k, "cond (H3 legs)"] if "cond (H3 legs)" in sized else np.nan,
                     "max abs H3 w": h3_size(sized).loc[k],
                     **{f"block sd {h}": sized.loc[k, f"block sd {h}"] for h in ("H1", "H2", "H3")}})
    return pd.DataFrame(rows)


def m4_robustness(clean, every):
    """H0-H3 rerun one change at a time from the planned M4 design (window 500, step 60, 7-year target,
    PCA on 0.25-10y). Inputs: F17 daily changes without / with the suspect days (data.daily_changes_bp).
    Output: (table, runs). table - one hedge_row per run, plus 'design' ('headline', 'planned variant',
    'comparator' or 'ex post') and 'target' (years); runs - label -> (resid, blocks).
    Every run starts at the planned first t (31 Dec 2018), so it evaluates the same days, except
    window 750: 750 changes are first available after 31 Dec 2018, so it starts at the first PLANNED
    re-estimation date with 750 changes behind it (the same dates as the planned design from then on,
    so only the window changes), and row 4 repeats the planned design on its days.
    'Without' rows drop evaluation days by date from the planned run (estimation unchanged).
    """
    main, blocks = run_hedges(clean, every)
    first_t = blocks["t"].iloc[0]
    short = [m for m in MATURITIES if m >= m4.PCA_FROM_CHECK]
    first_750 = blocks.loc[blocks["t"] >= clean.index[750 - 1], "t"].iloc[0]  # first planned t with 750 changes
    w750 = run_hedges(clean, every, window=750, first_t=first_750)
    specs = [  # (label, design, target, (resid, blocks))
        ("1 planned design (M4 headline)", "headline", m4.TARGET, (main, blocks)),
        ("2 window 250", "planned variant", m4.TARGET, run_hedges(clean, every, window=250, first_t=first_t)),
        ("3 window 750 (later start)", "planned variant", m4.TARGET, w750),
        ("4 planned design, same days as row 3", "comparator", m4.TARGET, (main[main.index >= w750[0].index[0]], blocks)),
        ("5 block 20 days", "planned variant", m4.TARGET, run_hedges(clean, every, step=20, first_t=first_t)),
        ("6 block 120 days", "planned variant", m4.TARGET, run_hedges(clean, every, step=120, first_t=first_t)),
    ]
    for k, (name, (first, last)) in enumerate(LEFT_OUT.items(), start=7):
        specs.append((f"{k} without the {name} evaluation days", "planned variant", m4.TARGET,
                      (main[(main.index < first) | (main.index > last)], blocks)))
    specs.append(("9 5y target: H1/H2 on 2y/10y, H3 on 2y/3y/10y", "planned variant", FIVE_YEAR,
                  run_hedges(clean, every, first_t=first_t, target=FIVE_YEAR, factor_hedges=FIVE_YEAR_HEDGES)))
    specs.append((f"10 PCA on 1-10y ({EX_POST})", "ex post", m4.TARGET,
                  run_hedges(clean[short], every[short], first_t=first_t)))
    table = pd.DataFrame([{**hedge_row(label, *run), "design": design, "target": target}
                          for label, design, target, run in specs])
    return table, {label: run for label, _, _, run in specs}


# 4. F2 bond-yield check

def h3_legs_cond(clean, blocks, legs):
    """Condition number, at each block's t, of the system H3 actually solves: the loadings of its
    hedge legs on PC1-3 from the PCA of that block's estimation window (the rows M4 uses, 'est first'
    to t). M4's 'cond (2/5/10y)' column is for 2y/5y/10y legs, so it does not describe an H3 on
    2y/3y/10y. Large = the three legs barely separate the three factors, so small estimation noise
    gives very large weights. Known at t (no look-ahead).
    Inputs: clean - the changes the run estimated on; blocks - its blocks frame; legs - H3 maturities.
    Output: Series aligned with 'blocks'. Raises ValueError if a window's row count differs from M4's.
    """
    out = []
    for _, r in blocks.iterrows():
        est = clean.loc[r["est first"]:r["t"]]
        if len(est) != r["n est"]:
            raise ValueError("h3_legs_cond: estimation window differs from M4's")
        _, load, _ = pca_loadings(est)
        out.append(float(np.linalg.cond(load.loc[legs, ["PC1", "PC2", "PC3"]].to_numpy(float))))
    return pd.Series(out, index=blocks.index)


def f2_check(f2, yields):
    """A 5-year position hedged with H1 (duration, 0.625 x 2y + 0.375 x 10y), H2 (PC1-2 neutral, 2y/10y)
    and H3 (PC1-3 neutral, 2y/3y/10y), PCA on the daily changes of the four maturities, M4's rolling
    design (window 500, step 60). Run on F2 bond yields and on F17 zero yields at the same maturities
    and dates (both with all four maturities, so each daily change covers the same interval in
    both). The suspect F17 days are dropped from both (data.daily_changes_bp works on any levels
    frame; the 4 changes are dropped from F2 too, so both runs use the same days).
    A third row (ex post, judgement) repeats the F2 run without the blocks whose largest |H3 weight|
    exceeds BIG_WEIGHT, to show what the rest of the sample says.
    Output: (table with one hedge_row per run, runs: name -> (resid, blocks)). Kept in memory:
    F2 residuals are a series derived from F2 and are never written.
    Raises ValueError if the two runs do not evaluate the same days.
    """
    a, b = f2[F2_MATURITIES].dropna(), yields[F2_MATURITIES].dropna()
    dates, runs = a.index.intersection(b.index), {}
    for name, levels in [("F2 bond yields", a.loc[dates]), ("F17 zero yields", b.loc[dates])]:
        clean = daily_changes_bp(levels, F2_MATURITIES, drop_suspect=True)
        every = daily_changes_bp(levels, F2_MATURITIES, drop_suspect=False)
        resid, blocks = run_hedges(clean, every, target=FIVE_YEAR, factor_hedges=FIVE_YEAR_HEDGES)
        cond = h3_legs_cond(clean, blocks, FIVE_YEAR_HEDGES["H3"][0])  # H3's own system (2y/3y/10y)
        runs[name] = (resid, blocks.assign(**{"cond (H3 legs)": cond}))
    (r2, b2), (r17, _) = runs.values()
    if not r2.index.equals(r17.index):
        raise ValueError("f2_check: F2 and F17 runs must evaluate the same days")
    big = b2.loc[h3_size(b2) > BIG_WEIGHT, "block"]
    runs[f"F2 without the {len(big)} blocks with an H3 weight above {BIG_WEIGHT:g} in size (ex post)"] = (r2[~r2["block"].isin(big)], b2)
    return pd.DataFrame([hedge_row(name, r, b) for name, (r, b) in runs.items()]), runs


def plot_hedge_sd(m4_table, f2_table, path):
    """hedge_sd_by_variant.png: residual sd (bp/day) of H1, H2 and H3 in every M4 variant and in the
    F2 check (one row each), so a reader sees at a glance where H3 drops below H1 and H2. H3 values
    beyond the axis are written at the right edge. Statistics only."""
    # Mark rows that are not the 7-year, 40-maturity setting, so the chart does not mix them silently.
    m4_rows = [r + (" [5y target]" if t == FIVE_YEAR else "") for r, t in zip(m4_table["rows"], m4_table["target"])]
    f2_rows = "F2 check: " + f2_table["rows"] + " [5y target, PCA on 2/3/5/10y]"
    table = pd.concat([m4_table.assign(rows=m4_rows), f2_table.assign(rows=f2_rows)], ignore_index=True)
    y, cap = np.arange(len(table))[::-1], 2.0
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(table) + 1.2), facecolor=COLORS["surface"])
    for h in ["H1", "H2", "H3"]:
        ax.scatter(table[f"sd {h}"].clip(upper=cap), y, s=46, color=HEDGE_COLORS[h], label=h, zorder=3,
                   edgecolors=COLORS["surface"], linewidths=1.5)
    for yy, v in zip(y, table["sd H3"]):
        if v > cap:
            ax.text(cap, yy + 0.25, f"H3 {v:.2f}", fontsize=7, color=COLORS["muted"], ha="right")
    ax.set_yticks(y, table["rows"], fontsize=8)
    ax.set_xlim(0, cap + 0.05)
    style_axes(ax, "", "")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_xlabel("residual sd (bp of DV01 per day)", color=COLORS["muted"])
    ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["ink"], loc="upper right", ncol=3,
              bbox_to_anchor=(1.0, -0.06))
    fig.suptitle("Out-of-sample residual sd by hedge and variant (lower = better)", fontsize=11,
                 color=COLORS["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Report

SHOW = ["rows", "n blocks", "n days", "sd H0", "sd H1", "sd H2", "sd H3", "t cl H3 vs H1", "p cl H3 vs H1",
        "t cl H3 vs H2", "p cl H3 vs H2", "t cl H2 vs H1", "p cl H2 vs H1", "H3 < H1 blocks", "max abs H3 w"]


def as_text(frame, digits, signed=(), pcols=()):
    """Copy of 'frame' for the markdown: floats rounded by m3_factors.to_text (digits, signed),
    p-value columns by m2.fmt_p ('<0.001' or 3 decimals). The CSVs keep the unrounded floats."""
    out = to_text(frame, digits, signed)
    for c in pcols:
        out[c] = [m2.fmt_p(p) for p in frame[c]]
    return out


def hedge_text(table):
    """The SHOW columns of a hedge table as text: sds and t (signed) to 2 decimals, p via fmt_p."""
    t_cols, p_cols = [c for c in SHOW if c.startswith("t cl")], [c for c in SHOW if c.startswith("p cl")]
    digits = {**{f"sd {h}": 2 for h in m4.HEDGES}, **{c: 2 for c in t_cols}, "max abs H3 w": 2}
    return as_text(table[SHOW], digits, t_cols, p_cols)


def short_name(label):
    """'row 9 (5y target)' from '9 5y target: H1/H2 ...' (row number, then the label to ':' or ' (').
    The 1-10y PCA row keeps its 'found after seeing the M4 results' label."""
    number, rest = label.split(" ", 1)
    name = rest.split(":")[0].split(" (")[0]
    return f"row {number} ({name}, {EX_POST})" if EX_POST in label else f"row {number} ({name})"


def join_words(items):
    """'a', 'a and b', 'a, b and c'; 'none' for an empty list (so a data refresh cannot crash it)."""
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def p_is(p):
    """'p = 0.012' or 'p < 0.001' (m2.fmt_p writes the small ones as '<0.001')."""
    s = m2.fmt_p(p)
    return f"p < {s[1:]}" if s.startswith("<") else f"p = {s}"


def span(row):
    """'26 Jun 2019 - 18 Sep 2019' from a row with 'eval first' and 'eval last'."""
    return f"{row['eval first']:%d %b %Y} - {row['eval last']:%d %b %Y}"


def write_report(ctx, path):
    """Write outputs/m5/robustness_report.md from the tables in ctx (built in main()).

    Every number in the text is read from the tables. Judgement, written for the 20 Oct 2026 data
    (re-read after a refresh): the wording of the answer, the explanations (the smooth F17 short end
    at 1 month; an ill-conditioned H3 system behind each H3 failure) and BIG_WEIGHT. 'checks' is
    evaluated before any text is built and stops the script if a qualitative claim in the text no
    longer matches the tables.
    """
    OIS, F17, NC = OIS_FORECASTS.values()
    a, d = ctx["acc"].set_index(["h", "forecast"]), ctx["dm"].set_index(["h", "test"])
    ea, ed = ctx["ext_acc"].set_index(["h", "forecast"]), ctx["ext_dm"].set_index("h")
    e, ee = ctx["ep"].set_index(["h", "episode"]), ctx["ext_ep"]
    m2t, m4t, f2t = ctx["m2"], ctx["m4"], ctx["f2"]
    common = ctx["common"][OIS_HORIZONS[0]]
    n_common, first, last, ois_last = len(common), common["t"].min(), common["t"].max(), ctx["ois_last"]

    # 1. OIS benchmark: p-values of the paired tests and the accuracy numbers the text quotes.
    pair_p = {}
    for x, y in ((OIS, F17), (OIS, NC), (F17, NC)):
        pair_p[(x, y)] = {h: d.loc[(h, f"{x} vs {y}"), "p"] for h in OIS_HORIZONS}
    p_of, p_on, p_fn = pair_p[(OIS, F17)], pair_p[(OIS, NC)], pair_p[(F17, NC)]
    rm = {h: {f: a.loc[(h, f), "RMSE (bp)"] for f in (OIS, F17, NC)} for h in OIS_HORIZONS}
    one = {f: a.loc[(1, f)] for f in (OIS, F17)}            # 1-month accuracy rows
    tight1, n_tight = e.loc[(1, "Tightening")], int(e.loc[(1, "Tightening"), "n"])
    bias_range = {}
    for f in (OIS, F17):
        biases = [a.loc[(h, f), "bias (bp)"] for h in OIS_HORIZONS]
        bias_range[f] = (min(biases), max(biases), min(a.loc[(h, f), "bias p"] for h in OIS_HORIZONS))
    ext_p = ", ".join(m2.fmt_p(ed.loc[h, "p"]) for h in OIS_HORIZONS)
    ext_bias = "; ".join(f"{ea.loc[(h, F17), 'bias (bp)']:+.1f} bp at h = {h} (p = {m2.fmt_p(ea.loc[(h, F17), 'bias p'])})"
                         for h in OIS_HORIZONS)
    ext_last = ctx["panels"][OIS_HORIZONS[0]]["t"].max()
    top = ee[ee["h"] == 3].nlargest(2, f"bias {F17}")          # episodes with the largest 3-month F17 bias
    starts = {name: pd.Timestamp(lo) for name, lo, _ in m2.EPISODES}

    # 2. M2 rows.
    below5 = m2t[m2t["DM p"] < 0.05]
    y22 = m2t[m2t["variant"].str.startswith("2022")].set_index("h")
    b3 = y22.loc[3]

    # 3. M4 rows. 'better' = H3 beats H1 at 5% (block-clustered); comparators are not variants.
    head, seven = m4t.iloc[0], m4t["target"] == m4.TARGET
    better = (m4t["p cl H3 vs H1"] < 0.05) & (m4t["t cl H3 vs H1"] < 0)
    planned = m4t["design"] == "planned variant"
    comparator = m4t[m4t["design"] == "comparator"].iloc[0]
    losing = m4t.loc[planned & ~better, "rows"].tolist()
    wb = ctx["worst"].set_index("rows")
    head_block = wb.loc[head["rows"]]
    # A losing run "keeps M4's block" if its worst block overlaps M4's worst block in time.
    same_block = [r for r in losing if wb.loc[r, "eval first"] <= head_block["eval last"]
                  and wb.loc[r, "eval last"] >= head_block["eval first"]]
    own_block = [r for r in losing if r not in same_block]
    flips = join_words(short_name(x) for x in m4t.loc[planned & better, "rows"])
    share = m4t["H3 < H1 blocks"] / m4t["n blocks"]
    h2_p7, five = m4t.loc[seven, "p cl H2 vs H1"].max(), m4t.loc[~seven].iloc[0]
    ex_post_row = m4t[m4t["design"] == "ex post"].iloc[0]
    pool = seven & (m4t["design"] != "comparator")           # ex-post weight pattern: 7-year runs only
    max_win = m4t.loc[pool & better, "max abs H3 w"].max()
    min_lose = m4t.loc[pool & ~better, "max abs H3 w"].min()

    # 4. F2 rows and the blocks where H3 blew up.
    f2r = f2t.set_index("rows")
    n_f2, n_f17, n_f2x = f2t["rows"]
    b2 = ctx["f2_runs"][n_f2][1]
    ycc_first, ycc_last = YCC_REGIMES[0]["first_close"], YCC_REGIMES[-1]["last_close"]
    ycc_share = []  # share of each block's estimation window (business days) inside yield-curve control
    for _, r in b2.iterrows():
        days = pd.bdate_range(r["est first"], r["t"])
        ycc_share.append(float(((days >= ycc_first) & (days <= ycc_last)).mean()))
    b2 = b2.assign(**{"ycc share": ycc_share})
    is_big = h3_size(b2) > BIG_WEIGHT
    big, rest = b2[is_big], b2[~is_big]
    ycc_heavy = rest[rest["ycc share"] > 0.5]
    big_span = f"{big['eval first'].min():%d %b %Y} - {big['eval last'].max():%d %b %Y}" if len(big) else "none"

    checks = {  # qualitative claim in the text -> still true? (evaluated before any text is built)
        "OIS and F17 zero cannot be told apart": min(p_of.values()) >= 0.05,
        "1 month: OIS better than F17 zero on RMSE, MAE and |bias|": rm[1][OIS] < rm[1][F17]
            and one[OIS]["MAE (bp)"] < one[F17]["MAE (bp)"] and abs(one[OIS]["bias (bp)"]) < abs(one[F17]["bias (bp)"]),
        "both beat no change on RMSE": all(rm[h][NC] > max(rm[h][OIS], rm[h][F17]) for h in OIS_HORIZONS),
        "1-month Tightening: F17 zero low, OIS not": tight1[f"bias {F17}"] > abs(tight1[f"bias {OIS}"]) + 5,
        "top 3-month F17 biases come after the Dec 2023 input change": all(starts[n] >= pd.Timestamp("2023-12-29")
                                                                         for n in top["episode"]),
        "losing 7-year runs have larger H3 weights than winning ones (ex post)": max_win < min_lose,
        "planned design on row 3's days: H3 beats H1": comparator["p cl H3 vs H1"] < 0.05
            and comparator["t cl H3 vs H1"] < 0,
        "H2 beats H1 at 7y, not at 5y or on F2": h2_p7 < 0.05 and five["p cl H2 vs H1"] >= 0.05
            and not (f2r.loc[n_f2, "p cl H2 vs H1"] < 0.05 and f2r.loc[n_f2, "t cl H2 vs H1"] < 0),
        "F2: no factor hedge significantly better than H1": f2r.loc[n_f2, "p cl H3 vs H1"] >= 0.05
            and f2r.loc[n_f2, "p cl H2 vs H1"] >= 0.05,
        "F2: H3 worse than no hedge": f2r.loc[n_f2, "sd H3"] > f2r.loc[n_f2, "sd H0"],
        "F2: H3's own system ill-conditioned only in the big blocks": len(big) > 0
            and big["cond (H3 legs)"].min() > rest["cond (H3 legs)"].max(),
        "F2: the big blocks' windows end inside yield-curve control": len(big) > 0
            and bool(((big["t"] >= ycc_first) & (big["t"] <= ycc_last)).all()),
        "F17 at four maturities: H3 not better than H1": f2r.loc[n_f17, "p cl H3 vs H1"] >= 0.05,
        "forward beats no change in every M2 row": bool((m2t["fwd RMSE"] < m2t["no-chg RMSE"]).all()),
        "DM p < 5% only at 1 month and in M2's 'excluding Tightening' at h = 3":
            [h for h in OIS_HORIZONS if ed.loc[h, "p"] < 0.05] == [1]
            and list(zip(below5["variant"], below5["h"])) == [(f"excluding Tightening ({EX_POST_M2})", 3)],
    }
    if not all(checks.values()):
        raise ValueError(f"report text no longer matches the tables: {[k for k, ok in checks.items() if not ok]}")

    answer = "\n".join([
        f"- **Forecasting the target (M2).** The F17 forward has a lower RMSE than no change in all {len(m2t)} M2 "
        f"variants, but the Diebold-Mariano (DM) p-value is below 5% only in 'excluding Tightening', a row "
        f"{EX_POST_M2}. From 2022 on the 3-month DM p is {m2.fmt_p(b3['DM p'])}.",
        f"- **OIS (F1).** Over the {n_common} month-ends where F17 and F1's OIS overlap ({first:%b %Y} - {last:%b %Y}; "
        f"F1's OIS ends {ois_last:%d %b %Y}), OIS and the F17 zero yield cannot be told apart (p = "
        f"{join_words(m2.fmt_p(p_of[h]) for h in OIS_HORIZONS)} at 1, 3 and 6 months). That is expected rather than "
        "reassuring: the RBA fits the 0-1 year part of its curve to OIS (RDP 2008-09; M1 section 9), so this does not "
        "show that bond prices alone carry the same information. At 1 month OIS is more accurate on every measure "
        f"(RMSE {rm[1][OIS]:.1f} vs {rm[1][F17]:.1f} bp, MAE {one[OIS]['MAE (bp)']:.1f} vs {one[F17]['MAE (bp)']:.1f}, "
        f"bias {one[OIS]['bias (bp)']:+.1f} vs {one[F17]['bias (bp)']:+.1f}).",
        f"- **F17 vs no change, to {ext_last:%b %Y}.** The F17 zero yield beats no change at 1, 3 and 6 months, "
        f"significantly only at 1 month (DM p = {ext_p}), and sits below the realised average cash rate (section 1c).",
        f"- **Hedging (M4).** H3 still fails to beat duration (H1) in {len(losing)} of {int(planned.sum())} planned "
        f"variants, but each failure rests on one ill-conditioned block: {join_words(short_name(r) for r in same_block)} "
        f"{'keeps' if len(same_block) == 1 else 'keep'} M4's {span(head_block)} trouble, and "
        f"{join_words(short_name(r) for r in own_block)} {'fails' if len(own_block) == 1 else 'fail'} on a block of "
        f"{'its' if len(own_block) == 1 else 'their'} own. It flips in {flips}, as it does in "
        f"{short_name(ex_post_row['rows'])}; on row 3's days the planned design itself has H3 beating H1 (row 4, t = "
        f"{comparator['t cl H3 vs H1']:+.2f}, {p_is(comparator['p cl H3 vs H1'])}). Level and slope (H2) beat duration in every 7-year run (block-"
        f"clustered p <= {m2.fmt_p(h2_p7)}), but not for the 5-year target (p = {m2.fmt_p(five['p cl H2 vs H1'])}).",
        f"- **Bond yields (F2), 5-year bond.** No factor hedge is significantly better than duration (H3 vs H1 p = "
        f"{m2.fmt_p(f2r.loc[n_f2, 'p cl H3 vs H1'])}, H2 vs H1 p = {m2.fmt_p(f2r.loc[n_f2, 'p cl H2 vs H1'])}). H3 did worse "
        f"than no hedge ({f2r.loc[n_f2, 'sd H3']:.2f} vs {f2r.loc[n_f2, 'sd H0']:.2f} bp/day) because of {len(big)} blocks "
        f"where the system H3 solves was ill-conditioned at t (cond {join_words(f'{c:.1f}' for c in big['cond (H3 legs)'])}, "
        f"against {rest['cond (H3 legs)'].max():.1f} or less in the other blocks). With four maturities H3 shows no edge "
        f"on F17 either ({f2r.loc[n_f17, 'sd H3']:.2f} vs H1 {f2r.loc[n_f17, 'sd H1']:.2f}), so this check cannot test "
        "M4's 40-maturity result.",
    ])
    acc_cols = {"bias (bp)": 1, "MAE (bp)": 1, "RMSE (bp)": 1}
    dm_digits = {"mean d (bp^2)": 0, "t": 2}
    ep_txt = as_text(ctx["ep"], {c: 1 for c in ctx["ep"].columns if c.startswith(("bias", "RMSE"))})
    m2_txt = as_text(m2t, {c: 1 for c in ["fwd bias", "fwd RMSE", "gap-adj RMSE", "no-chg RMSE"]} | {"DM t": 2, "Fama b": 2},
                     ["DM t"], ["DM p", "HLN p", "p b=1"])
    worst = as_text(ctx["worst"], {"cond (2/5/10y)": 1, "cond (H3 legs)": 1, "max abs H3 w": 2,
                                   **{f"block sd {h}": 2 for h in ("H1", "H2", "H3")}})
    worst = worst.assign(**{c: ctx["worst"][c].dt.strftime("%d %b %Y") for c in ("eval first", "eval last")})
    winners = m4t.loc[better & (m4t["design"] != "comparator"), "rows"]
    if len(ycc_heavy):
        ycc_note = (f"{len(ycc_heavy)} other blocks also had more than half their window inside YCC but stayed well "
                    f"conditioned (largest H3 weight {h3_size(ycc_heavy).max():.2f}), so YCC alone does not explain it.")
    else:
        ycc_note = "No other block had more than half its window inside YCC."

    parts = [
        "# M5 - robustness report",
        f"Generated by `src/m5_robustness.py` on {pd.Timestamp.today():%d %b %Y}. The planned designs of M2 and M4 stay "
        "the headline; everything here is robustness. The M4 PCA on 1-10y was found after seeing the M4 results and is "
        "labelled that way. F1 (OIS: FENICS data) and F2 (licence unchecked) stay on disk: this folder holds only "
        "statistics and charts of statistics, no series.",
        "## Answer: which results survive", answer,
        "## 1. OIS benchmark (the headline addition)",
        f"Forecast dates: the last F17 business day of each month (as M2). Forecasts of the average actual cash rate over "
        f"the next h months: the h-month OIS on t (on the average-rate basis), the F17 zero yield at h/12 years and no "
        f"change (F1's cash rate on t). Outcome: the average of F1's cash rate over the calendar days in (t, t + h months]. "
        f"Error = outcome - forecast (bp). F1's OIS columns end on {ois_last:%d %b %Y} (F1 gives no reason), so sections "
        f"1a-1b use the {n_common} month-ends where F17 and OIS overlap, {first:%b %Y} - {last:%b %Y}: Steady, Easing to "
        f"the floor, Floor and YCC and the first {n_tight} Tightening forecasts. 'bias p' tests mean error = 0 and DM "
        "uses d = e_model^2 - e_bench^2 (negative = model better), both Newey-West with lag h; 'HLN p' is the "
        "small-sample version (M2). At h = 1 HLN uses no lags, so there it is the less cautious p.",
        "### 1a. Over the whole OIS sample",
        md_table(as_text(ctx["acc"], acc_cols, pcols=["bias p"])),
        md_table(as_text(ctx["dm"], dm_digits, ["t"], ["p", "HLN p"])),
        f"OIS and the F17 zero yield cannot be told apart (p = {join_words(m2.fmt_p(p_of[h]) for h in OIS_HORIZONS)}). "
        "The RBA fits the 0-1 year part of its curve to OIS (RDP 2008-09, Appendix A; not checked for F17 after its "
        "Dec 2023 input change), so close agreement is built in. At 1 month OIS is more accurate on RMSE, MAE and bias. "
        f"Both beat no change at every horizon, significantly only at 1 month (p = {m2.fmt_p(p_on[1])} for OIS, "
        f"{m2.fmt_p(p_fn[1])} for F17 zero). OIS has almost no bias ({bias_range[OIS][0]:+.1f} to "
        f"{bias_range[OIS][1]:+.1f} bp); the F17 zero yield is {bias_range[F17][0]:+.1f} to {bias_range[F17][1]:+.1f} bp "
        f"(lowest p {m2.fmt_p(bias_range[F17][2])}).",
        "### 1b. By episode", md_table(ep_txt), "![RMSE by episode](ois_rmse_by_episode.png)",
        f"The forecasts differ most in Tightening at 1 month (F17 zero bias {tight1[f'bias {F17}']:+.1f} bp, OIS "
        f"{tight1[f'bias {OIS}']:+.1f}, {n_tight} forecasts). F17 publishes no 1-month point, and its fitted curve is smooth "
        "and pinned at the overnight rate on t, so neither the curve nor an interpolation of it shows a hike priced for "
        "the first days of the month. In Steady the target never moved, so no change is perfect at 1 month.",
        "### 1c. F17 zero yield vs no change, to the last full window inside F1",
        md_table(as_text(ctx["ext_acc"], acc_cols, pcols=["bias p"])),
        md_table(as_text(ctx["ext_dm"], dm_digits, ["t"], ["p", "HLN p"])),
        f"Without OIS the sample runs to {ext_last:%b %Y} (h = 1). The F17 zero yield still beats no change and sits "
        f"below the realised average: {ext_bias}. Most of that bias comes from "
        f"{join_words(f'{n} ({b:+.1f} bp at 3 months)' for n, b in zip(top['episode'], top[f'bias {F17}']))}, both after "
        "F17's Dec 2023 input change; Newey-West lag h is a floor and the bias test has no small-sample version, so read "
        "these p-values as rough. By episode: `f17_vs_no_change_episodes.csv`.",
        "## 2. M2 checks",
        "Forward vs no change for the target in effect (M2's outcome); bias and RMSE in bp. Rows built with M2's own "
        "functions; '2022 onwards' keeps forecasts made from Jan 2022.", md_table(m2_txt),
        f"The forward's RMSE is below no change in {int((m2t['fwd RMSE'] < m2t['no-chg RMSE']).sum())} of {len(m2t)} rows; "
        f"DM p is below 5% only for {', '.join(f'{v} at h = {h}' for v, h in zip(below5['variant'], below5['h']))}. "
        f"From 2022 on the 3-month DM p is {m2.fmt_p(b3['DM p'])} and the Fama slope b = {b3['Fama b']:.2f} (p for b = 1: "
        f"{m2.fmt_p(b3['p b=1'])}); the 6-month forward bias is {y22.loc[6, 'fwd bias']:+.1f} bp.",
        "## 3. M4 checks",
        "H0-H3 rerun with `m4.rolling_hedges`, one change at a time; suspect days out of estimation and evaluation as "
        "in M4. t and p are block-clustered (negative t = the first hedge better); 'H3 < H1 blocks' counts blocks where "
        "H3's mean squared residual is lower; 'max abs H3 w' is the largest H3 weight in any block. Row 4 is a "
        "comparator (the planned design on row 3's days), not a variant.",
        md_table(hedge_text(m4t)), "![Residual sd by variant](hedge_sd_by_variant.png)",
        f"Row 1 reproduces M4 (sd {head['sd H0']:.2f} / {head['sd H1']:.2f} / {head['sd H2']:.2f} / {head['sd H3']:.2f}). "
        f"H3 beats H1 at 5% in {join_words(short_name(x) for x in winners)}, and in the comparator row 4. In every "
        f"section-3 run H3 has the smaller mean squared residual in most blocks ({share.min():.0%}-{share.max():.0%}). "
        f"Pattern found after seeing the results (7-year runs only): every losing run has a block with an H3 weight of "
        f"{min_lose:.2f} or more in size, and no winning run exceeds {max_win:.2f}. It describes these runs; it is not "
        "a rule. The worst H3 block of each run:", md_table(worst),
        "## 4. F2 bond-yield check",
        "A 5-year position hedged by duration (H1, 0.625 x 2y + 0.375 x 10y), PC1-2 on 2y/10y (H2) and PC1-3 on "
        "2y/3y/10y (H3), PCA on the daily changes of the four yields, M4's rolling design, on F2 bond yields and on F17 "
        "zero yields at the same maturities and days.", md_table(hedge_text(f2t)),
        f"H3's weights exploded in {len(big)} F2 blocks ({big_span}; largest |w| {h3_size(big).max():.1f}). At t the "
        "system H3 solves (its 2y/3y/10y loadings on PC1-3, 'cond (H3 legs)') was ill-conditioned there (cond "
        f"{join_words(f'{c:.1f}' for c in big['cond (H3 legs)'])}) and well conditioned in the other {len(rest)} blocks "
        f"({rest['cond (H3 legs)'].max():.1f} or less). Both windows ended inside yield-curve control (YCC: the RBA "
        f"targeted a 3-year bond yield, so the 3-year leg barely moved). {ycc_note} Without the big blocks (ex post) H3 "
        f"leaves {f2r.loc[n_f2x, 'sd H3']:.2f} vs {f2r.loc[n_f2x, 'sd H1']:.2f} (H1), p = "
        f"{m2.fmt_p(f2r.loc[n_f2x, 'p cl H3 vs H1'])}. Over all blocks H3 has the smaller mean squared residual in "
        f"{int(f2r.loc[n_f2, 'H3 < H1 blocks'])} of {int(f2r.loc[n_f2, 'n blocks'])} F2 blocks, against "
        f"{int(f2r.loc[n_f17, 'H3 < H1 blocks'])} on F17 at the same maturities. On F17 at the four maturities H3 leaves "
        f"{f2r.loc[n_f17, 'sd H3']:.2f}, but {five['sd H3']:.2f} with the PCA on all 40 maturities (section 3, row 9, same "
        "days): four points pin PC3 poorly. F17 is rounded to 2 decimals and F2 to 3, so rounding noise is larger in "
        "the F17 run.",
        "## Limitations",
        f"- **OIS sample.** F1's OIS ends {ois_last:%d %b %Y}: {n_common} overlapping month-ends (about "
        f"{n_common // 6} independent 6-month windows), {n_tight} Tightening forecasts and none in Plateau, Easing or "
        "Re-tightening. Not rejecting 'OIS = F17' with so few observations is not evidence that they are equal.\n"
        "- **OIS and F17 are not independent.** F17's 0-1 year segment is fitted to OIS (RDP 2008-09), so section 1 "
        "mostly tests that fit, not what bond prices alone know.\n"
        "- **Timing and basis.** F1 does not say when in the day OIS is sampled; F17 is the 4:30 pm close. The OIS "
        "conversion assumes daily compounding on actual/365 over (t, t + h months]. Small conventions (when OIS starts "
        "accruing; whether the night of t is in F17's window) move these RMSEs by tenths of a bp, the size of the "
        "OIS-F17 gaps at 3 and 6 months. The 1-month F17 point is interpolated.\n"
        f"- **Ex-post choices.** The 1-10y PCA ({EX_POST}), M2's 'excluding Tightening' row ({EX_POST_M2}), the F2 split at "
        f"an H3 weight of {BIG_WEIGHT:g} and the weight pattern in section 3 were chosen after seeing results. Many "
        "variants, no multiple-testing adjustment: read each p-value as one row's.\n"
        "- **F2 is not a zero curve.** Interpolated coupon-bond yields, DV01 weights set by maturity (not duration), "
        "a 3-year point held near the YCC target in 2020-2021; 3 decimals, so less rounding noise than F17.",
        "Files: this report, `ois_rmse_by_episode.png`, `hedge_sd_by_variant.png` and one CSV per table (unrounded).",
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# Main

def main():
    """Run the four parts, write the CSVs (statistics only), the two charts and the report, and print
    a short summary."""
    # F1 and F2 are gitignored, so a fresh clone does not have them: stop with a clear message.
    for name in (data.F1_FILE, data.F2_FILE):
        if not (data.RAW_DIR / name).exists():
            sys.exit(f"M5 needs data/raw/{name} (gitignored): re-download it from the URL in data/MANIFEST.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forwards, yields, changes, f1, f2 = load_f17("forwards"), load_f17("yields"), load_a2(), load_f1(), load_f2()

    # 1. OIS benchmark: month-end F17 dates, as in M2.
    month_ends = m2.month_end_dates(yields.index)
    panels = {h: ois_panel(f1, yields, month_ends, h) for h in OIS_HORIZONS}
    common = {h: p[p["ois"].notna()] for h, p in panels.items()}
    acc, dm, ep = forecast_tables(common, list(OIS_FORECASTS), OIS_PAIRS)
    ext_acc, ext_dm, ext_ep = forecast_tables(panels, ["f17_zero", "no_change"], [("f17_zero", "no_change")])
    adj = pd.concat([(p["ois_quote"] - p["ois"]) * m2.BP for p in common.values()])  # quote - average basis
    # No change = F1's cash rate on t: check it equals F17's 0-year point (M1: y0 = the actual cash rate).
    both = yields.index.intersection(f1.index)
    y0_gap = float((yields.loc[both, 0.0] - f1.loc[both, "cash_rate"]).abs().max() * m2.BP)

    # 2-4.
    m2_table = m2_robustness(forwards, yields, changes)
    clean = daily_changes_bp(yields, MATURITIES, drop_suspect=True)
    every = daily_changes_bp(yields, MATURITIES, drop_suspect=False)
    m4_table, m4_runs = m4_robustness(clean, every)
    f2_table, f2_runs = f2_check(f2, yields)
    worst = worst_block_table({**m4_runs, **{f"F2 check: {k}": v for k, v in f2_runs.items()}})

    ois_last = f1[OIS_COLUMN[OIS_HORIZONS[0]]].last_valid_index()   # last F1 date with an OIS quote
    ctx = {"ois_last": ois_last, "panels": panels, "common": common, "acc": acc, "dm": dm, "ep": ep, "ext_acc": ext_acc,
           "ext_dm": ext_dm, "ext_ep": ext_ep, "adj": adj, "y0_gap": y0_gap, "y0_n": len(both), "m2": m2_table,
           "m4": m4_table, "m4_runs": m4_runs, "f2": f2_table, "f2_runs": f2_runs, "worst": worst}
    tables = {"ois_accuracy": acc, "ois_dm_tests": dm, "ois_episodes": ep, "f17_vs_no_change_accuracy": ext_acc,
              "f17_vs_no_change_dm_tests": ext_dm, "f17_vs_no_change_episodes": ext_ep,
              "m2_robustness": m2_table, "m4_robustness": m4_table, "f2_check": f2_table, "worst_h3_blocks": worst}
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
    period = f"{common[1]['t'].min():%b %Y} - {common[1]['t'].max():%b %Y}, where F17 and OIS overlap"
    plot_ois_episodes(ep, OUT_DIR / "ois_rmse_by_episode.png", period)
    plot_hedge_sd(m4_table, f2_table, OUT_DIR / "hedge_sd_by_variant.png")
    write_report(ctx, REPORT_PATH)

    print(f"M5: OIS sample {len(common[1])} month-ends; RMSE (bp):")
    print(acc.pivot(index="h", columns="forecast", values="RMSE (bp)").round(1).to_string())
    for table in (m4_table, f2_table):
        print(table[["rows", "sd H1", "sd H2", "sd H3", "t cl H3 vs H1", "p cl H3 vs H1"]].round(3).to_string(index=False))
    print(f"Report: {REPORT_PATH}")
    return ctx


if __name__ == "__main__":
    main()
