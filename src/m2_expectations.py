"""M2 - expectations test: does the F17 forward rate predict the cash rate target?

Run from the project folder: python src/m2_expectations.py
Forecasts made at the F17 close on t: 'forward', 'gap_adj'
and 'no_change' (see build_panel), judged against target_in_effect(t + h months). Writes outputs/m2/
(report, two PNGs, forecasts.csv). Rates in per cent per annum; errors in bp, error = outcome - forecast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no screen needed
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Make "import data" work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402  (import after the sys.path line on purpose)
    PROJECT_DIR, SUSPECT_DAYS, load_f17, load_a2, target_in_effect, target_known_at_close,
)
from m1_validate import COLORS, md_table, style_axes  # noqa: E402  (chart colours and table helper)


# Settings

OUT_DIR = PROJECT_DIR / "outputs" / "m2"
REPORT_PATH = OUT_DIR / "expectations_report.md"

BP = 100.0                    # per cent -> basis points
MAIN_HORIZONS = [3, 6]        # months
ALL_HORIZONS = [3, 6, 12]     # 12 months is the robustness horizon
LAST_OUTCOME_DATE = pd.Timestamp("2026-09-30")  # keep forecasts whose outcome date t + h is on or before this
MID_MONTH_DAY = 15            # mid-month variant: last F17 day on or before the 15th
N_MISSES = 10                 # largest forward misses listed per horizon

# Episodes by forecast month (the boundaries are judgement).
EPISODES = [
    ("Steady", "2017-01", "2019-04"),
    ("Easing to the floor", "2019-05", "2020-11"),
    ("Floor and YCC", "2020-12", "2022-03"),
    ("Tightening", "2022-04", "2023-11"),
    ("Plateau", "2023-12", "2024-12"),
    ("Easing", "2025-01", "2025-08"),
    ("Re-tightening", "2025-09", "2026-09"),
]

# The three forecasts: panel column -> (long label, short label for wide tables).
FORECASTS = {"forward": ("forward", "fwd"), "gap_adj": ("gap-adjusted forward", "gap-adj"),
             "no_change": ("no change", "no-chg")}


# Dates

def month_end_dates(index):
    """Last F17 business day of every calendar month.

    Input: index - the F17 DatetimeIndex. Output: DatetimeIndex named 't', one date per month.
    Convention: "business day" = a date with an F17 row, so a month-end holiday falls back.
    """
    index = pd.DatetimeIndex(index)
    last = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    return pd.DatetimeIndex(last.to_numpy(), name="t")


def mid_month_dates(index, skip=()):
    """Mid-month forecast dates: the last F17 day on or before the 15th of each month.

    Inputs: index - the F17 DatetimeIndex; skip - dates never to use (main passes the M1 suspect
            days: a level read on a probable fitting artefact is not a clean forecast).
    Output: DatetimeIndex named 't', one date per month. A skipped day falls back to the previous
            F17 day of the same month (13 Dec 2019 -> 12 Dec 2019).
    """
    index = pd.DatetimeIndex(index)
    usable = index[(index.day <= MID_MONTH_DAY) & ~index.isin(pd.DatetimeIndex(list(skip)))]
    last = pd.Series(usable, index=usable).groupby(usable.to_period("M")).max()
    return pd.DatetimeIndex(last.to_numpy(), name="t")


def outcome_dates(dates, h, rule="months"):
    """The calendar day whose target each forecast is judged against.

    Inputs: dates - forecast dates t; h - horizon in months; rule - 'months' (main): t + h calendar
            months via pandas DateOffset, so 30 Nov + 3 months = 28 (or 29) Feb; 'days' (robustness):
            t + round(365.25 h / 12) = 91, 183 or 365 days, the night the h/12-year forward is for.
    Output: DatetimeIndex of outcome dates, same order as 'dates'.
    """
    dates = pd.DatetimeIndex(dates)
    if rule == "months":
        return dates + pd.DateOffset(months=h)
    if rule == "days":
        return dates + pd.Timedelta(days=int(round(365.25 * h / 12)))
    raise ValueError(f"rule must be 'months' or 'days', got {rule!r}")


def episode_of(dates):
    """Episode name for each forecast date, from EPISODES (by calendar month).

    Input: dates. Output: numpy array of names. Raises ValueError if a date is in no episode
    (EPISODES must cover Jan 2017 - Sep 2026 without gaps; test_episodes_cover_sample checks it).
    """
    months = pd.DatetimeIndex(dates).to_period("M")
    names = np.full(len(months), None, dtype=object)
    for name, first, last in EPISODES:
        names[(months >= pd.Period(first, "M")) & (months <= pd.Period(last, "M"))] = name
    if any(n is None for n in names):
        raise ValueError("a forecast date falls outside every episode")
    return names


# The forecast panel

def build_panel(forwards, yields, changes, dates, h, rule="months"):
    """One row per forecast date t: the three forecasts, the outcome, the errors.

    Inputs: forwards, yields - load_f17('forwards'), load_f17('yields'); changes - load_a2(); dates -
            forecast dates t (F17 dates, not announcement days); h - months (h/12 on the grid); rule -
            outcome-date rule (outcome_dates).
    Output: DataFrame: t, h, episode, forward (F17 instantaneous forward at h/12 years, M1 check 9),
            y0 (F17 0-year yield on t), known (target_known_at_close(t)), gap_bp (known - y0, bp),
            gap_adj (forward + known - y0), no_change (= known), outcome_date, outcome, and err_forward,
            err_gap_adj, err_no_change (outcome - forecast, bp). Kept: outcome date <= 30 Sep 2026.
    No look-ahead: forecasts use only the F17 row dated t and A2 decisions announced on or before t;
    known = target_in_effect(t + 1 day) includes a decision announced on t (2.30 pm release, 4:30 pm
    close; data.py). Only the outcome looks forward.
    Raises ValueError if a t is an announcement day (known is the new target but y0 still the old cash
    rate, so known - y0 would add the decision to a forward that already prices it), or if an outcome
    inside the sample or a forecast input is missing.
    """
    t = pd.DatetimeIndex(dates)
    maturity = h / 12
    if maturity not in forwards.columns:
        raise ValueError(f"h = {h} months is not on the F17 grid")
    if t.isin(changes["announcement_date"]).any():
        raise ValueError("a forecast date is an announcement day; the gap adjustment would count it twice")
    panel = pd.DataFrame({"t": t, "h": h, "episode": episode_of(t)})
    panel["forward"] = forwards.loc[t, maturity].to_numpy()
    panel["y0"] = yields.loc[t, 0.0].to_numpy()
    panel["known"] = target_known_at_close(t, changes).to_numpy()
    panel["gap_bp"] = (panel["known"] - panel["y0"]) * BP
    panel["gap_adj"] = panel["forward"] + panel["known"] - panel["y0"]
    panel["no_change"] = panel["known"]
    panel["outcome_date"] = outcome_dates(t, h, rule)
    panel["outcome"] = target_in_effect(panel["outcome_date"], changes).to_numpy()
    for name in FORECASTS:
        panel[f"err_{name}"] = (panel["outcome"] - panel[name]) * BP

    panel = panel[panel["outcome_date"] <= LAST_OUTCOME_DATE].reset_index(drop=True)
    if panel["outcome"].isna().any():
        raise ValueError("missing outcome inside the sample (A2 should cover every outcome date)")
    if panel[["forward", "y0", "known"]].isna().any().any():
        raise ValueError("a kept forecast row has a missing input")
    return panel


# Statistics

def error_stats(errors):
    """Input: errors in bp. Output: dict n, bias (mean error), mae, rmse, all in bp."""
    e = np.asarray(errors, dtype=float)
    return {"n": len(e), "bias": e.mean(), "mae": np.abs(e).mean(), "rmse": np.sqrt(np.mean(e ** 2))}


def hac_ols(y, X, h):
    """OLS of y on X with Newey-West (HAC) standard errors.

    Inputs: y - 1-D array; X - 2-D array (include the constant yourself); h - number of lags.
    Output: statsmodels results object (params, bse, tvalues, pvalues).
    Conventions (spelled out in the call; tests/test_m2.py checks them by hand): Bartlett weights
    1 - j / (h + 1) for lags j = 1..h, no small-sample scaling, p-values from t(n - k).
    Why lag h: an h-month forecast made every month overlaps the next h - 1, so errors
    are at least MA(h - 1). Lag h is a floor, not a margin: Bartlett gives lag h - 1 only weight
    2 / (h + 1) (0.5 at h = 3), and clustered rate changes can keep errors correlated beyond h - 1,
    so these p-values are on the low side (diebold_mariano also reports the HLN version).
    """
    cov = {"maxlags": h, "kernel": "bartlett", "use_correction": False}
    return sm.OLS(np.asarray(y, dtype=float), np.asarray(X, dtype=float)).fit(cov_type="HAC", cov_kwds=cov, use_t=True)


def diebold_mariano(e_model, e_bench, h):
    """Diebold-Mariano test on squared errors: Newey-West and Harvey-Leybourne-Newbold versions.

    Inputs: e_model, e_bench - errors of two forecasts on the same dates (bp); h - horizon (months).
    Output: dict n, mean_d (bp^2), se, stat, p (Newey-West) and stat_hln, p_hln (HLN).
    Sign: d = e_model^2 - e_bench^2; a NEGATIVE mean_d and stat mean the model is better.
    Newey-West: d regressed on a constant with hac_ols (lag h), so stat = mean_d / NW se.
    HLN (Harvey, Leybourne and Newbold 1997, the usual small-sample DM for h-step forecasts):
    lrv = gamma_0 + 2 (gamma_1 + ... + gamma_(h-1)) with equal weights, gamma_j = sum u_t u_(t-j) / n
    and u = d - mean_d; stat = mean_d / sqrt(lrv / n) * sqrt((n + 1 - 2h + h(h - 1) / n) / n), p from
    t(n - 1). NaN if lrv is not positive (equal weights do not guarantee it).
    """
    d = np.asarray(e_model, dtype=float) ** 2 - np.asarray(e_bench, dtype=float) ** 2
    res = hac_ols(d, np.ones((len(d), 1)), h)
    n, u = len(d), d - d.mean()
    lrv = u @ u / n + 2 * sum(u[j:] @ u[:n - j] / n for j in range(1, h))
    factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat_hln = d.mean() / np.sqrt(lrv / n) * factor if lrv > 0 else np.nan
    return {"n": n, "mean_d": res.params[0], "se": res.bse[0], "stat": res.tvalues[0], "p": res.pvalues[0],
            "stat_hln": stat_hln, "p_hln": 2 * stats.t.sf(abs(stat_hln), n - 1)}


def fama_regression(outcome, known, forecast, h):
    """Fama regression (outcome - known) = a + b (forecast - known) + u.

    Inputs: outcome, known, forecast - per cent, same dates; h - Newey-West maxlags (tests_tables also tries 2h).
    Output: dict n, a (bp), se_a, p_a0, b, se_b, p_b1 (b = 1), p_b0 (b = 0), r2. Both sides in bp.
    Reading: a = 0 and b = 1 is an unbiased forecast with no term premium; b > 0 means the priced
    change carries information. Standard errors from hac_ols; p_b1 uses t(n - 2), like statsmodels.
    """
    y = (np.asarray(outcome, dtype=float) - np.asarray(known, dtype=float)) * BP
    x = (np.asarray(forecast, dtype=float) - np.asarray(known, dtype=float)) * BP
    res = hac_ols(y, np.column_stack([np.ones(len(x)), x]), h)
    a, b = res.params
    se_a, se_b = res.bse
    p_b1 = 2 * stats.t.sf(abs((b - 1) / se_b), res.df_resid)
    return {"n": len(y), "a": a, "se_a": se_a, "p_a0": res.pvalues[0], "b": b, "se_b": se_b,
            "p_b1": p_b1, "p_b0": res.pvalues[1], "r2": res.rsquared}


# Tables

def full_sample_table(panels):
    """Accuracy of each forecast over the full sample.

    Input:  panels - dict h -> panel from build_panel.
    Output: DataFrame, one row per horizon and forecast: n, bias, bias p, MAE, RMSE (bp). 'bias p'
            tests mean error = 0: errors on a constant with hac_ols (lag h), as overlap correlates them.
    """
    rows = []
    for h, panel in panels.items():
        for name, (label, _) in FORECASTS.items():
            e = panel[f"err_{name}"].to_numpy()
            s = error_stats(e)
            bias_p = hac_ols(e, np.ones((len(e), 1)), h).pvalues[0]
            rows.append({"h (months)": h, "forecast": label, "n": s["n"], "bias (bp)": s["bias"],
                         "bias p": bias_p, "MAE (bp)": s["mae"], "RMSE (bp)": s["rmse"]})
    return pd.DataFrame(rows)


def episode_table(panel):
    """Accuracy of each forecast by episode.

    Input:  panel - one horizon's panel from build_panel.
    Output: DataFrame, one row per episode with forecasts: episode, n, then '<short> bias',
            '<short> MAE', '<short> RMSE' (bp) for short = fwd, gap-adj, no-chg.
    """
    rows = []
    for name, _, _ in EPISODES:
        part = panel[panel["episode"] == name]
        if part.empty:
            continue
        row = {"episode": name, "n": len(part)}
        for col, (_, short) in FORECASTS.items():
            s = error_stats(part[f"err_{col}"])
            row.update({f"{short} bias": s["bias"], f"{short} MAE": s["mae"], f"{short} RMSE": s["rmse"]})
        rows.append(row)
    return pd.DataFrame(rows)


def window_context(t, outcome_date, changes):
    """Describe the target changes a forecast made on t had to anticipate.

    Inputs: t, outcome_date - Timestamps; changes - load_a2().
    Output: text such as '5 hikes, +225 bp (04 May 2022 - 07 Sep 2022)' or 'no change'. Counts
    changes effective after t + 1 day (later than anything known at the close on t) and on or
    before the outcome date, so their sum is outcome - known.
    """
    eff = changes["effective_date"]
    sel = changes[(eff > t + pd.Timedelta(days=1)) & (eff <= outcome_date) & (changes["change_pp"] != 0)]
    if sel.empty:
        return "no change"
    moves = sel["change_pp"]
    kind = "hike" if (moves > 0).all() else "cut" if (moves < 0).all() else "move"
    span = f"{sel['effective_date'].iloc[0]:%d %b %Y}"
    if len(sel) > 1:
        span += f" - {sel['effective_date'].iloc[-1]:%d %b %Y}"
    return f"{len(sel)} {kind}{'s' if len(sel) > 1 else ''}, {moves.sum() * BP:+.0f} bp ({span})"


def largest_misses(panel, changes, k=N_MISSES):
    """The forecast dates with the largest absolute forward error.

    Inputs: panel - one horizon's panel from build_panel; changes - load_a2(); k - rows wanted.
    Output: DataFrame, largest |err_forward| first: dates, episode, known / forward / outcome (%), priced
    change = forward - known and error (whole bp: 2-decimal rates), 'what happened' (window_context).
    Ties: |error| is rounded to 6 decimals (binary noise) and every row tied with the k-th is kept, so
    there can be more than k rows; among ties the earlier forecast date comes first.
    """
    order = panel.assign(abs_err=panel["err_forward"].abs().round(6))
    order = order.sort_values("abs_err", ascending=False, kind="stable")
    top = order[order["abs_err"] >= order["abs_err"].iloc[k - 1]]
    return pd.DataFrame({
        "forecast date": top["t"].dt.strftime("%Y-%m-%d"),
        "target date": top["outcome_date"].dt.strftime("%Y-%m-%d"),
        "episode": top["episode"],
        "known (%)": top["known"], "forward (%)": top["forward"], "outcome (%)": top["outcome"],
        "priced change (bp)": ((top["forward"] - top["known"]) * BP).round().astype(int),
        "error (bp)": top["err_forward"].round().astype(int),
        "what happened": [window_context(r.t, r.outcome_date, changes) for r in top.itertuples()],
    })


def tests_tables(panels):
    """Diebold-Mariano and Fama results for each horizon.

    Input:  panels - dict h -> panel from build_panel.
    Output: (dm, fama) DataFrames, one row per horizon and forecast (forward, gap-adjusted
            forward), each against no change. dm: n, mean d and NW se (bp^2, whole numbers), t, p,
            HLN t, HLN p. fama: n, a (bp), se a, p a=0, b, se b, p b=1, p b=1 with lag 2h, p b=0, R2.
    """
    dm_rows, fama_rows = [], []
    for h, p in panels.items():
        for name in ("forward", "gap_adj"):
            label = FORECASTS[name][0]
            dm = diebold_mariano(p[f"err_{name}"], p["err_no_change"], h)
            dm_rows.append({"h": h, "model vs no change": label, "n": dm["n"],
                            "mean d (bp^2)": int(round(dm["mean_d"])), "NW se": int(round(dm["se"])),
                            "t": dm["stat"], "p": dm["p"], "HLN t": dm["stat_hln"], "HLN p": dm["p_hln"]})
            f = fama_regression(p["outcome"], p["known"], p[name], h)
            p_b1_2h = fama_regression(p["outcome"], p["known"], p[name], 2 * h)["p_b1"]
            fama_rows.append({"h": h, "forecast": label, "n": f["n"], "a (bp)": f["a"], "se a": f["se_a"],
                              "p a=0": f["p_a0"], "b": f["b"], "se b": f["se_b"], "p b=1": f["p_b1"],
                              "p b=1 (lag 2h)": p_b1_2h, "p b=0": f["p_b0"], "R2": f["r2"]})
    return pd.DataFrame(dm_rows), pd.DataFrame(fama_rows)


def robustness_row(variant, panel, h):
    """One line of the robustness table.

    Inputs: variant - label; panel - a panel from build_panel (possibly with rows removed); h - lag.
    Output: dict: variant, h, n, forward bias and the three RMSEs (bp), then forward vs no change:
            DM t and p (Newey-West), HLN p, Fama b and p for b = 1.
    """
    dm = diebold_mariano(panel["err_forward"], panel["err_no_change"], h)
    f = fama_regression(panel["outcome"], panel["known"], panel["forward"], h)
    rmse = {name: error_stats(panel[f"err_{name}"])["rmse"] for name in FORECASTS}
    return {"variant": variant, "h": h, "n": len(panel), "fwd bias": panel["err_forward"].mean(),
            "fwd RMSE": rmse["forward"], "gap-adj RMSE": rmse["gap_adj"], "no-chg RMSE": rmse["no_change"],
            "DM t": dm["stat"], "DM p": dm["p"], "HLN p": dm["p_hln"], "Fama b": f["b"], "p b=1": f["p_b1"]}


# Charts

def plot_path(panel, daily, path):
    """Realised target against the 6-month forward, each forecast drawn at its target date.

    Inputs: panel - main panel for h = 6; daily - target_in_effect on every calendar day; path -
    PNG file. A dot and the black line at the same x are the forecast and outcome for that day.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(daily.index, daily.to_numpy(), drawstyle="steps-post", color=COLORS["ink"], lw=1.6,
            label="cash rate target in effect (A2)")
    ax.scatter(panel["outcome_date"], panel["forward"], s=26, color=COLORS["blue"], zorder=3,
               label="6-month forward from t, plotted at t + 6 months")
    ax.scatter(panel["outcome_date"], panel["gap_adj"], s=26, facecolors="none", edgecolors=COLORS["orange"],
               linewidths=1.2, zorder=3, label="gap-adjusted forward")
    style_axes(ax, "Cash rate target vs what the curve priced 6 months earlier", "per cent")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(panels, path):
    """Forward errors (bars) and no-change errors (line) by forecast date, h = 3 and 6.

    Inputs: panels - dict h -> panel (uses MAIN_HORIZONS); path - PNG file. Episodes are shaded
    bands. The no-change error equals the actual change in the target over the horizon, so the
    gap between bar and line is the change the curve priced.
    """
    fig, axes = plt.subplots(len(MAIN_HORIZONS), 1, figsize=(10, 6.8), sharex=True)
    for ax, h in zip(axes, MAIN_HORIZONS):
        p = panels[h]
        for i, (name, first, last) in enumerate(EPISODES):
            start, end = pd.Period(first, "M").start_time, pd.Period(last, "M").end_time
            if i % 2 == 0:
                ax.axvspan(start, end, color=COLORS["grid"], alpha=0.6, lw=0)
            ax.text(start + (end - start) / 2, 1.0, name, ha="center", va="bottom", fontsize=7,
                    color=COLORS["muted"], transform=ax.get_xaxis_transform())
        ax.bar(p["t"], p["err_forward"], width=20, color=COLORS["blue"], label="forward error")
        ax.plot(p["t"], p["err_no_change"], color=COLORS["orange"], lw=1.4,
                label="no-change error (= actual change)")
        ax.axhline(0, color=COLORS["muted"], lw=0.8)
        style_axes(ax, "", "bp")
        ax.set_title(f"h = {h} months: error = outcome - forecast, by forecast date", loc="left",
                     fontsize=10, color=COLORS["ink"], pad=16)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Report

def fmt_p(p):
    """p-value as text: 3 decimals, or '<0.001'."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def with_text(frame, cols, fmt=fmt_p):
    """Copy of frame with the columns 'cols' turned into text by fmt (default: p-values)."""
    return frame.assign(**{c: frame[c].map(fmt) for c in cols})


def write_report(ctx):
    """Write outputs/m2/expectations_report.md from the results in ctx (built in main).

    ctx keys: panels, episodes, misses (dicts h -> frame); setup, full, dm, fama, rule_cases,
    robust (frames); kept_dm_t (number); tight_share, kurt (dicts h -> number).
    Numbers in the text are read from those tables. The interpretive clauses (which hikes a miss
    came before, which episodes drive the bias, what is or is not significant) were written by
    hand; 'checks' stops the script if a data refresh makes one untrue. Re-read after a refresh.
    """
    panels, rob, misses, share = ctx["panels"], ctx["robust"], ctx["misses"], ctx["tight_share"]
    full = ctx["full"].set_index(["h (months)", "forecast"])
    fwd = {h: full.loc[(h, "forward")] for h in ALL_HORIZONS}
    gap = {h: full.loc[(h, "gap-adjusted forward")] for h in ALL_HORIZONS}
    nochg = {h: full.loc[(h, "no change")] for h in ALL_HORIZONS}
    dm = ctx["dm"].set_index(["h", "model vs no change"])
    fama = ctx["fama"].set_index(["h", "forecast"])
    d3, d6 = dm.loc[(3, "forward")], dm.loc[(6, "forward")]
    f3, f6 = fama.loc[(3, "forward")], fama.loc[(6, "forward")]
    variant = rob.set_index(["variant", "h"])  # robustness rows, looked up by (variant, h)
    no_tight3 = variant.loc[("excluding Tightening", 3)]
    other3 = rob.loc[(rob["h"] == 3) & (rob["variant"] != "excluding Tightening"), "DM t"]
    n6, gap6, top3 = len(panels[6]), panels[6]["gap_bp"], misses[6].head(3)
    late25 = misses[6][misses[6]["forecast date"].between("2025-08", "2025-12-31")]
    bias_drop = [fwd[h]["bias (bp)"] - gap[h]["bias (bp)"] for h in ALL_HORIZONS]
    sig_bias = [k for k, r in full.iterrows() if k[1] != "no change" and r["bias p"] < 0.05]
    ep_bias6 = ctx["episodes"][6].set_index("episode")["fwd bias"]
    steady, top_bias6 = ep_bias6["Steady"], ep_bias6.nlargest(3)
    below5 = rob[rob["DM p"] < 0.05]
    nc_wins = {h: ep.loc[ep["no-chg RMSE"] < ep["fwd RMSE"], "episode"].tolist() for h, ep in ctx["episodes"].items()}

    checks = {  # hand-written claim -> still true?
        "forward beats no change": all(fwd[h][c] < nochg[h][c] for h in ALL_HORIZONS for c in ("MAE (bp)", "RMSE (bp)")),
        "full-sample DM not significant": min(d3["p"], d6["p"]) >= 0.05,
        "only raw 3-month bias significant": sig_bias == [(3, "forward")],
        "top 3 six-month misses from 2022, Aug-Dec 2025 misses positive": top3["forecast date"].str.startswith(
            "2022").all() and len(late25) > 0 and (late25["error (bp)"] > 0).all(),
        "Steady bias negative": steady < 0,
        "bias episodes": set(top_bias6.index) == {"Floor and YCC", "Re-tightening", "Easing"},
    }
    if not all(checks.values()):
        raise ValueError(f"report text no longer matches the tables: {[k for k, ok in checks.items() if not ok]}")

    lines = [
        "# M2 - Expectations test", "",
        f"Generated by `src/m2_expectations.py`. Forecast dates t: last F17 business day of each month from "
        f"{panels[3]['t'].min():%b %Y}. Error = outcome - forecast, in bp; positive = target ended above the forecast.",
        "", "## Answer", "", "Partly.", "",
        f"- **Accuracy.** The F17 forward has a {1 - fwd[3]['RMSE (bp)'] / nochg[3]['RMSE (bp)']:.0%} lower RMSE than "
        f"\"no change\" at 3 months ({fwd[3]['RMSE (bp)']:.0f} vs {nochg[3]['RMSE (bp)']:.0f} bp) and "
        f"{1 - fwd[6]['RMSE (bp)'] / nochg[6]['RMSE (bp)']:.0%} lower at 6 ({fwd[6]['RMSE (bp)']:.0f} vs "
        f"{nochg[6]['RMSE (bp)']:.0f} bp).",
        f"- **Significance.** Not at 5% over the full sample: Diebold-Mariano p = {fmt_p(d3['p'])} (Newey-West, lag h) "
        f"to {fmt_p(d3['HLN p'])} (Harvey-Leybourne-Newbold) at 3 months, {fmt_p(d6['p'])} to {fmt_p(d6['HLN p'])} at "
        f"6. Tightening forecasts supply {share[3]:.0%} of the summed squared-error gain at 3 months; without them "
        f"the 3-month p is {fmt_p(no_tight3['DM p'])} (HLN {fmt_p(no_tight3['HLN p'])}).",
        f"- **Fama slope.** The priced change carries information (p for b = 0: {fmt_p(f3['p b=0'])} at 3 months, "
        f"{fmt_p(f6['p b=0'])} at 6). At 6 months b = {f6['b']:.2f} (p for b = 1: {fmt_p(f6['p b=1'])}); at 3 months "
        f"b = {f3['b']:.2f} is borderline below 1 (p = {fmt_p(f3['p b=1'])} with lag h, "
        f"{fmt_p(f3['p b=1 (lag 2h)'])} with lag 2h).",
        f"- **Bias.** Mean errors are positive ({fwd[3]['bias (bp)']:+.1f} bp at 3 months, {fwd[6]['bias (bp)']:+.1f} "
        f"at 6), the opposite sign to a term premium, but only the raw 3-month bias differs from zero (p = "
        f"{fmt_p(fwd[3]['bias p'])}; {fmt_p(fwd[6]['bias p'])} and {fmt_p(fwd[12]['bias p'])} at 6 and 12 months). "
        f"The gap adjustment cuts it to {gap[3]['bias (bp)']:+.1f} bp (p = {fmt_p(gap[3]['bias p'])}): "
        f"{bias_drop[0] / fwd[3]['bias (bp)']:.0%} of it is the actual-vs-target cash rate gap. The sample cannot sign "
        "a term premium.",
        f"- **Largest misses** are turning points priced late: the three largest 6-month misses "
        f"({', '.join(f'{e:+d}' for e in top3['error (bp)'])} bp) were forecasts made on "
        f"{', '.join(top3['forecast date'])}, before the 2022 hikes; {len(late25)} more of the top {len(misses[6])} "
        "are Aug-Dec 2025 forecasts that missed the 2026 hikes.",
        "", f"With about {n6 // 6} independent 6-month observations and 26 decisions in four clusters, treat every "
        "p-value here as rough.",
        "", "## Setup", "", md_table(ctx["setup"], 0), "",
        "Gap-adjusted forward = forward + (target_known_at_close(t) - F17 0-year yield on t). Outcome = "
        "target_in_effect(t + h calendar months); t is kept only if that date is on or before 30 Sep 2026.",
        "", "## Accuracy over the full sample", "", md_table(with_text(ctx["full"], ["bias p"]), 1), "",
        "The forward beats no change on MAE and RMSE at every horizon. 'bias p' tests mean error = 0 (errors on a "
        "constant, Newey-West, lag h); for the two forwards it is below 5% only for the raw forward at 3 months. The "
        f"gap adjustment lowers the bias by {min(bias_drop):.1f}-{max(bias_drop):.1f} bp but hardly moves RMSE: the "
        f"gap (mean {gap6.mean():.1f} bp on the 6-month dates) is small next to the misses.",
        "", "## Accuracy by episode", "",
        "Episodes by forecast date (boundaries are judgement). fwd = forward, gap-adj = gap-adjusted "
        "forward, no-chg = no change; bias, MAE, RMSE in bp.", "",
    ]
    for h in MAIN_HORIZONS:
        lines += [f"**h = {h} months** (episodes where no change has the lower RMSE: "
                  f"{', '.join(nc_wins[h]) or 'none'})", "", md_table(ctx["episodes"][h], 1), ""]
    lines += [
        f"In Steady the 6-month error is negative ({steady:.1f} bp): 2017-2018 forwards sat a few bp above an unchanged "
        "target (term premium, or hikes priced that never came) and early-2019 forecasts missed the mid-2019 cuts. "
        f"The largest positive episode biases are {', '.join(f'{e} ({b:+.1f})' for e, b in top_bias6.items())}: "
        "cuts priced in 2025 followed by hikes in 2026, and Floor and YCC forecasts made before the 2022 hikes.",
        "", "## Tests", "",
        "Diebold-Mariano on d = e_model^2 - e_no_change^2 (negative = model better): 'p' Newey-West (Bartlett, lag h), "
        "'HLN' Harvey-Leybourne-Newbold (equal weights on lags 1 to h - 1, small-sample factor, t(n - 1)). Fama: "
        "(outcome - known) = a + b (forecast - known) + u in bp, lag h; 'p b=1 (lag 2h)' doubles the lags.", "",
        md_table(with_text(ctx["dm"], ["p", "HLN p"]), 2), "",
        md_table(with_text(ctx["fama"], ["p a=0", "p b=1", "p b=1 (lag 2h)", "p b=0"]), 2), "",
        f"**Weaknesses.** {n6} overlapping six-month forecasts hold roughly {n6 // 6} independent observations. Bartlett "
        "weights at lag h under-weight the overlap (lag h - 1 gets 2 / (h + 1)) and Newey-West is biased down in small "
        f"samples, so 'p' is on the low side; HLN is the cautious reading. d is heavy-tailed (excess kurtosis "
        f"{ctx['kurt'][3]:.0f} at 3 months, {ctx['kurt'][6]:.0f} at 6) and concentrated in Tightening, so the t "
        "approximation is rough. Dropping an episode leaves a gap that Newey-West treats as adjacent months.",
        "", "## Largest forward misses", "",
        "'priced change' = forward - known; 'what happened' = target changes effective after t + 1 day and on or "
        "before the target date (their sum is outcome - known). Rows tied with the 10th are kept.", "",
    ]
    for h in MAIN_HORIZONS:
        lines += [f"**h = {h} months**", "", md_table(misses[h], 2), ""]
    bp_cols = ["fwd bias", "fwd RMSE", "gap-adj RMSE", "no-chg RMSE"]
    lines += [
        "## Robustness", "",
        "Forward errors in bp; DM and Fama compare the forward with no change.", "",
        md_table(with_text(with_text(rob, ["DM p", "HLN p", "p b=1"]), bp_cols, "{:.1f}".format), 2), "",
        f"The forward RMSE is below no change in {(rob['fwd RMSE'] < rob['no-chg RMSE']).sum()} of {len(rob)} rows. "
        f"DM p is below 5% only for {', '.join(f'{v} at h = {h}' for v, h in zip(below5['variant'], below5['h'])) or 'no row'}. "
        f"Tightening forecasts supply {share[3]:.0%} / {share[6]:.0%} / {share[12]:.0%} of the summed d at 3 / 6 / 12 "
        f"months and no other variant drops them, so the 3-month DM t barely moves elsewhere ({other3.min():.3f} to "
        f"{other3.max():.3f}; the repeats are genuine).",
        "", "## Charts", "",
        "![Target vs 6-month forward path](target_vs_forward_6m.png)", "",
        "![Forecast errors by episode](errors_by_episode.png)", "",
        "Every forecast row (all horizons, main variant) is in `outputs/m2/forecasts.csv`.",
        "", "## Limitations", "",
        "- **Revised data, not real time.** F17 is the RBA's fitted curve as published in Oct 2026 and can be revised "
        "(its inputs changed on 29 Dec 2023). A trader on day t saw something else.",
        "- **Fitted short end.** The forwards come from a smooth fitted curve anchored at the actual cash rate, not "
        "traded OIS (M5 section 1 compares the two). The gap adjustment fixes the level gap on t only.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Main

def main():
    """Build every panel, run the tests, write the report, charts and CSV to outputs/m2/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forwards, yields, changes = load_f17("forwards"), load_f17("yields"), load_a2()
    month_ends = month_end_dates(forwards.index)
    mid = mid_month_dates(forwards.index, skip=pd.DatetimeIndex(sorted(SUSPECT_DAYS)))

    # build_panel stops with an error if any of these dates is an announcement day.
    panels = {h: build_panel(forwards, yields, changes, month_ends, h) for h in ALL_HORIZONS}
    by_days = {h: build_panel(forwards, yields, changes, month_ends, h, rule="days") for h in ALL_HORIZONS}
    by_mid = {h: build_panel(forwards, yields, changes, mid, h) for h in ALL_HORIZONS}
    # Mid-month dates with the suspect day 13 Dec 2019 kept, to show its effect on the 3-month DM t.
    kept3 = build_panel(forwards, yields, changes, mid_month_dates(forwards.index), 3)

    setup = pd.DataFrame([{"h (months)": h, "F17 maturity read (years)": f"{h / 12:.2f}", "n": len(p),
                           "first t": p["t"].min(), "last t": p["t"].max(), "last outcome date": p["outcome_date"].max()}
                          for h, p in panels.items()])
    # Forecast dates where the two outcome-date rules give different targets.
    both = pd.concat(panels.values()).merge(pd.concat(by_days.values()), on=["h", "t"], suffixes=("_m", "_d"))
    rule_cases = both.loc[both["outcome_m"] != both["outcome_d"],
                          ["h", "t", "outcome_date_m", "outcome_m", "outcome_date_d", "outcome_d"]]

    dm_table, fama_table = tests_tables(panels)
    robust = []
    for h in ALL_HORIZONS:
        variants = {"main (month-end, t + h months)": panels[h], "outcome date t + days": by_days[h],
                    "mid-month forecast dates": by_mid[h]}
        variants.update({f"excluding {e}": panels[h][panels[h]["episode"] != e] for e in ("Floor and YCC", "Tightening")})
        robust += [robustness_row(name, p, h) for name, p in variants.items()]

    # Loss differential d (forward vs no change): share of its sum from Tightening, and its excess kurtosis.
    d = {h: p["err_forward"] ** 2 - p["err_no_change"] ** 2 for h, p in panels.items()}
    ctx = {
        "panels": panels, "setup": setup, "full": full_sample_table(panels),
        "episodes": {h: episode_table(panels[h]) for h in MAIN_HORIZONS},
        "dm": dm_table, "fama": fama_table,
        "misses": {h: largest_misses(panels[h], changes) for h in MAIN_HORIZONS},
        "rule_cases": rule_cases, "robust": pd.DataFrame(robust),
        "kept_dm_t": diebold_mariano(kept3["err_forward"], kept3["err_no_change"], 3)["stat"],
        "tight_share": {h: d[h][panels[h]["episode"] == "Tightening"].sum() / d[h].sum() for h in d},
        "kurt": {h: stats.kurtosis(d[h]) for h in d},
    }

    daily = target_in_effect(pd.date_range(month_ends.min(), LAST_OUTCOME_DATE, freq="D"), changes)
    plot_path(panels[6], daily, OUT_DIR / "target_vs_forward_6m.png")
    plot_errors(panels, OUT_DIR / "errors_by_episode.png")
    csv = pd.concat(panels.values(), ignore_index=True)
    csv = csv.assign(t=csv["t"].dt.strftime("%Y-%m-%d"), outcome_date=csv["outcome_date"].dt.strftime("%Y-%m-%d"))
    csv.to_csv(OUT_DIR / "forecasts.csv", index=False, float_format="%.4f")
    write_report(ctx)
    print(f"M2 expectations: report written to {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(ctx["full"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
