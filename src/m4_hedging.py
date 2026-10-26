"""M4 - hedging test: does a 3-factor hedge beat a duration hedge out of sample? the README, "M4".

Position: long a 7-year F17 zero, 1 unit of DV01, hedged with 2, 5 and 10-year zeros. Daily residual
in bp of DV01: r = dy7 - sum_i w_i dy_i (dy = daily changes in bp from data.daily_changes_bp; w_i = DV01
of instrument i per unit of the position's DV01; the long position's P&L is -r x DV01). H0 unhedged;
H1 duration (2y/10y by maturity interpolation); H2 2y/10y with zero PC1-2 exposure; H3 2y/5y/10y with
zero PC1-3 exposure (PCA = m3_factors.pca_loadings, covariance, 0.25-10y). Out of sample: PCA on the
trailing 500 changes up to and including t (suspect days dropped), weights fixed for the next 60 F17
changes, then step 60. Robustness: suspect days kept; PCA on 1-10y only; an ex-post fallback H3 -> H2.
Writes outputs/m4/ (report, PNGs, CSVs).
Run from the project folder:  python src/m4_hedging.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no screen needed
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Make "import data" work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import PROJECT_DIR, load_f17, daily_changes_bp  # noqa: E402
from m1_validate import md_table, style_axes, COLORS  # noqa: E402  (same table and chart style as M1)
from m2_expectations import EPISODES, episode_of, fmt_p  # noqa: E402  (the README episodes, one definition)
from m3_factors import (  # noqa: E402  (M3's PCA and its PC3 diagnostics, reused as they are)
    MATURITIES, ROUNDING_VAR_BP2, pca_loadings, best_match, separation_table, to_text,
)


# Settings

OUT_DIR = PROJECT_DIR / "outputs" / "m4"
REPORT_PATH = OUT_DIR / "hedging_report.md"

TARGET = 7.0                      # the position: a 7-year zero, 1 unit of DV01
H1_LEGS = (2.0, 10.0)             # duration hedge legs (the README M4)
FACTOR_HEDGES = {                 # hedge -> (instruments, PCs whose exposure is set to zero)
    "H2": ([2.0, 10.0], ["PC1", "PC2"]),
    "H3": ([2.0, 5.0, 10.0], ["PC1", "PC2", "PC3"]),
}
HEDGES = ["H0", "H1", "H2", "H3"]
HEDGE_TEXT = {"H0": "unhedged", "H1": "duration: 0.375 x 2y + 0.625 x 10y",
              "H2": "PC1-2 neutral, 2y/10y", "H3": "PC1-3 neutral, 2y/5y/10y"}
WINDOW = 500                      # estimation window, in F17 changes (business days)
STEP = 60                         # weights held for the next 60 changes, then re-estimated
PAIRS = [("H3", "H1"), ("H3", "H2")]   # paired tests: model vs benchmark
COND_FLAG = 10                    # cond(2/5/10y) above this: block flagged, fallback rows (judgement, ex post)
THRESHOLDS = [5, 8, 10, 20, 50]   # other fallback thresholds tried in section 5 (all ex post)
SHORT_END = 1.0                   # 'PC3 share <= 1y' = sum of PC3's squared loadings at maturities <= 1y
PC3_SHOWN = [2.0, 5.0, 7.0, 10.0]  # PC3 loadings logged per block: the hedge legs and the target
PCA_FROM_CHECK = 1.0              # robustness rows 6-7: PCA on maturities >= 1y only (chosen ex post)
H3_YLIM = (-0.5, 1.2)             # H3 weight panel range; weights outside it are written on the chart
# Weight columns of the blocks table, one per hedge leg ('H3 w5' = H3's DV01 weight on the 5y).
WEIGHT_COLS = ["H1 w2", "H1 w10"] + [f"{h} w{m:g}" for h, (inst, _) in FACTOR_HEDGES.items() for m in inst]
BENCH_COLORS = {"H1": COLORS["blue"], "H2": COLORS["orange"]}     # cumulative chart: one line per benchmark
LEG_COLORS = {2.0: COLORS["blue"], 5.0: COLORS["aqua"], 10.0: COLORS["orange"]}


# Hedge maths

def h1_weights(target=TARGET, legs=H1_LEGS):
    """Duration (DV01-neutral) hedge weights by linear interpolation in maturity.

    Inputs: target - the position's maturity (years); legs - (short, long) hedge maturities.
    Output: Series indexed by the two legs: w_long = (target - short) / (long - short), w_short =
            1 - w_long. For 7y with 2y/10y: 0.375 and 0.625. Nothing is estimated.
    Convention: weights are in DV01 units, so w_short + w_long = 1 makes the hedge DV01-neutral (a
    parallel 1 bp move gives r = 0). Interpolation also gives w_short x short + w_long x long =
    target, so any change that is linear in maturity is hedged exactly.
    """
    short, long = legs
    w_long = (target - short) / (long - short)
    return pd.Series([1.0 - w_long, w_long], index=[short, long], name="weight")


def factor_weights(loadings, instruments, pcs, target=TARGET):
    """Hedge weights that set the position's exposure to the chosen PCs to zero.

    Inputs: loadings - DataFrame from pca_loadings (index maturity, columns PC1, PC2, ...);
            instruments - hedge maturities, as many as pcs; pcs - e.g. ['PC1', 'PC2'];
            target - the position's maturity.
    Output: Series of weights (DV01 units) indexed by instrument maturity.
    Maths: with dy_m = sum_k L(m, k) f_k + e_m, the residual r = dy_target - sum_i w_i dy_i has
    exposure L(target, k) - sum_i w_i L(i, k) to factor k. Setting all len(pcs) exposures to zero
    is the square system A w = b with A[k, i] = L(instrument i, PC k) and b[k] = L(target, PC k),
    solved by numpy.linalg.solve (LinAlgError if A is singular). A PC's sign does not matter:
    flipping PC k flips row k of A and b[k] together, so w is unchanged.
    """
    a = loadings.loc[list(instruments), pcs].to_numpy(float).T
    b = loadings.loc[target, pcs].to_numpy(float)
    return pd.Series(np.linalg.solve(a, b), index=list(instruments), name="weight")


def factor_exposure(loadings, weights, pcs, target=TARGET):
    """Exposure of the residual to each PC: L(target, k) - sum_i w_i L(i, k). Inputs as factor_weights,
    with 'weights' a Series indexed by maturity (empty = unhedged). Output: Series indexed by pcs; zero
    (to round-off) for the PCs a factor hedge was built on.
    """
    exposure = loadings.loc[target, pcs].astype(float)
    for m, w in weights.items():
        exposure = exposure - w * loadings.loc[m, pcs]
    return exposure


def residuals(changes, weights, target=TARGET):
    """Residual r = dy_target - sum_i w_i dy_i in bp of DV01, one value per day. Inputs: changes - daily
    changes (bp), columns = maturities; weights - Series indexed by instrument maturity (empty = unhedged,
    r = dy_target). Output: Series indexed like changes (its sign does not affect any sd or square).
    """
    r = changes[target].astype(float)
    for m, w in weights.items():
        r = r - w * changes[m]
    return r


def hedge_weights(loadings, target=TARGET, factor_hedges=FACTOR_HEDGES):
    """Weights of H0-H3 for one window: dict hedge -> Series indexed by maturity. H0 empty, H1 =
    h1_weights(target) (fixed, legs H1_LEGS), H2 / H3 = factor_weights on 'loadings' with the legs and
    PCs in factor_hedges. The defaults are the planned M4 design; M5 passes a 5-year target and other
    legs (e.g. H3 on 2y/3y/10y) for its robustness runs.
    """
    out = {"H0": pd.Series(dtype=float), "H1": h1_weights(target)}
    for name, (instruments, pcs) in factor_hedges.items():
        out[name] = factor_weights(loadings, instruments, pcs, target)
    return out


# Rolling out-of-sample test

def block_schedule(est_index, eval_index, window=WINDOW, step=STEP, first_t=None):
    """Estimation windows and evaluation blocks of the rolling test.

    Inputs: est_index - dates of the changes the PCA may use (main: suspect days dropped); eval_index -
            dates of every change a residual is recorded for (keeps the suspect days, so they can be
            reported both ways); window, step - counts of changes; first_t - first estimation date
            (default: the window-th date of est_index, the first day with a full window).
    Output: list of dicts, one per block: 't' (last date of the estimation window), 'est' (the last
            'window' dates of est_index on or before t), 'eval' (the next 'step' dates of eval_index
            after t; the last block may be shorter). The next t is this block's last eval date, so
            blocks cover every date after the first t once. No look-ahead: est <= t < every eval date.
    Raises ValueError if an estimation window has fewer than 'window' dates.
    """
    est_index, eval_index = pd.DatetimeIndex(est_index), pd.DatetimeIndex(eval_index)
    t = est_index[window - 1] if first_t is None else pd.Timestamp(first_t)
    blocks = []
    while (eval_index > t).any():
        est = est_index[est_index <= t][-window:]
        if len(est) < window:
            raise ValueError(f"block_schedule: only {len(est)} estimation dates up to {t:%Y-%m-%d}")
        ev = eval_index[eval_index > t][:step]
        blocks.append({"t": t, "est": est, "eval": ev})
        t = ev[-1]
    return blocks


def rolling_hedges(est_changes, eval_changes, reference=None, window=WINDOW, step=STEP, first_t=None,
                   target=TARGET, factor_hedges=FACTOR_HEDGES):
    """Run the out-of-sample hedge test.

    Inputs: est_changes - changes (bp) the PCA is estimated on (main: suspect days dropped; its
            columns set the PCA's maturity range); eval_changes - changes residuals are recorded on
            (main: all F17 changes); reference - loadings each window's PC3 is named against (main():
            the full-sample PCA; a label only, it uses later data and never touches the weights);
            window, step, first_t - block_schedule; target, factor_hedges - the position and the
            H2 / H3 legs (hedge_weights; defaults = the planned design, a 7-year target).
    Output: (resid, blocks)
      resid  - DataFrame indexed by evaluation date: 'block', 't' (its estimation date), 'dy7' (the
               target's change; named 'dy5' for a 5-year target), and
               one residual column per hedge H0-H3 (bp of DV01).
      blocks - one row per block: dates and counts; the weights ('H1 w2', ..., 'H3 w10') and 'H3
               total' (H3's DV01 per unit of the position's); 'PC3/PC4 eigenvalue ratio' and 'cond
               (2/5/10y)' (m3_factors.separation_table); 'PC3 share <= 1y' (sum of PC3's squared
               loadings at maturities <= SHORT_END: near 1 = a money-market factor; known at t); 'PC3
               at 2y' ... 'PC3 at 10y' (PC3_SHOWN, NaN if a maturity is not in the PCA; M3's sign
               rule, positive at 5y); 'max |H3 exposure|'
               (in-sample, ~1e-16 if the solve is right); 'PC3 best match' / 'its cos' (reference PC1-5
               closest to the window's PC3 over all its maturities, so the short end weighs most).
    Per block: pca_loadings on the window's est_changes rows (all dated <= t) -> weights fixed ->
    residuals on the block's eval_changes rows.
    """
    resid_parts, rows = [], []
    for k, b in enumerate(block_schedule(est_changes.index, eval_changes.index, window, step, first_t)):
        eig, load, _ = pca_loadings(est_changes.loc[b["est"]])
        weights = hedge_weights(load, target, factor_hedges)
        ev = eval_changes.loc[b["eval"]]
        part = pd.DataFrame({"block": k, "t": b["t"], f"dy{target:g}": ev[target]}, index=ev.index)
        for h in HEDGES:
            part[h] = residuals(ev, weights[h], target)
        resid_parts.append(part)

        with np.errstate(divide="ignore", invalid="ignore"):   # PC4 eigenvalue 0 in a synthetic test
            sep = separation_table({"window": {"load": load, "eig": eig}}).iloc[0]
        row = {"block": k, "t": b["t"], "est first": b["est"][0], "n est": len(b["est"]),
               "eval first": b["eval"][0], "eval last": b["eval"][-1], "n eval": len(b["eval"]),
               **{f"{h} w{m:g}": w for h in HEDGES for m, w in weights[h].items()},
               "H3 total": weights["H3"].sum(),
               "PC3/PC4 eigenvalue ratio": sep["PC3/PC4 eigenvalue ratio"],
               "cond (2/5/10y)": sep["cond (2/5/10y)"],
               f"PC3 share <= {SHORT_END:g}y": float((load.loc[load.index <= SHORT_END, "PC3"] ** 2).sum()),
               **{f"PC3 at {m:g}y": load["PC3"].get(m, np.nan) for m in PC3_SHOWN},
               "max |H3 exposure|": factor_exposure(load, weights["H3"], factor_hedges["H3"][1],
                                                    target).abs().max()}
        if reference is not None:
            row["PC3 best match"], row["its cos"] = best_match(load[["PC3"]], reference)[0]
        rows.append(row)
    return pd.concat(resid_parts), pd.DataFrame(rows)


def with_block_sd(blocks, resid):
    """'blocks' plus 'block sd H0'..'block sd H3': each hedge's residual sd (bp/day) over the days of
    'resid' in that block. Pass the days the run's statistics use (main run: suspect days out)."""
    return blocks.join(resid.groupby("block")[HEDGES].std().add_prefix("block sd "), on="block")


# Tables (floats; the report rounds them)

def nw_lags(n):
    """Newey-West (1994) rule-of-thumb lags for n daily observations: floor(4 (n / 100)^(2/9)), an int."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def dm_test(r_model, r_bench, lags=None, groups=None):
    """Paired Diebold-Mariano style test on squared residuals.

    Inputs: r_model, r_bench - residuals of two hedges on the same days (bp); lags - Newey-West lags
            (used when groups is None); groups - optional block id per day for block-clustered errors.
    Output: dict n, 'se type' ('NW lag k' or 'cluster, G blocks'), 'mean d' (bp^2/day), 'se', 't', 'p'.
    Sign: d_t = r_model_t^2 - r_bench_t^2, so a NEGATIVE mean d and t mean the model hedge leaves
    smaller squared residuals. Method: d regressed on a constant with statsmodels. Newey-West: HAC with
    Bartlett weights 1 - j / (lags + 1), no small-sample correction, two-sided p from t(n - 1).
    Clustered: var(mean d) = G / (G - 1) x sum over blocks of (sum of demeaned d in the block)^2 / n^2,
    any correlation allowed within a block (one set of weights), none across; p from t(G - 1).
    """
    d = np.asarray(r_model, dtype=float) ** 2 - np.asarray(r_bench, dtype=float) ** 2
    if groups is None:
        cov_type, label = "HAC", f"NW lag {lags}"
        cov = {"maxlags": lags, "kernel": "bartlett", "use_correction": False}
    else:
        groups = np.asarray(groups)
        cov_type, cov, label = "cluster", {"groups": groups}, f"cluster, {len(np.unique(groups))} blocks"
    fit = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type=cov_type, cov_kwds=cov, use_t=True)
    return {"n": len(d), "se type": label, "mean d": fit.params[0], "se": fit.bse[0],
            "t": fit.tvalues[0], "p": fit.pvalues[0]}


def summary_table(resid, blocks):
    """Residual statistics per hedge on the days in 'resid' (blocks gives the weights in force).

    Output: one row per hedge: n days, mean and sd (ddof=1) of r in bp/day, 'variance removed %' =
    100 (1 - var(r_h) / var(r_H0)) on the same days, and 'rounding floor sd' = sqrt(mean over days of
    (1 + sum_i w_i^2) x 1/6), the sd r would have from rounding alone: F17 yields have 2 decimals, so a
    daily change carries rounding noise of variance 1/6 bp^2 (m3_factors.ROUNDING_VAR_BP2), assumed
    independent across maturities and days. Compare floor^2 with sd^2 (variances add; sds do not).
    """
    var0, by_block, rows = resid["H0"].var(), blocks.set_index("block"), []
    for h in HEDGES:
        sum_w2 = (by_block[[c for c in WEIGHT_COLS if c.startswith(h + " ")]] ** 2).sum(axis=1)  # 0 for H0
        floor = np.sqrt(((1.0 + resid["block"].map(sum_w2)) * ROUNDING_VAR_BP2).mean())
        rows.append({"hedge": h, "what": HEDGE_TEXT[h], "n days": len(resid), "mean (bp/day)": resid[h].mean(),
                     "sd (bp/day)": resid[h].std(), "variance removed %": 100 * (1 - resid[h].var() / var0),
                     "rounding floor sd": floor})
    return pd.DataFrame(rows)


def episode_table(resid, blocks):
    """Residual sd per hedge and variance removed vs H0, by the README episode. Each out-of-sample day goes
    to the episode of its own date (m2_expectations.episode_of, by calendar month). Output: one row per
    episode with days: 'n days', 'n blocks' (blocks with a day in it), 'sd H0'..'sd H3' (bp/day),
    '% removed H1'..'% removed H3', 'floor H3' (H3's rounding floor sd on those days, as in summary_table).
    """
    names = episode_of(resid.index)
    rows = []
    for name, first, last in EPISODES:
        part = resid[names == name]
        if len(part) < 2:
            continue
        row = {"episode": name, "months": f"{first} to {last}", "n days": len(part),
               "n blocks": part["block"].nunique()}
        row.update({f"sd {h}": part[h].std() for h in HEDGES})
        row.update({f"% removed {h}": 100 * (1 - part[h].var() / part["H0"].var()) for h in HEDGES[1:]})
        row["floor H3"] = summary_table(part, blocks).set_index("hedge").loc["H3", "rounding floor sd"]
        rows.append(row)
    return pd.DataFrame(rows)


def dm_table(resid):
    """Paired tests for PAIRS. Full sample: Newey-West at nw_lags(n) (the planned rule) and clustered by
    block (all days of a block share one set of estimated weights, so an estimation error can shift d
    for a whole block). Each episode: Newey-West at nw_lags(n) only (an episode spans too few blocks to
    cluster).
    Input: resid with a 'block' column. Output: one row per (pair, sample, se type).
    """
    names = episode_of(resid.index)
    rows = []
    for model, bench in PAIRS:
        pair = f"{model} vs {bench}"
        rows.append({"test": pair, "sample": "all", **dm_test(resid[model], resid[bench], nw_lags(len(resid)))})
        rows.append({"test": pair, "sample": "all", **dm_test(resid[model], resid[bench], groups=resid["block"])})
        for name, _, _ in EPISODES:
            part = resid[names == name]
            if len(part) > 1:
                rows.append({"test": pair, "sample": name, **dm_test(part[model], part[bench], nw_lags(len(part)))})
    return pd.DataFrame(rows)


def variant_row(label, resid):
    """One row of the robustness and threshold tables: blocks, days, sd per hedge, H3's variance removed,
    and for each pair in PAIRS the paired-test t at Newey-West lag nw_lags(n) ('t H3 vs H1'; NaN for
    fewer than 3 days) and block-clustered ('t cl H3 vs H1'; NaN for a single block)."""
    row = {"rows": label, "n blocks": resid["block"].nunique(), "n days": len(resid),
           **{f"sd {h}": resid[h].std() for h in HEDGES},
           "% removed H3": 100 * (1 - resid["H3"].var() / resid["H0"].var())}
    for a, b in PAIRS:
        row[f"t {a} vs {b}"] = dm_test(resid[a], resid[b], nw_lags(len(resid)))["t"] if len(resid) > 2 else np.nan
        row[f"t cl {a} vs {b}"] = (dm_test(resid[a], resid[b], groups=resid["block"])["t"]
                                   if resid["block"].nunique() > 1 else np.nan)
    return row


def with_fallback(resid, blocks, threshold=COND_FLAG):
    """Copy of 'resid' with H3 replaced by H2 in every block whose cond(2/5/10y) at t > threshold.
    Usable at t (cond uses only the window), but the threshold was set after seeing the main results,
    so this is a robustness check, not a clean out-of-sample test. 'resid' itself is not changed.
    """
    swap = resid["block"].isin(blocks.loc[blocks["cond (2/5/10y)"] > threshold, "block"])
    return resid.assign(H3=resid["H3"].where(~swap, resid["H2"]))


def threshold_table(runs, thresholds=THRESHOLDS):
    """The fallback (with_fallback) at several thresholds. Input: runs - dict run name -> (resid, blocks).
    Output: one variant_row per (run, threshold), plus 'blocks swapped' (blocks with cond > threshold)."""
    return pd.DataFrame([{**variant_row(f"{run}, cond > {th}", with_fallback(resid, blocks, th)),
                          "blocks swapped": int((blocks["cond (2/5/10y)"] > th).sum())}
                         for run, (resid, blocks) in runs.items() for th in thresholds])


# Charts

def plot_weights(resid, blocks, path):
    """weights_over_time.png: H2 and H3 weights per out-of-sample day (steps at re-estimation), H1's
    0.375 / 0.625 dashed for reference, and cond(2/5/10y) on a log scale with COND_FLAG dashed.

    Inputs: resid - main-run days with a 'block' column (the x axis); blocks - main-run blocks table
    (reads 'H2 w2'..'H3 w10', 'cond (2/5/10y)', 'eval first', 'eval last'); path - PNG file. The H3
    panel is clipped to H3_YLIM so the stable blocks stay readable; blocks with a weight outside it
    are written on the panel.
    """
    by_block = blocks.set_index("block")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, facecolor=COLORS["surface"])
    for ax, h in zip(axes[:2], FACTOR_HEDGES):
        ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
        for m in FACTOR_HEDGES[h][0]:
            ax.plot(resid.index, resid["block"].map(by_block[f"{h} w{m:g}"]), color=LEG_COLORS[m],
                    linewidth=2, label=f"{m:g}y")
        for m, w in h1_weights().items():
            ax.axhline(w, color=LEG_COLORS[m], linewidth=1, linestyle="--", alpha=0.7)
        style_axes(ax, f"{h} weights ({HEDGE_TEXT[h]}); dashed = H1", "DV01 weight")
        ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["ink"], loc="lower right", ncol=3)
    cols = [f"H3 w{m:g}" for m in FACTOR_HEDGES["H3"][0]]
    axes[1].set_ylim(*H3_YLIM)
    off = by_block[((by_block[cols] < H3_YLIM[0]) | (by_block[cols] > H3_YLIM[1])).any(axis=1)]
    for _, b in off.iterrows():
        note = f" off scale {b['eval first']:%d %b} - {b['eval last']:%d %b %Y}: "
        note += " / ".join(f"{b[c]:+.2f}" for c in cols)
        axes[1].text(b["eval first"], H3_YLIM[0] + 0.05, note, fontsize=7, color=COLORS["muted"])
    ax = axes[2]
    ax.plot(resid.index, resid["block"].map(by_block["cond (2/5/10y)"]), color=COLORS["ink"], linewidth=1.5)
    ax.axhline(COND_FLAG, color=COLORS["muted"], linewidth=1, linestyle="--")
    ax.set_yscale("log")
    style_axes(ax, f"Condition number of PC1-3 at 2/5/10y (the H3 system); dashed = {COND_FLAG}", "cond (log scale)")
    ax.set_xlabel("out-of-sample date", color=COLORS["muted"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cumulative(resid, path):
    """cumulative_loss_difference.png: running sum of r_H3^2 - r_bench^2 for each pair in PAIRS
    (falling = H3 ahead); grey lines mark episode starts (the README episodes). Inputs: resid - main-run
    days with columns H1-H3; path - PNG file."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=COLORS["surface"])
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
    for name, first, _ in EPISODES:
        start = pd.Timestamp(first + "-01")
        if resid.index[0] < start <= resid.index[-1]:
            ax.axvline(start, color=COLORS["grid"], linewidth=1)
            ax.text(start, 1, f" {name}", transform=ax.get_xaxis_transform(), fontsize=7, va="top",
                    color=COLORS["muted"])
    for model, bench in PAIRS:
        ax.plot(resid.index, (resid[model] ** 2 - resid[bench] ** 2).cumsum(), color=BENCH_COLORS[bench],
                linewidth=2, label=f"{model} vs {bench}")
    style_axes(ax, "Cumulative r_H3^2 - r_bench^2 out of sample (falling = H3 better)", "bp^2, cumulative")
    ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["ink"], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Report

def verdict(row):
    """Plain words for a paired test of H3 against a benchmark (row from dm_table; t < 0 = H3 better)."""
    if row["p"] < 0.05:
        return "beats" if row["t"] < 0 else "does worse than"
    return "is not significantly better than" if row["t"] < 0 else "does not beat"


def row_of(frame, prefix):
    """The one row of a variant_row table whose 'rows' label starts with 'prefix' (text never depends on
    row order). .item() raises ValueError unless exactly one row matches."""
    return frame.loc[frame.index[frame["rows"].str.startswith(prefix)].item()]


def write_report(ctx, path):
    """Write outputs/m4/hedging_report.md from the tables in ctx (built in main()).

    ctx keys: 'main' (main-run residuals, suspect days out; columns block, H0-H3), 'resid' (main run,
    every evaluation day), 'reference' (full-sample loadings), 'blocks' / 'blocks_k' / 'blocks_s'
    (blocks tables with block sds: main, keep-suspect and PCA 1-10y runs), and the tables 'summary',
    'episodes', 'episodes_ok' (episodes on the cond <= COND_FLAG blocks), 'dm', 'robust' (rows found by
    their number with row_of), 'thresholds'. Every number in the text is read from these. Judgement,
    written for data to 30 Sep 2026 (re-read it if the data change): the Answer's bullet titles,
    COND_FLAG, the limitations.
    """
    main, blocks, blocks_k = ctx["main"], ctx["blocks"], ctx["blocks_k"]
    ep, ep_ok = ctx["episodes"], ctx["episodes_ok"]
    summ, robust, thr = ctx["summary"].set_index("hedge"), ctx["robust"], ctx["thresholds"]
    n_blocks, lag, cond = len(blocks), nw_lags(len(main)), blocks["cond (2/5/10y)"]
    full = ctx["dm"][ctx["dm"]["sample"] == "all"].set_index(["test", "se type"])
    h3_h1, h3_h1_cl, h3_h2, h3_h2_cl = (full.loc[(f"H3 vs {b}", se)] for b in ("H1", "H2")
                                        for se in (f"NW lag {lag}", f"cluster, {n_blocks} blocks"))
    worst_rows = blocks.loc[[cond.idxmax()]]                           # the worst-conditioned block (one row)
    worst, k = worst_rows.iloc[0], int(worst_rows["block"].iloc[0])
    worst_k = blocks_k.set_index("block").loc[k]                        # the same block, suspect days kept
    worst_floor = summary_table(main[main["block"] == k], blocks).set_index("hedge").loc["H3", "rounding floor sd"]
    others, bad_k = blocks[blocks["block"] != k], blocks_k[blocks_k["cond (2/5/10y)"] > COND_FLAG]
    keep, short, short_k, good = (row_of(robust, f"{i} ") for i in (3, 6, 7, 8))
    share = f"PC3 share <= {SHORT_END:g}y"
    belly = blocks[blocks["PC3 best match"] != "PC3"]                   # windows whose PC3 is a 2-10y curvature
    ref_share = float((ctx["reference"].loc[ctx["reference"].index <= SHORT_END, "PC3"] ** 2).sum())
    wins = {b: int(((main["H3"] ** 2 - main[b] ** 2).groupby(main["block"]).mean() < 0).sum()) for b in ("H1", "H2")}
    d_h1 = main["H3"] ** 2 - main["H1"] ** 2
    acf = [" / ".join(f"{d.autocorr(j):.2f}" for j in (1, 5, 10)) for d in (d_h1, d_h1[main["block"] != k])]
    best = ep[["sd H1", "sd H2", "sd H3"]].idxmin(axis=1) == "sd H3"   # episodes where H3 has the lowest sd
    best_ok = ep_ok[["sd H1", "sd H2", "sd H3"]].idxmin(axis=1) == "sd H3"
    not_best, ep_ok_by = ep.loc[~best, "episode"].tolist(), ep_ok.set_index("episode")
    rounding_pct = 100 * ep["floor H3"] ** 2 / ep["sd H3"] ** 2      # variances add; sds do not
    other_sd = np.sqrt((ep["sd H3"] ** 2 - ep["floor H3"] ** 2).clip(lower=0))
    floor_pct = ", ".join(f"{100 * summ.loc[h, 'rounding floor sd'] ** 2 / summ.loc[h, 'sd (bp/day)'] ** 2:.0f}% ({h})"
                          for h in HEDGES[1:])
    missed = blocks_k[(blocks_k["cond (2/5/10y)"] <= COND_FLAG) & (blocks_k["block sd H3"] > blocks_k["block sd H2"])]
    n_ep_tests = int((ctx["dm"]["sample"] != "all").sum())

    def span(frame):
        """'first eval day - last eval day' of a set of blocks."""
        return f"{frame['eval first'].min():%d %b %Y} - {frame['eval last'].max():%d %b %Y}"

    # Text versions of the tables ('t cl' = block-clustered t; NaN for a single block -> 'n/a').
    tcols = [f"t {a} vs {b}" for a, b in PAIRS] + [f"t cl {a} vs {b}" for a, b in PAIRS]
    grp_digits = {**{f"sd {h}": 2 for h in HEDGES}, "% removed H3": 1, **{c: 2 for c in tcols}}
    sum_txt = to_text(ctx["summary"], {"mean (bp/day)": 3, "sd (bp/day)": 2, "variance removed %": 1,
                                       "rounding floor sd": 2}, signed=["mean (bp/day)"])
    ep_txt = to_text(ep, {c: (1 if c.startswith("%") else 2) for c in ep.columns
                          if c.startswith(("sd", "%", "floor"))})
    dm_txt = to_text(ctx["dm"], {"mean d": 3, "se": 3, "t": 2}, signed=["mean d", "t"]).assign(
        p=[fmt_p(p) for p in ctx["dm"]["p"]])
    rob_txt = to_text(robust, grp_digits, signed=tcols).replace({"nan": "n/a", "+nan": "n/a"})
    thr_txt = to_text(thr[["rows", "blocks swapped", "sd H3"] + tcols], {"sd H3": 2, **{c: 2 for c in tcols}}, tcols)
    shown = blocks[(cond > COND_FLAG) | (blocks["PC3 best match"] != "PC3")]   # rows of the transposed table
    fields = {"eval first": "{:%d %b %Y}", "eval last": "{:%d %b %Y}", "cond (2/5/10y)": "{:.1f}",
              "PC3/PC4 eigenvalue ratio": "{:.2f}", "PC3 best match": "{}", "its cos": "{:.2f}", share: "{:.0%}",
              **{f"PC3 at {m:g}y": "{:+.3f}" for m in PC3_SHOWN},
              **{c: "{:+.2f}" for c in ["H3 w2", "H3 w5", "H3 w10", "H3 total"]},
              **{f"block sd {h}": "{:.2f}" for h in HEDGES}}
    blk_txt = pd.DataFrame({"field": list(fields), **{f"block {int(b['block'])}": [f.format(b[c]) for c, f in
                                                                                   fields.items()]
                                                     for _, b in shown.iterrows()}})

    answer = [
        f"- **No, not over the whole sample with the planned PCA.** Out of sample ({len(main):,} days, "
        f"{span(blocks)}, {n_blocks} blocks of up to {STEP} days) the unhedged 7-year position moves "
        f"{summ.loc['H0', 'sd (bp/day)']:.2f} bp/day; duration (H1) leaves {summ.loc['H1', 'sd (bp/day)']:.2f}, "
        f"level + slope (H2) {summ.loc['H2', 'sd (bp/day)']:.2f} and level + slope + curvature (H3) "
        f"{summ.loc['H3', 'sd (bp/day)']:.2f}. H3 {verdict(h3_h1_cl)} H1 (t = {h3_h1['t']:+.2f}, p = "
        f"{fmt_p(h3_h1['p'])} at Newey-West lag {lag}; p = {fmt_p(h3_h1_cl['p'])} block-clustered) and "
        f"{verdict(h3_h2_cl)} H2 (t = {h3_h2['t']:+.2f}, p = {fmt_p(h3_h2['p'])}; p = {fmt_p(h3_h2_cl['p'])} "
        "block-clustered).",
        f"- **One block decides it.** For {span(worst_rows)}, cond(2/5/10y) was {worst['cond (2/5/10y)']:.0f} at t "
        f"({others['cond (2/5/10y)'].max():.1f} or less in every other block). The window's PC3 had "
        f"{worst[share]:.0%} of its squared loading at 1y or shorter and loaded {worst['PC3 at 2y']:+.2f} / "
        f"{worst['PC3 at 5y']:+.2f} / {worst['PC3 at 10y']:+.2f} at 2/5/10y, so H3's weights were "
        f"{worst['H3 w2']:+.2f} / {worst['H3 w5']:+.2f} / {worst['H3 w10']:+.2f} (total DV01 "
        f"{worst['H3 total']:.2f}). "
        f"H3 left {worst['block sd H3']:.2f} bp/day there, more than no hedge ({worst['block sd H0']:.2f}); rounding "
        f"noise amplified by those weights could be {100 * worst_floor ** 2 / worst['block sd H3'] ** 2:.0f}% of "
        "that variance.",
        f"- **Elsewhere H3 is usually the best hedge** (a split made after seeing the results). Its mean squared "
        f"residual is below H1's in {wins['H1']} of {n_blocks} blocks and below H2's in {wins['H2']}. On the other "
        f"{int(good['n blocks'])} blocks it leaves {good['sd H3']:.2f} bp/day against {good['sd H1']:.2f} (H1) and "
        f"{good['sd H2']:.2f} (H2): t vs H1 = {good['t H3 vs H1']:+.1f} at lag {lag}, {good['t cl H3 vs H1']:+.1f} "
        "block-clustered.",
        f"- **The verdict is fragile.** Keeping the two suspect F17 days in estimation fixes that block (cond "
        f"{worst_k['cond (2/5/10y)']:.1f}, H3 {worst_k['block sd H3']:.2f} vs H1 {worst_k['block sd H1']:.2f}), but "
        f"then {len(bad_k)} blocks ({span(bad_k)}) have cond > {COND_FLAG} and H3's sd is {keep['sd H3']:.2f}. With "
        f"the PCA on {PCA_FROM_CHECK:g}-10y instead of 0.25-10y (section 5, rows 6-7, chosen after seeing the "
        f"results) cond stays at or below {ctx['blocks_s']['cond (2/5/10y)'].max():.1f} and H3 leaves "
        f"{short['sd H3']:.2f} bp/day (block-clustered t = {short['t cl H3 vs H1']:+.1f} vs H1, "
        f"{short['t cl H3 vs H2']:+.1f} vs H2; {short_k['sd H3']:.2f} with the suspect days). So the answer turns "
        "on letting the 0.25-0.75y end shape PC3.",
        f"- **H2 is the steady hedge:** {summ.loc['H2', 'sd (bp/day)']:.2f} bp/day overall, weights within "
        f"{blocks['H2 w2'].min():.2f}-{blocks['H2 w2'].max():.2f} (2y) and {blocks['H2 w10'].min():.2f}-"
        f"{blocks['H2 w10'].max():.2f} (10y) in every block.",
    ]
    parts = [
        "# M4 - hedging test report",
        f"Generated by `src/m4_hedging.py` on {pd.Timestamp.today():%d %b %Y} from F17 zero-coupon yields "
        "(`data.load_f17`, `data.daily_changes_bp`). Position: long a 7-year zero, 1 unit of DV01; residual r = dy7 "
        "- sum_i w_i dy_i in bp of DV01 (weights in DV01 units; hedges in section 1). PCA (`m3_factors.pca_loadings`, "
        f"covariance, 0.25-10y) on the trailing {WINDOW} changes up to and including t, weights fixed for the next "
        f"{STEP} F17 changes, then t moves {STEP} changes on.",
        "## Answer", "Does a 3-factor hedge (H3) leave less risk than duration (H1)?", "\n".join(answer),
        "## 1. Overall, out of sample", md_table(sum_txt),
        f"Means are at most {summ['mean (bp/day)'].iloc[1:].abs().max():.3f} bp/day for H1-H3, so sd and root mean "
        "square barely differ. 'rounding floor sd' = the sd r would have from F17's rounding to whole bp alone: it "
        f"could account for {floor_pct} of the residual variance.",
        "## 2. By episode", md_table(ep_txt),
        f"On all blocks H3 has the lowest sd of H1-H3 in {int(best.sum())} of {len(ep)} episodes (not in "
        f"{', '.join(not_best) or 'none'}); without block {k} in {int(best_ok.sum())} of {len(ep_ok)}"
        + (f" ({not_best[0]}: " + ", ".join(f"{h} {ep_ok_by.loc[not_best[0], 'sd ' + h]:.2f}"
                                             for h in ("H3", "H2", "H1")) + ")" if not_best else "")
        + f". Where H3 is lowest, rounding alone could account for {rounding_pct[best].min():.0f}-"
        f"{rounding_pct[best].max():.0f}% of its variance; the rest has an sd of {other_sd[best].min():.2f}-"
        f"{other_sd[best].max():.2f} bp/day (sqrt(sd^2 - floor^2)). {ep['episode'].iloc[0]} holds only "
        f"{int(ep['n days'].iloc[0])} days.",
        "## 3. Paired tests on squared residuals", md_table(dm_txt),
        "![Cumulative loss difference](cumulative_loss_difference.png)",
        f"Negative t = H3 better. Lag {lag} is too short for the full sample: the autocorrelation of d (H3 vs H1) "
        f"at lags 1 / 5 / 10 is {acf[0]} over all blocks and {acf[1]} without block {k}, so the persistence is "
        "mostly that block's shift in mean. All days of a block share one set of weights, so the block-clustered "
        f"rows (p from t({n_blocks - 1})) are the conservative check; neither full-sample verdict changes. H3's mean "
        f"squared residual is smaller in {wins['H1']} of {n_blocks} blocks against H1 and {wins['H2']} against H2: "
        "in the chart the sums turn on one 2019 jump. Episode p-values are optimistic: an episode spans only "
        f"{ep['n blocks'].min()}-{ep['n blocks'].max()} blocks (too few to cluster), and they are not adjusted for "
        f"the {n_ep_tests} tests (Bonferroni would need p < {0.05 / n_ep_tests:.4f}).",
        "## 4. Hedge weights over time and the 2019 block",
        "![Weights over time](weights_over_time.png)",
        f"Blocks with cond > {COND_FLAG} or whose PC3 best matches another full-sample PC ('H3 total' = H3's DV01 "
        "per unit of the position's; 'block sd' = residual sd in that block):",
        md_table(blk_txt),
        f"In this 0.25-10y PCA, PC3 is mostly a money-market factor: {ref_share:.0%} of the full-sample PC3's "
        f"squared loading sits at 1y or shorter, {worst[share]:.0%} in block {k}, whose PC3 barely touches 2-10y "
        f"(table above), so zeroing a negligible exposure took the large weights. 'PC3 best match' compares all 40 "
        f"maturities, so the short end decides it and it does not show this. The {len(belly)} blocks whose PC3 is "
        "a 2-10y curvature "
        f"(best match {', '.join(sorted(set(belly['PC3 best match'])))}, at most {belly[share].max():.0%} at 1y or "
        f"shorter) hedged well (block sd H3 {belly['block sd H3'].min():.2f}-{belly['block sd H3'].max():.2f} vs H1 "
        f"{belly['block sd H1'].min():.2f}-{belly['block sd H1'].max():.2f}), while other windows with a short-end "
        f"PC3 (shares up to {others.loc[~others['block'].isin(belly['block']), share].max():.0%}) kept cond <= "
        f"{others['cond (2/5/10y)'].max():.1f}. PC3's identity and the conditioning cannot be separated here. "
        "Section 5, rows 8-9, splits the blocks by cond.",
        "## 5. Suspect days, PCA range and a fallback rule", md_table(rob_txt),
        "Rows 2-3 put the 4 suspect-day changes back (evaluation only; estimation and evaluation). Rows 4-5 replace "
        f"H3 by H2 in blocks with cond > {COND_FLAG} at t. Rows 6-7 estimate the PCA on {PCA_FROM_CHECK:g}-10y only "
        f"(max cond {ctx['blocks_s']['cond (2/5/10y)'].max():.1f}). Rows 8-9 split row 1 by cond at t. The "
        "threshold and the range were chosen after seeing row 1, so rows 4-9 show what such a rule would do, not a "
        "clean test of it. The fallback at other thresholds:",
        md_table(thr_txt),
        f"In the main run any threshold between {others['cond (2/5/10y)'].max():.1f} and "
        f"{worst['cond (2/5/10y)']:.0f} swaps only block {k} and gives row 4. In the keep-suspect run the answer "
        f"moves with the threshold, and {len(missed)} block(s) with cond <= {COND_FLAG} (ids "
        f"{', '.join(str(int(b)) for b in missed['block'])}) still have a higher H3 than H2 block sd, so a gate "
        "would not catch them. No gate is used in the main test.",
        "## 6. Limitations",
        "- **Fitted smooth curve flatters factor hedges.** The 2, 5, 7 and 10-year points come from one smooth RBA "
        "fit, so the 7-year change is close to a fixed mix of its neighbours by construction; three points pin a "
        "smooth curve better than two, so H3 gains most. Residuals here are a lower bound.\n"
        "- **Constant maturity, first-order P&L.** No ageing, carry, roll-down, funding, convexity or re-hedging "
        "cost; DV01 weights stay fixed for 60 days. The rounding floors assume independent errors.\n"
        f"- **One design.** One position (7y), window {WINDOW}, step {STEP}; M5 varies them. Episode bounds are "
        "judgement (M2). Changes over holiday gaps (including the missing 20-21 Jul 2026) count as one day.",
        "Files in `outputs/m4/`: this report, the two PNGs, `residuals.csv` (main run, every evaluation day), "
        "`blocks.csv` (every block of the 4 runs) and each table above as a CSV (unrounded).",
    ]
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# Main

CSV_NAMES = {"summary": "summary", "episodes": "episodes", "episodes_ok": "episodes_cond_le_flag",
             "dm": "dm_tests", "robust": "robustness", "thresholds": "fallback_thresholds"}


def main():
    """Run M4 end to end: daily changes without (main) and with the suspect days; full-sample PCA (only
    to name each window's PC3); rolling test (estimate on clean changes, evaluate on all); the
    keep-suspect run and the two PCA 1-10y runs on the same blocks; tables, charts, CSVs, report;
    a short printed summary.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yields = load_f17("yields")
    changes = daily_changes_bp(yields, MATURITIES, drop_suspect=True)
    changes_all = daily_changes_bp(yields, MATURITIES, drop_suspect=False)
    _, reference, _ = pca_loadings(changes)

    resid, blocks = rolling_hedges(changes, changes_all, reference)
    resid["suspect"] = ~resid.index.isin(changes.index)
    resid["episode"] = episode_of(resid.index)
    main_days = resid[~resid["suspect"]]
    first_t, short = blocks["t"].iloc[0], [m for m in MATURITIES if m >= PCA_FROM_CHECK]
    resid_k, blocks_k = rolling_hedges(changes_all, changes_all, reference, first_t=first_t)
    resid_s, blocks_s = rolling_hedges(changes[short], changes_all[short], reference, first_t=first_t)
    resid_sk, blocks_sk = rolling_hedges(changes_all[short], changes_all[short], reference, first_t=first_t)
    for other in (resid_k, resid_s, resid_sk):
        if not other.index.equals(resid.index):
            raise ValueError("every run must evaluate the same days as the main run")
    main_s = resid_s[~resid["suspect"].to_numpy()]
    blocks, blocks_k = with_block_sd(blocks, main_days), with_block_sd(blocks_k, resid_k)
    blocks_s, blocks_sk = with_block_sd(blocks_s, main_s), with_block_sd(blocks_sk, resid_sk)
    good = main_days["block"].isin(blocks.loc[blocks["cond (2/5/10y)"] <= COND_FLAG, "block"])  # ex-post split

    variants = {
        "1 main: suspect days out of estimation and evaluation": main_days,
        "2 evaluation keeps the suspect days": resid,
        "3 estimation and evaluation keep them": resid_k,
        f"4 row 1, H3 -> H2 when cond > {COND_FLAG}": with_fallback(main_days, blocks),
        f"5 row 3, H3 -> H2 when cond > {COND_FLAG}": with_fallback(resid_k, blocks_k),
        f"6 row 1, PCA on {PCA_FROM_CHECK:g}-10y": main_s,
        f"7 row 3, PCA on {PCA_FROM_CHECK:g}-10y": resid_sk,
        f"8 row 1, only blocks with cond <= {COND_FLAG}": main_days[good],
        f"9 row 1, only blocks with cond > {COND_FLAG}": main_days[~good],
    }
    ctx = {"main": main_days, "resid": resid, "reference": reference,
           "blocks": blocks, "blocks_k": blocks_k, "blocks_s": blocks_s,
           "summary": summary_table(main_days, blocks), "episodes": episode_table(main_days, blocks),
           "episodes_ok": episode_table(main_days[good], blocks), "dm": dm_table(main_days),
           "robust": pd.DataFrame([variant_row(k, v) for k, v in variants.items()]),
           "thresholds": threshold_table({"main": (main_days, blocks), "keep suspect days": (resid_k, blocks_k)})}

    resid.to_csv(OUT_DIR / "residuals.csv", index_label="date")
    runs = {"main": blocks, "keep suspect days": blocks_k, f"PCA {PCA_FROM_CHECK:g}-10y": blocks_s,
            f"PCA {PCA_FROM_CHECK:g}-10y, keep suspect days": blocks_sk}
    pd.concat([b.assign(run=name) for name, b in runs.items()]).to_csv(OUT_DIR / "blocks.csv", index=False)
    for key, name in CSV_NAMES.items():
        ctx[key].to_csv(OUT_DIR / f"{name}.csv", index=False)
    plot_weights(main_days, blocks, OUT_DIR / "weights_over_time.png")
    plot_cumulative(main_days, OUT_DIR / "cumulative_loss_difference.png")
    write_report(ctx, REPORT_PATH)

    print(f"M4: {len(blocks)} blocks, {len(main_days)} out-of-sample days "
          f"({int(resid['suspect'].sum())} suspect-day changes left out of the main statistics)")
    print(ctx["summary"][["hedge", "sd (bp/day)", "variance removed %"]].round(3).to_string(index=False))
    full = ctx["dm"][ctx["dm"]["sample"] == "all"]
    print(full[["test", "se type", "mean d", "t", "p"]].round(4).to_string(index=False))
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
