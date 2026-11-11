"""M3 - factor structure of daily F17 yield changes (PCA).

Daily changes in bp of F17 zero-coupon yields at 0.25-10 years (40 series;
the 0-year point is the cash rate), without the changes into and out of
data.SUSPECT_DAYS. PCA on the covariance matrix for the full sample and four
periods (signs: PC1 sums > 0, PC2 > 0 at 10y, PC3 > 0 at 5y); variance
explained; loadings charts; stability (cosines with the full sample and with
the rest of the sample); PC3 separation and the conditioning of M4's 2/5/10y
hedge; robustness (keep suspect days, correlation matrix, 0.5-10 years, YCC
windows). Writes outputs/m3/ (report, two PNGs, unrounded CSVs).
pca_loadings() is the one function M4 imports; it has no side effects.

Run from the project folder:  python src/m3_factors.py
Units: bp per F17 business day; loadings are length-1 eigenvectors (no unit).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no screen needed
import matplotlib.pyplot as plt

# Make "import data" work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402  (import after the sys.path line on purpose)
    PROJECT_DIR, MATURITY_GRID, SUSPECT_DAYS, load_f17, daily_changes_bp,
)
from m1_validate import md_table, style_axes, COLORS  # noqa: E402  (same table and chart style as M1)


# Settings

OUT_DIR = PROJECT_DIR / "outputs" / "m3"
REPORT_PATH = OUT_DIR / "factors_report.md"

MATURITIES = [m for m in MATURITY_GRID if m > 0]           # 0.25 to 10 years, 40 series
MATURITIES_NO_3M = [m for m in MATURITIES if m >= 0.5]     # robustness: 0.5 to 10 years, 39 series
N_SHOW = 3                                                 # factors reported: level, slope, curvature
PC_NAMES = ["PC1", "PC2", "PC3"]
N_MATCH = 5            # 'best match' searches the reference's PC1-PC5, so a swap with PC4 or PC5 shows
HEDGE_MATURITIES = [2.0, 5.0, 10.0]                        # M4's H3 hedge instruments
ROUNDING_VAR_BP2 = 1 / 6   # F17 yields have 2 decimals: rounding error U(-0.5, 0.5) bp has variance 1/12,
                           # and a change is the difference of two such errors (2/12 = 1/6 bp^2 per maturity
                           # per day, if the errors are independent from day to day)

# Sign rules. PC2 and PC3 are signed at these maturities; if a
# maturity is not in the data, the nearest one is used (see apply_sign_rules).
PC2_SIGN_MATURITY = 10.0
PC3_SIGN_MATURITY = 5.0

# Periods. A change belongs to the date it ends on. Bounds are
# inclusive calendar dates; None = start or end of the data. The YCC period
# runs by month (Mar 2020 - Nov 2021), so it also holds the
# 2-18 Mar 2020 sell-off before YCC began. Jan-Feb 2020 and Dec 2021 are in
# the full sample but in no period.
YCC_LABEL = "YCC (Mar 2020 - Nov 2021)"
PERIODS = {
    "Full sample": (None, None),
    "2017-2019": ("2017-01-01", "2019-12-31"),
    YCC_LABEL: ("2020-03-01", "2021-11-30"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-12-31"),
}
# YCC robustness windows, compared with the main YCC period. Regime closes are
# 19 Mar 2020 to 1 Nov 2021 (m1_validate.YCC_REGIMES). A change is inside the
# regime only if both of its closes are: the change dated 19 Mar 2020 runs
# from the 18 Mar close (before the announcement), so the windows start on
# 20 Mar 2020. The second window also leaves out 27 Oct - 1 Nov 2021, the
# late-Oct 2021 break in M1 section 10 (the cut date is judgement).
YCC_VARIANTS = {
    "regime closes (20 Mar 2020 - 1 Nov 2021)": ("2020-03-20", "2021-11-01"),
    "regime before the break (20 Mar 2020 - 26 Oct 2021)": ("2020-03-20", "2021-10-26"),
}

# Maturities shown in the loadings tables of the report (the CSVs hold all 40).
TABLE_MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
PC_COLORS = {"PC1": COLORS["blue"], "PC2": COLORS["orange"], "PC3": COLORS["aqua"]}


# PCA (reused by M4)

def _nearest(index, target):
    """The value in 'index' closest to 'target' (used to pick a sign maturity)."""
    values = np.asarray(index, dtype=float)
    return index[int(np.argmin(np.abs(values - target)))]


def apply_sign_rules(loadings):
    """Fix the arbitrary sign of each eigenvector with the rules.

    Input:  loadings - DataFrame, index = maturities in years (floats),
            columns = PC1, PC2, ... (eigenvectors).
    Output: a copy with columns multiplied by -1 where needed so that
              PC1 sums to a positive number (a rise raises yields: "level");
              PC2 > 0 at PC2_SIGN_MATURITY, 10y (steepens: "slope");
              PC3 > 0 at PC3_SIGN_MATURITY, 5y (lifts the belly: "curvature").
            If the sign maturity is not in the index, the nearest maturity
            is used. Exactly 0 is left as it is. PC4 onwards are not changed.
    Why:    an eigenvector v and -v describe the same factor. Without a rule,
            two runs (or two periods) can return opposite signs, and charts,
            cosines and factor-score series would not be comparable. Hedge
            weights that zero factor exposures do not depend on the sign.
    """
    out = loadings.copy()
    rules = [("PC1", None), ("PC2", PC2_SIGN_MATURITY), ("PC3", PC3_SIGN_MATURITY)]
    for name, maturity in rules:
        if name not in out.columns:
            continue
        if maturity is None:
            value = out[name].sum()
        else:
            value = out.loc[_nearest(out.index, maturity), name]
        if value < 0:
            out[name] = -out[name]
    return out


def pca_loadings(changes, standardise=False):
    """Principal components of daily yield changes.

    Input:
        changes     - DataFrame of daily changes (bp), rows = days, columns =
                      maturities in years. No missing values allowed: rows
                      are never dropped silently (a NaN raises ValueError).
        standardise - False (main): PCA on the covariance matrix, so
                      maturities that move more weigh more. True: PCA on the
                      correlation matrix (each maturity scaled to sd 1).
    Output: (eigenvalues, loadings, share)
        eigenvalues - Series indexed PC1..PCn, largest first: the variance of
                      each factor (bp^2 per day; no unit if standardise).
                      Round-off negatives (about -1e-15) are set to 0.
        loadings    - DataFrame indexed by maturity, columns PC1..PCn: the
                      eigenvectors, each of length 1, signs fixed by
                      apply_sign_rules.
        share       - Series indexed PC1..PCn: eigenvalue / sum of all
                      eigenvalues (sums to 1). The sum of eigenvalues equals
                      the total variance (trace of the matrix).
    Convention: covariance with ddof=1 (pandas default), changes demeaned by
    the covariance itself. numpy's eigh is used because the matrix is
    symmetric; it returns eigenvalues in ascending order, so they are reversed.
    """
    if changes.isna().any().any():
        raise ValueError("pca_loadings: changes contain NaN; clean them first")
    matrix = changes.corr() if standardise else changes.cov()
    values, vectors = np.linalg.eigh(matrix.to_numpy())
    order = np.argsort(values)[::-1]                        # largest first
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    names = [f"PC{k + 1}" for k in range(len(values))]
    eigenvalues = pd.Series(values, index=names, name="eigenvalue")
    loadings = pd.DataFrame(vectors, index=changes.columns, columns=names)
    loadings.index.name = "maturity_years"
    loadings = apply_sign_rules(loadings)
    share = pd.Series(values / values.sum(), index=names, name="share")
    return eigenvalues, loadings, share


def aligned_cosine(a, b):
    """Cosine similarity of two loading vectors after sign alignment.

    Inputs: a, b - Series indexed by maturity (e.g. one PC in a period and in
            the full sample).
    Output: (cosine, flipped): cosine = |a.b| / (|a| |b|) on the maturities
            both share (0 to 1; 1 = same shape); flipped = True if a.b < 0
            (the sign rules gave the two vectors opposite signs).
    The common index matters for the 0.5-10 year variant (39 maturities).
    Scale: smooth curves have high cosines with each other (a flat vector
    has about 0.98 with the full-sample PC1), so 0.9 is not "nearly equal".
    """
    common = a.index.intersection(b.index)
    x, y = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    return abs(cos), cos < 0


def best_match(load, reference, n_ref=N_MATCH):
    """For each column of 'load', the reference PC it is most similar to.

    Inputs: load - loadings to look up (one column per PC); reference -
            loadings with columns PC1, PC2, ...; n_ref - search PC1..PC{n_ref}.
    Output: list of (reference PC name, aligned cosine), one per column of
            'load'. A period PC3 that best matches reference PC4 = a swap.
    """
    out = []
    for pc in load.columns:
        cos = {ref: aligned_cosine(load[pc], reference[ref])[0] for ref in reference.columns[:n_ref]}
        name = max(cos, key=cos.get)
        out.append((name, cos[name]))
    return out


def compare_pcs(load, base):
    """Compare PC1-3 of 'load' with PC1-3 of 'base' (both loadings DataFrames).

    Output: dict with 'cos PC1'..'cos PC3' (aligned_cosine of the same PC),
    'sign flipped' (PCs that needed a flip to align, or 'none') and 'best
    match' (best_match of load's PC1-3 in base PC1-PC5, e.g. 'PC1 PC2 PC3').
    """
    row, flips = {}, []
    for pc in PC_NAMES:
        row[f"cos {pc}"], flipped = aligned_cosine(load[pc], base[pc])
        flips += [pc] if flipped else []
    row["sign flipped"] = " ".join(flips) or "none"
    row["best match"] = " ".join(name for name, _ in best_match(load[PC_NAMES], base))
    return row


def zero_crossings(series):
    """Pairs of neighbouring maturities between which 'series' changes sign.

    Input: a loading Series indexed by maturity (in maturity order).
    Output: list of (maturity before, maturity after), e.g. [(5.25, 5.5)].
    Used so the report's shape descriptions are computed, not typed.
    """
    s = np.sign(series.to_numpy(float))
    return [(series.index[i], series.index[i + 1]) for i in range(len(s) - 1) if s[i] * s[i + 1] < 0]


# Tables (floats; the report rounds them with to_text)

def slice_period(changes, bounds):
    """Rows of 'changes' whose date is inside bounds = (start, end), inclusive; None = open."""
    start, end = bounds
    return changes.loc[start:end]


def run_periods(changes, periods, standardise=False):
    """PCA for every period.

    Inputs: changes - daily changes (bp) for all dates; periods - dict
            label -> (start, end); standardise - passed to pca_loadings.
    Output: dict label -> {'bounds', 'n' (daily changes), 'first', 'last'
            (dates of the first and last change), 'eig', 'load', 'share'
            (pca_loadings), 'shape'}. 'shape' = loadings in bp units, used in
            every comparison with the main run: 'load' itself for covariance
            PCA; for correlation PCA each loading times its maturity's sd (a
            factor with standardised loading v moves maturity i by sd_i v_i).
    """
    results = {}
    for label, bounds in periods.items():
        part = slice_period(changes, bounds)
        eig, load, share = pca_loadings(part, standardise)
        shape = load.mul(part.std(), axis=0) if standardise else load
        results[label] = {"bounds": bounds, "n": len(part), "first": part.index[0], "last": part.index[-1],
                          "eig": eig, "load": load, "share": share, "shape": shape}
    return results


def variance_table(results):
    """One row per period: dates, changes, total variance and shares (%).

    Columns: first / last change date, n changes, total variance (sum of
    eigenvalues, bp^2/day), PC1-4 and cumulative PC1-3 shares, and 'rounding
    floor %' = ROUNDING_VAR_BP2 x maturities / total (compare 100 - PC1-3).
    """
    rows = []
    for label, r in results.items():
        share, total = r["share"] * 100, r["eig"].sum()
        rows.append({"period": label, "first change": r["first"], "last change": r["last"],
                     "n changes": r["n"], "total variance (bp^2/day)": total,
                     **{f"{pc} %": share[pc] for pc in PC_NAMES + ["PC4"]},
                     "PC1-3 cum %": share.iloc[:N_SHOW].sum(),
                     "rounding floor %": 100 * ROUNDING_VAR_BP2 * len(r["load"]) / total})
    return pd.DataFrame(rows)


def stability_table(results, changes, reference_label="Full sample"):
    """Cosine similarity of each period's PC1-3 with two references.

    Inputs: results - run_periods output (main); changes - the changes it used.
    Output: one row per period: 'weight in full %' (its sum of squared changes
    as a % of the full sample's); compare_pcs against the full sample (cosines
    inflated for heavy periods, which are part of it); 'cos PCk vs rest' =
    aligned_cosine with PCk of a PCA on all changes outside the period.
    """
    reference = results[reference_label]["load"]
    total_ss = float((changes ** 2).to_numpy().sum())
    rows = []
    for label, r in results.items():
        if label == reference_label:
            continue
        part = slice_period(changes, r["bounds"])
        _, rest, _ = pca_loadings(changes.drop(index=part.index))
        row = {"period": label, "weight in full %": 100 * float((part ** 2).to_numpy().sum()) / total_ss,
               **compare_pcs(r["load"], reference)}
        row.update({f"cos {pc} vs rest": aligned_cosine(r["load"][pc], rest[pc])[0] for pc in PC_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def separation_table(results):
    """How well PC3 is identified, and how well 2/5/10y pin down PC1-3 (for M4).

    Output: one row per period: 'PC3/PC4 eigenvalue ratio' (near 1 = sampling
    noise can rotate or swap them); PC3's loadings at HEDGE_MATURITIES; 'cond
    (2/5/10y)' = condition number of the 3x3 matrix of PC1-3 loadings at
    2/5/10y, the system M4's H3 hedge solves. Large = big, offsetting weights
    that are sensitive to small loading errors. A PC's sign does not change it.
    """
    rows = []
    for label, r in results.items():
        load, eig = r["load"], r["eig"]
        row = {"period": label, "PC3/PC4 eigenvalue ratio": eig["PC3"] / eig["PC4"]}
        row.update({f"PC3 at {m:g}y": load.loc[m, "PC3"] for m in HEDGE_MATURITIES})
        row["cond (2/5/10y)"] = float(np.linalg.cond(load.loc[HEDGE_MATURITIES, PC_NAMES].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(main, variants):
    """Compare each robustness variant with the main analysis, period by period.

    Inputs: main - run_periods output (main); variants - dict label ->
            (run_periods output, dict variant period -> main period).
    Output: one row per (variant, period): changes, PC1-3 shares (%),
            compare_pcs of its 'shape' (bp units) against the main period's
            loadings, and 'main PC3 found as' / 'its cos' = the variant PC
            (1-5) closest to the main PC3. If not PC3, curvature moved.
    """
    rows = []
    for variant, (results, match) in variants.items():
        for label, r in results.items():
            base = main[match[label]]["load"]
            row = {"variant": variant, "period": label, "n changes": r["n"],
                   **{f"{pc} %": r["share"][pc] * 100 for pc in PC_NAMES}, **compare_pcs(r["shape"], base)}
            row["main PC3 found as"], row["its cos"] = best_match(base[["PC3"]], r["shape"])[0]
            rows.append(row)
    return pd.DataFrame(rows)


def to_text(frame, digits, signed=()):
    """Copy of 'frame' with float columns turned into text for the report.

    digits - dict column -> decimals; signed - columns printed with + or -.
    round(v, d) + 0.0 turns -0.0 into 0.0, so -0.0004 never prints as
    '-0.00'. Only the markdown uses this; the CSVs keep unrounded floats.
    """
    out = frame.copy()
    for col, d in digits.items():
        sign = "+" if col in signed else ""
        out[col] = [f"{round(float(v), d) + 0.0:{sign}.{d}f}" for v in frame[col]]
    return out


def loadings_table(loadings, label):
    """PC1-3 loadings at TABLE_MATURITIES as a small text table (rows = PCs), 2 decimals, signed."""
    part = loadings.loc[TABLE_MATURITIES, PC_NAMES].T.reset_index(drop=True)
    part.columns = [f"{m:g}y" for m in TABLE_MATURITIES]
    table = pd.concat([pd.DataFrame({"period": label, "PC": PC_NAMES}), part], axis=1)
    return to_text(table, {c: 2 for c in part.columns}, signed=part.columns)


# Charts

def _draw_loadings(ax, r, title, ref=None):
    """Draw one run_periods entry's PC1-3 loadings on 'ax' (solid), and 'ref'
    loadings (the full sample) dashed in the same colours if given. Colours:
    blue PC1, orange PC2, aqua PC3; legend = each PC's share of variance.
    """
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
    for pc in PC_NAMES:
        if ref is not None:
            ax.plot(ref.index, ref[pc], color=PC_COLORS[pc], linewidth=1, linestyle="--", alpha=0.7)
        ax.plot(r["load"].index, r["load"][pc], color=PC_COLORS[pc], linewidth=2,
                label=f"{pc} {r['share'][pc] * 100:.1f}%")
    style_axes(ax, title, "loading (length-1 eigenvector)")
    ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["ink"], loc="lower right")


def plot_loadings(results, out_dir, reference_label="Full sample"):
    """Write loadings_full_sample.png and loadings_by_period.png (one panel
    per period vs the full sample dashed; shared axes) into out_dir. Titles
    count daily changes, not calendar days.
    """
    full = results[reference_label]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=COLORS["surface"])
    _draw_loadings(ax, full, "PCA of daily F17 zero-yield changes, 0.25-10 years, full sample "
                             f"({full['n']} changes)")
    ax.set_xlabel("maturity (years)", color=COLORS["muted"])
    fig.tight_layout()
    fig.savefig(out_dir / "loadings_full_sample.png", dpi=150)
    plt.close(fig)

    labels = [k for k in results if k != reference_label]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, facecolor=COLORS["surface"])
    for ax, label in zip(axes.ravel(), labels):
        _draw_loadings(ax, results[label], f"{label} ({results[label]['n']} changes)", ref=full["load"])
    for ax in axes[1]:
        ax.set_xlabel("maturity (years)", color=COLORS["muted"])
    fig.suptitle("Loadings by period (solid) vs full sample (dashed); % = share of the period's variance",
                 x=0.01, ha="left", fontsize=11, color=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(out_dir / "loadings_by_period.png", dpi=150)
    plt.close(fig)


# Report

def write_report(main, variants, tables, changes, n_all, path):
    """Write outputs/m3/factors_report.md.

    Inputs: main / variants / tables as built in main(); changes - the main
    daily changes; n_all - changes before the suspect-day rule; path.
    Every number in the text is read from the tables or results. Judgement,
    written for data to 30 Sep 2026 (re-read it if the data change): the
    factor names, "money-market end against the belly", the reading of YCC
    (the transitions drive PC2), "a reason to drop them" for the suspect
    days, and the cut date of the late-Oct 2021 break (YCC_VARIANTS).
    """
    var, stab = tables["variance"].set_index("period"), tables["stability"].set_index("period")
    sep, rob = tables["separation"].set_index("period"), tables["robustness"].set_index(["variant", "period"])
    fl, share = main["Full sample"]["load"], main["Full sample"]["share"] * 100
    pc1, pc2, pc3 = fl["PC1"], fl["PC2"], fl["PC3"]
    per = var["PC1-3 cum %"].drop(index="Full sample")
    lo, hi = per.idxmin(), per.idxmax()
    resid, floor = 100 - per, var["rounding floor %"].drop(index="Full sample")
    total = var["total variance (bp^2/day)"].drop(index="Full sample")
    rest = stab[[f"cos {pc} vs rest" for pc in PC_NAMES]]
    rest_other = rest.drop(index=YCC_LABEL)
    low = rest_other.iloc[:, 1:].min(axis=1).idxmin()    # period with the lowest PC2/PC3 cosine outside YCC
    ycc = main[YCC_LABEL]["load"]
    pc2_other = var["PC2 %"].drop(index=["Full sample", YCC_LABEL])
    y_close, y_calm = (rob.loc[("YCC dates", k)] for k in YCC_VARIANTS)
    keep = rob.loc[("Keep suspect days", "2017-2019")]
    keep_pc3 = variants["Keep suspect days"][0]["2017-2019"]["load"]["PC3"]
    corr = rob.loc["Correlation matrix"]
    corr_less = bool((corr["PC1 %"] < var["PC1 %"].reindex(corr.index)).all())
    corr_odd = corr.index[(corr["best match"] != " ".join(PC_NAMES)) | (corr["main PC3 found as"] != "PC3")]
    ycc_short = slice_period(changes, PERIODS[YCC_LABEL])[0.25]
    mats = pd.Series(np.asarray(fl.index, dtype=float), index=fl.index)   # benchmarks: flat and straight line
    flat_cos, line_cos = aligned_cosine(mats * 0 + 1, pc1)[0], aligned_cosine(mats - mats.mean(), pc2)[0]
    ratio, cond = sep["PC3/PC4 eigenvalue ratio"], sep["cond (2/5/10y)"]
    shares3 = ["PC1 %", "PC2 %", "PC3 %"]

    def f(values, d=2):
        """Join numbers as 'a / b / c' with d decimals (for the text)."""
        return " / ".join(f"{v:.{d}f}" for v in values)

    def between(series):
        """Where a loading changes sign, as 'between a and by'."""
        return " and ".join(f"between {a:g} and {b:g}y" for a, b in zero_crossings(series))

    # Text versions of the tables: shares 1 decimal, cosines 3, PC3 loadings 3 and signed.
    hedge_cols = [f"PC3 at {mat:g}y" for mat in HEDGE_MATURITIES]
    var_txt = to_text(tables["variance"], {"total variance (bp^2/day)": 0, **{c: 1 for c in shares3},
                                           "PC4 %": 2, "PC1-3 cum %": 1, "rounding floor %": 2})
    stab_txt = to_text(tables["stability"], {"weight in full %": 1, **{c: 3 for c in tables["stability"].columns
                                                                        if c.startswith("cos")}})
    sep_txt = to_text(tables["separation"], {"PC3/PC4 eigenvalue ratio": 2, **{c: 3 for c in hedge_cols},
                                             "cond (2/5/10y)": 1}, signed=hedge_cols)
    rob_txt = to_text(tables["robustness"], {**{c: 1 for c in shares3}, **{f"cos {pc}": 3 for pc in PC_NAMES},
                                             "its cos": 3})

    lines = [
        "# M3 - factor structure report",
        "",
        "Generated by `src/m3_factors.py` from `data/raw/rba_f17_zero_coupon_yields.csv` (`data.load_f17`, "
        f"`data.daily_changes_bp`): daily changes in bp of F17 zero-coupon yields at 0.25-10 years "
        f"({len(MATURITIES)} series; the 0-year point is the cash rate). PCA on the covariance matrix; loadings "
        "are eigenvectors of length 1.",
        "",
        "## Answer",
        "",
        f"- **Shares.** PC1-3 explain {var.loc['Full sample', 'PC1-3 cum %']:.1f}% of the variance of daily "
        f"changes (PC1 / PC2 / PC3 {f(share.iloc[:3], 1)}%), and {per[lo]:.1f}% ({lo}) to {per[hi]:.1f}% ({hi}) "
        f"by period. Most of the rest is rounding of yields to whole bp ({floor.min():.1f}-{floor.max():.1f}% "
        f"of variance, against {resid.min():.1f}-{resid.max():.1f}% left; section 1), so the period ranking "
        "says little about factor structure, and a fitted smooth curve flatters every share.",
        f"- **Shapes.** PC1 is level ({pc1.loc[0.25]:+.2f} at 0.25y, {pc1.loc[4.0]:+.2f} at 4y, "
        f"{pc1.loc[10.0]:+.2f} at 10y). PC2 is slope (lowest at {pc2.idxmin():g}y, {pc2.min():+.2f}; "
        f"{pc2.loc[10.0]:+.2f} at 10y; crossing zero {between(pc2)}). PC3 is curvature of the money-market "
        f"end against the belly (lowest at {pc3.idxmin():g}y, highest at {pc3.idxmax():g}y, changing sign "
        f"{between(pc3)}).",
        f"- **Stability.** Against the rest of the sample, PC1 keeps its shape outside YCC (cosine >= "
        f"{rest_other['cos PC1 vs rest'].min():.3f}); PC2 / PC3 vary more ({low}: {f(rest.loc[low].iloc[1:])}). "
        f"Under YCC (vs rest {f(rest.loc[YCC_LABEL])}) PC1 sits on the long end (0.25y "
        f"{round(ycc.loc[0.25, 'PC1'], 2) + 0.0:+.2f}, 10y {ycc.loc[10.0, 'PC1']:+.2f}) and PC2 has "
        f"{var.loc[YCC_LABEL, 'PC2 %']:.1f}% of variance ({pc2_other.min():.1f}-{pc2_other.max():.1f}% "
        f"elsewhere), but only {y_calm['PC2 %']:.1f}% in the regime before the late-Oct 2021 break: the extra "
        "comes mostly from 2-19 Mar 2020 and the break, not from the months the target held.",
        f"- **PC3 is weakly identified.** Its eigenvalue is {ratio['2017-2019']:.1f}x PC4's in 2017-2019 and "
        f"{ratio[YCC_LABEL]:.1f}x under YCC ({ratio['Full sample']:.1f}x full sample). Kept, the suspect days "
        "form a short-end factor that becomes PC3 in 2017-2019 and pushes curvature to "
        f"{keep['main PC3 found as']} (cosine {keep['its cos']:.2f} with the main PC3): a reason to drop them. "
        "For M4, a PC's sign does not change hedge weights; PC3's identity and how well 2/5/10y pin it down "
        f"do (condition number {cond['2017-2019']:.0f} in 2017-2019, {cond['Full sample']:.1f} full sample).",
        "",
        "## 1. Variance explained",
        "",
        md_table(var_txt),
        "",
        f"Total variance = the sum of the 40 maturities' daily variances ({total.idxmax()} most volatile, "
        f"{total.idxmin()} least). 'rounding floor %' = the rounding noise as a share of it; it is most of the "
        "100 - 'PC1-3 cum %' left over, so a calm period shows a lower PC1-3 share for that reason alone.",
        "",
        "## 2. Loadings, full sample",
        "",
        "![Full-sample loadings](loadings_full_sample.png)",
        "",
        md_table(loadings_table(fl, "Full sample")),
        "",
        "PC1 is small at 0.25y, which fits a money-market end tied to the cash rate between decisions. PC3 "
        "moves the money-market end against the belly, so it is not the textbook 2-5-10 butterfly.",
        "",
        "## 3. Loadings by period and stability",
        "",
        "![Loadings by period](loadings_by_period.png)",
        "",
        md_table(pd.concat([loadings_table(r["load"], k) for k, r in main.items() if k != "Full sample"])),
        "",
        "Cosines after sign alignment with the full sample ('cos PCk') and with a PCA on all changes outside "
        "the period ('vs rest'). Read 'vs rest': 'cos PCk' is inflated for a period with a large 'weight in "
        f"full %'. For scale, a flat vector has cosine {flat_cos:.2f} with full-sample PC1 and a straight line "
        f"{line_cos:.2f} with PC2, so the 0.9s show the same broad shape, not the same factor in detail.",
        "",
        md_table(stab_txt),
        "",
        "## 4. PC3 separation and the hedge maturities (for M4)",
        "",
        md_table(sep_txt),
        "",
        "A PC3/PC4 ratio near 1 means the two can rotate or swap in a shorter window. 'cond' is the condition "
        f"number of PC1-3 at 2/5/10y, the system M4's H3 hedge solves: {cond['2017-2019']:.0f} in 2017-2019 "
        "means large, offsetting H3 weights. M4 should log, per window, PC3's best match (including PC4), "
        "this ratio and cond.",
        "",
        "## 5. Robustness",
        "",
        "Against the main run in the same period (YCC dates rows: the main YCC period). Correlation-matrix "
        "loadings are put in bp units first (loading x the maturity's sd).",
        "",
        md_table(rob_txt),
        "",
        "- Keep suspect days (only periods that hold them): in 2017-2019 the artefact becomes PC3 (0.25y "
        f"{keep_pc3.loc[0.25]:+.2f}, 0.5y {keep_pc3.loc[0.5]:+.2f}; best match '{keep['best match']}') and "
        f"curvature moves to {keep['main PC3 found as']} ({keep['its cos']:.2f}).",
        f"- Correlation matrix: PC1 explains {'less in every period' if corr_less else 'less in some periods'} "
        f"(YCC {corr.loc[YCC_LABEL, 'PC1 %']:.1f}% vs {var.loc[YCC_LABEL, 'PC1 %']:.1f}%). Under YCC the 0.25y "
        f"change is 0 bp on {int((ycc_short == 0).sum())} of {len(ycc_short)} days and +/-1 bp on "
        f"{int((ycc_short.abs() == 1).sum())}, so standardising scales rounding ticks up to unit variance. In "
        f"bp units PC1 keeps its shape (>= {corr['cos PC1'].min():.3f}); PC2 "
        f"({corr['cos PC2'].min():.2f}-{corr['cos PC2'].max():.2f}) and PC3 "
        f"({corr['cos PC3'].min():.2f}-{corr['cos PC3'].max():.2f}) change, and PC3's identity moves in "
        f"{', '.join(corr_odd) or 'no period'}.",
        "- 0.5-10 years: lowest cosine "
        f"{rob.loc['0.5-10 years'][[f'cos {pc}' for pc in PC_NAMES]].to_numpy().min():.3f}.",
        f"- YCC dates: regime closes {f(y_close[shares3], 1)}% (main {f(var.loc[YCC_LABEL, shares3], 1)}%), "
        f"PC3 cosine {y_close['cos PC3']:.2f}; before the break {f(y_calm[shares3], 1)}%, and the main PC3 is "
        f"found as {y_calm['main PC3 found as']} ({y_calm['its cos']:.2f}): without the transitions PC3 and "
        "PC4 swap.",
        "",
        "## 6. Limitations",
        "",
        "- **Fitted smooth curve.** F17 is fitted, not 40 traded bonds, so neighbouring maturities move "
        "together by construction and 3 factors explain more than they would for bond yields. M4's "
        "out-of-sample hedge test is the check that matters.",
        "- **YCC.** The RBA held the 3-year bond yield and the cash rate near zero, so the factors there "
        "describe a constrained curve; the fitted curve did not follow the target bond closely (M1 section 10).",
        "- **Gaps.** A change after a weekend or holiday covers several calendar days, including the missing "
        "20-21 Jul 2026 (M1 section 4). Not adjusted.",
        f"- **Stationarity.** Total variance differs by a factor of {total.max() / total.min():.1f} between "
        "periods, so a full-sample PCA averages regimes; M4 re-estimates on a trailing window.",
        "- **Suspect days** are a judgement (M1 section 5) made with hindsight (the curve came back the next "
        "day); the robustness rows show what keeping them does.",
        "",
        "## Files",
        "",
        "`outputs/m3/`: this report, `loadings_full_sample.png`, `loadings_by_period.png`, `loadings.csv` "
        "(PC1-3 at all 40 maturities), `variance_explained.csv`, `stability.csv`, `separation.csv`, "
        "`robustness.csv` (CSVs unrounded).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# Main

def main():
    """Run M3 end to end and print a short summary.

    Steps: load F17 yields; daily changes with and without the suspect days;
    PCA by period (main, keep suspect days, correlation matrix, 0.5-10 years,
    YCC regime windows); tables, charts, CSVs (unrounded) and the report.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yields = load_f17("yields")
    changes = daily_changes_bp(yields, MATURITIES, drop_suspect=True)
    changes_all = daily_changes_bp(yields, MATURITIES, drop_suspect=False)
    main_res = run_periods(changes, PERIODS)

    # Robustness variants: each maps its period labels to the main period it
    # is compared with. Keep-suspect only runs the periods that hold a suspect
    # day (elsewhere it is identical to the main run).
    suspect_dates = [pd.Timestamp(k) for k in SUSPECT_DAYS]
    with_suspect = {k: b for k, b in PERIODS.items()
                    if slice_period(changes_all, b).index.isin(suspect_dates).any()}
    same = {k: k for k in PERIODS}
    variants = {
        "Keep suspect days": (run_periods(changes_all, with_suspect), same),
        "Correlation matrix": (run_periods(changes, PERIODS, standardise=True), same),
        "0.5-10 years": (run_periods(changes[MATURITIES_NO_3M], PERIODS), same),
        "YCC dates": (run_periods(changes, YCC_VARIANTS), {k: YCC_LABEL for k in YCC_VARIANTS}),
    }
    tables = {"variance": variance_table(main_res), "stability": stability_table(main_res, changes),
              "separation": separation_table(main_res), "robustness": robustness_table(main_res, variants)}

    # CSVs: loadings at every maturity for every period (long format), and the tables.
    long = pd.concat([r["load"][PC_NAMES].assign(period=k).reset_index() for k, r in main_res.items()])
    long[["period", "maturity_years"] + PC_NAMES].to_csv(OUT_DIR / "loadings.csv", index=False)
    tables["variance"].to_csv(OUT_DIR / "variance_explained.csv", index=False)
    for name in ("stability", "separation", "robustness"):
        tables[name].to_csv(OUT_DIR / f"{name}.csv", index=False)

    plot_loadings(main_res, OUT_DIR)
    write_report(main_res, variants, tables, changes, len(changes_all), REPORT_PATH)

    print(f"M3: {len(changes)} daily changes x {len(MATURITIES)} maturities "
          f"({len(changes_all) - len(changes)} suspect-day changes dropped)")
    print(tables["variance"][["period", "n changes", "PC1-3 cum %"]].round(2).to_string(index=False))
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
