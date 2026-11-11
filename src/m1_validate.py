"""M1 - load and validate the data.

Run from the project folder (01-yield-curve-vs-rba):
    python src/m1_validate.py

What it does:
  1. Loads F17 yields and forwards and the A2 cash rate changes with the
     loaders in src/data.py. Checks 11 and 12 (added in M5) also load F2
     and F1. Those two files are gitignored third-party data: they stay in
     memory, and only statistics and one chart derived from them are written.
  2. Runs every M1 check and writes outputs/m1/validation_report.md: a summary table with one PASS / FLAG / NOT RUN line per check,
     then a section of detail for each check.
  3. Saves figures (PNG) and one audit CSV to outputs/m1/.

Result words:
  PASS    - the check came out as expected.
  FLAG    - something to read before M2. Not necessarily an error: it can be a
            finding about the data that later steps must allow for.
  NOT RUN - an input is missing.

Units: yields and rates in per cent per annum. Differences in basis points
(bp): 1 bp = 0.01 percentage points, so a difference in per cent times 100.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no screen needed
import matplotlib.pyplot as plt
from dateutil.easter import easter  # dateutil is installed with pandas

# Make "import data" work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402  (import after the sys.path line on purpose)
    PROJECT_DIR, RAW_DIR, MANIFEST_PATH, F17_FILES, A2_FILE,
    MATURITY_GRID, detect_encoding, sha256_of_file, load_f17, load_a2,
    target_in_effect, target_known_at_close, is_range_text,
)
# Added in M5 for checks 11 and 12 (F2 and F1 loaders, and the shared
# daily-change helper that M3 and M4 also use).
from data import F1_FILE, F2_FILE, load_f1, load_f2, daily_changes_bp  # noqa: E402


# Settings. Every threshold a check uses is here, so an auditor can see and
# change it in one place.

OUT_DIR = PROJECT_DIR / "outputs" / "m1"
REPORT_PATH = OUT_DIR / "validation_report.md"

BP = 100.0                 # per cent -> basis points
TOL = 0.005                # two 2-decimal numbers count as equal if they differ by < 0.5 bp
EXPECTED_ENCODING = "utf-8-sig"  # F17 CSVs: UTF-8 with a byte order mark
EXPECTED_ROWS = 2448       # F17 rows, 3 Jan 2017 - 30 Sep 2026
EXPECTED_CHANGES = 26      # A2 changes inside the F17 sample
EXPECTED_CUTS = 9
EXPECTED_HIKES = 17
MAX_GAP_DAYS = 4           # report gaps longer than 4 calendar days
VALUE_BOUNDS = (-1.0, 10.0)  # plausible range for AUD yields 2017-2026, per cent
REVERSAL_BP = 15           # check 5: a move of at least this many bp that the next F17 day
                           # reverses by at least this many bp, at the same maturity
STEP_TOL_BP = 2            # check 8: the 0-year point "moves by the change" if within 2 bp of it
ZERO_MATCH_SHARE = 0.99    # check 8: PASS if the 0-year point equals the target on > 99% of days
YCC_BAND_BP = 5            # check 10: "on target" means within 5 bp of the YCC target
M2_MATURITIES = [0.25, 0.5, 1.0]  # forward maturities (years) M2 will read: 3, 6, 12 months
M2_HORIZONS = [3, 6, 12]   # M2 horizons in months
M3_FIRST_PERIOD = ("2017-01-01", "2019-12-31")  # first M3 sub-period
F2_CORR_MIN = 0.95         # check 11: PASS needs daily 10-year changes to correlate at least this much
F2_GAP_MAX_BP = 10         # check 11: ... and the F2 - F17 10-year level gap to stay within +/- 10 bp
                           # (both thresholds are judgement: loose bounds for "same market, different
                           # yield type"; the report prints the actual numbers next to them)

# Standard deviation of the rounding error of a number published to 2
# decimals. The error is uniform on [-0.005, +0.005] per cent, whose standard
# deviation is 0.005 / sqrt(3).
ROUNDING_SD = 0.005 / np.sqrt(3)

# On 29 Dec 2023 the RBA replaced all F17 series with "series calculated
# using the new series" of its table F16 (RBA, Changes to Statistical Tables,
# https://www.rba.gov.au/statistics/tables/changes-to-tables.html). That is a
# change of inputs. The note does not say whether past dates were recalculated.
# The forward check is also run before and after this date.
F17_INPUT_CHANGE = pd.Timestamp("2023-12-29")

# One-day reversals in the F17 yields (check 5). Every day the check finds is
# looked up in these two lists. The notes are judgement, written after
# looking at the whole curve on the day (the reversal table in the report
# shows the move at 0.25, 1, 3 and 10 years so it can be re-checked).
#   KNOWN_MARKET_MOVES - broad moves across many maturities around a known
#                        market or policy event: real data, keep them.
#   SUSPECT_DAYS       - moves that look like a curve-fitting artefact: keep
#                        the data (data/raw/ is never changed) but test later
#                        results with and without them.
# A reversal day in neither list is reported as unexplained (FLAG).
KNOWN_MARKET_MOVES = {
    "2020-03-19": "RBA package (cash rate cut, 3-year yield target, bond buying from 20 Mar); long end up, "
                  "then down as buying began",
    "2021-02-26": "global bond sell-off (YCC target bond tested); long end up about 20 bp, reversed on 1 Mar "
                  "as the RBA bought extra bonds",
    "2021-10-29": "end of yield curve control (target not defended, 28-29 Oct 2021); partly reversed on 1 Nov",
    "2022-05-04": "broad rise of about 10-15 bp the day after the first hike of the cycle (announced 3 May "
                  "2022), reversed the next day",
    "2022-06-15": "mid-June 2022 global bond sell-off; reversed after the US Federal Reserve's 75 bp hike "
                  "(15 Jun US time)",
    "2023-03-16": "March 2023 bank stress (Silicon Valley Bank, Credit Suisse): large swings on consecutive days",
    "2023-03-17": "March 2023 bank stress (Silicon Valley Bank, Credit Suisse): large swings on consecutive days",
    "2025-04-09": "US tariff turmoil; most new US tariffs paused on 9 Apr US time (10 Apr in Sydney)",
}
from data import SUSPECT_DAYS  # noqa: E402  (one shared list in data.py; later milestones use it too)

# Yield curve control (YCC). Source: RBA, Review of the Yield Target,
# https://www.rba.gov.au/monetary-policy/reviews/yield-target/index.html
#   - 19 Mar 2020: target for the 3-year AGS yield of around 0.25 per cent;
#   - 3 Nov 2020: target lowered to around 0.10 per cent;
#   - 2 Nov 2021: target discontinued.
# Timing rule (same as target_known_at_close): a decision announced on day d
# applies to the F17 close on d, because the announcement comes before the
# close (2.30 pm release, 4:30 pm close; sources next to NEXT_DAY_EFFECT_FROM
# in data.py). The 19 Mar 2020 package was released before its 4.00 pm press
# conference (media release 2020-08), so also before the close. So the 0.25
# regime covers closes 19 Mar 2020 to 2 Nov 2020, the 0.10 regime closes
# 3 Nov 2020 to 1 Nov 2021, and there is no target from the close on
# 2 Nov 2021.
# 'rba_bond_avg' is the RBA's own figure for the average yield on the target
# bond in each regime (same review: 0.24 and 0.09 per cent). It is shown next
# to F17 so the reader can see how far the fitted curve sits from the bond.
YCC_REGIMES = [
    {"label": "around 0.25", "target": 0.25, "first_close": "2020-03-19", "last_close": "2020-11-02",
     "rba_bond_avg": 0.24},
    {"label": "around 0.10", "target": 0.10, "first_close": "2020-11-03", "last_close": "2021-11-01",
     "rba_bond_avg": 0.09},
]
# The target applied to one bond, not to a constant 3-year point: the April
# 2023 bond until 20 Oct 2020, then the April 2024 bond (same RBA review).
# Both bonds mature on 21 April of their year (their coupon date).
YCC_BOND_SWITCH = pd.Timestamp("2020-10-21")
YCC_BOND_MATURITY = {"April 2023": pd.Timestamp("2023-04-21"), "April 2024": pd.Timestamp("2024-04-21")}

# Chart colours: the first three categorical slots of the dataviz reference
# palette (blue, orange, aqua), with dark ink for text and a light grid.
COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb",
}


# Small formatting helpers

def fmt_cell(value, digits=2):
    """Format one table cell for markdown.

    Input:  any value; digits = decimals for floats.
    Output: text. Missing values (NaN, NaT, None, pd.NA) become an empty
            cell, dates become YYYY-MM-DD, floats get a fixed number of
            decimals.
    Note:   pd.NaT is tested first because it also counts as a date
            (isinstance(pd.NaT, date) is True) but has no strftime.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (pd.Timestamp, date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def md_table(frame, digits=2):
    """Render a DataFrame as a markdown table (no extra package needed).

    Input:  frame - the table, its columns become the header;
            digits - decimals for float cells.
    Output: the markdown text. The index is not printed, so put anything you
            want shown into a column first.
    """
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt_cell(v, digits) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def result(name, status, line, body):
    """Bundle one check's outcome for the report.

    name   - short check name (used in the summary table and as a heading);
    status - 'PASS', 'FLAG' or 'NOT RUN';
    line   - one line of interpretation for the summary table;
    body   - markdown detail for the check's own section;
    """
    return {"name": name, "status": status, "line": line, "body": body}


def explain_day(day, changes):
    """Say why the F17 curve might move sharply on 'day'.

    Inputs: day     - a date (Timestamp);
            changes - the output of load_a2().
    Output: text, the first that applies:
              the note in KNOWN_MARKET_MOVES;
              'SUSPECT - ' + the note in SUSPECT_DAYS;
              'cash rate change of +50 bp takes effect' if a target change
              takes effect that day (A2 effective dates);
              'NONE - not explained'.
    The hand-written notes come first: a day can be both an effective date
    and a day with a separate market move further out the curve (4 May 2022).
    Used by check 5 for the largest daily moves and the one-day reversals.
    """
    key = f"{day:%Y-%m-%d}"
    if key in KNOWN_MARKET_MOVES:
        return KNOWN_MARKET_MOVES[key]
    if key in SUSPECT_DAYS:
        return "SUSPECT - " + SUSPECT_DAYS[key]
    hit = changes[changes["effective_date"] == day]
    if len(hit) and hit["change_pp"].iloc[0] != 0:
        return f"cash rate change of {hit['change_pp'].iloc[0] * BP:+.0f} bp takes effect"
    return "NONE - not explained"


def style_axes(ax, title, ylabel):
    """Apply one quiet style to a chart: left title, light y grid, no box."""
    ax.set_title(title, loc="left", fontsize=11, color=COLORS["ink"])
    ax.set_ylabel(ylabel, color=COLORS["muted"])
    ax.set_facecolor(COLORS["surface"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)


# Curve maths used by the forward-rate check
#
# Notation: m is maturity in years, y(m) the zero-coupon yield and P(m) the
# discount factor. Under continuous compounding P(m) = exp(-m y(m) / 100)
# (y in per cent), so m y(m) = -100 ln P(m). Then:
#   - the instantaneous forward is f(m) = d/dm [m y(m)] = y(m) + m y'(m);
#   - the period forward from a to b is F(a, b) = (b y(b) - a y(a)) / (b - a);
#   - y(m) is the average of instantaneous forwards over [0, m]:
#       y(m) = (1/m) * integral from 0 to m of f(s) ds.
# The F17 grid has step h = 0.25 years, so a "3-month period forward" is one
# grid step.

def grid_step(frame):
    """Return (maturities as a numpy array, step h); stop if uneven."""
    m = np.asarray(frame.columns, dtype=float)
    steps = np.diff(m)
    if not np.allclose(steps, steps[0]):
        raise ValueError("The maturity grid must be evenly spaced")
    return m, float(steps[0])


def instantaneous_forwards(yields):
    """Instantaneous forwards implied by zero yields (continuous compounding).

    Input:  yields - DataFrame, dates x maturities (per cent), evenly spaced.
    Output: DataFrame of the same shape:
              column 0     - y(0): the limit of d/dm [m y(m)] at m = 0;
              interior     - central difference of g(m) = m y(m):
                             (g(m + h) - g(m - h)) / (2h);
              last column  - NaN. A central difference needs a point beyond
                             the end of the grid. A one-sided difference there
                             has about 3.6 times the rounding error of the
                             central difference next to it, so it is left out
                             rather than compared unfairly.
    """
    m, h = grid_step(yields)
    y = yields.to_numpy(dtype=float)
    g = y * m
    f = np.full_like(y, np.nan)
    f[:, 0] = y[:, 0]
    f[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / (2 * h)
    return pd.DataFrame(f, index=yields.index, columns=yields.columns)


def period_forwards(yields, position):
    """One-step (3-month) period forwards implied by zero yields.

    Inputs: yields   - DataFrame, dates x maturities (per cent).
            position - 'ending': the value in column m is F(m - h, m),
                                 NaN at m = 0;
                       'starting': the value in column m is F(m, m + h),
                                 NaN at the last maturity.
    Output: DataFrame of the same shape (per cent, continuous compounding).
    """
    m, h = grid_step(yields)
    y = yields.to_numpy(dtype=float)
    g = y * m
    step = (g[:, 1:] - g[:, :-1]) / h  # forward over [m_k, m_k + h]
    f = np.full_like(y, np.nan)
    if position == "ending":
        f[:, 1:] = step
    elif position == "starting":
        f[:, :-1] = step
    else:
        raise ValueError("position must be 'ending' or 'starting'")
    return pd.DataFrame(f, index=yields.index, columns=yields.columns)


def average_forward(forwards, rule):
    """Average of forwards over [0, m] for every maturity m on the grid.

    Inputs: forwards - DataFrame, dates x maturities (per cent).
            rule     - how the forward in column m is read:
                'trapezoid' - an instantaneous forward at the point m; the
                              integral is the trapezoid rule;
                'left'      - a period forward covering [m, m + h]
                              ("starting at m"): left Riemann sum;
                'right'     - a period forward covering [m - h, m]
                              ("ending at m"): right Riemann sum.
    Output: DataFrame of the same shape; column 0 is NaN (0 / 0).
    Use:    under continuous compounding this average should equal y(m). If
            the forward file holds the kind of forward the rule assumes, the
            match is exact up to rounding (and, for the trapezoid, up to the
            curvature of the forward curve inside each 3-month step).
    """
    m, h = grid_step(forwards)
    f = forwards.to_numpy(dtype=float)
    if rule == "trapezoid":
        pieces = (f[:, 1:] + f[:, :-1]) / 2 * h
    elif rule == "left":
        pieces = f[:, :-1] * h
    elif rule == "right":
        pieces = f[:, 1:] * h
    else:
        raise ValueError("rule must be 'trapezoid', 'left' or 'right'")
    out = np.full_like(f, np.nan)
    out[:, 1:] = np.cumsum(pieces, axis=1) / m[1:]
    return pd.DataFrame(out, index=forwards.index, columns=forwards.columns)


def diff_stats(estimate, actual, maturities, rows=None):
    """Summary of estimate - actual, in bp, over the chosen maturities.

    Inputs: estimate, actual - DataFrames with the same index and columns
                               (per cent);
            maturities       - list of columns to include;
            rows             - optional boolean mask of dates to include.
    Output: dict with MAD (mean absolute difference), mean (signed: a
            positive mean means the estimate is above the actual), the 95th
            percentile and maximum of the absolute difference, and the number
            of cells compared. NaN cells are skipped.
    """
    diff = (estimate[maturities] - actual[maturities]) * BP
    if rows is not None:
        diff = diff[rows]
    values = diff.to_numpy().ravel()
    values = values[~np.isnan(values)]
    return {
        "MAD (bp)": np.mean(np.abs(values)),
        "mean (bp)": np.mean(values),
        "95th pct abs (bp)": np.percentile(np.abs(values), 95),
        "max abs (bp)": np.max(np.abs(values)),
        "cells": len(values),
    }


def rounding_floor_mad(weights_sq_sum):
    """MAD (bp) that 2-decimal rounding alone would produce.

    Input:  weights_sq_sum - sum of squared weights of the rounded yields in
            an estimate (for example (g(m+h) - g(m-h)) / 2h puts weights
            (m+h)/2h and -(m-h)/2h on y(m+h) and y(m-h)).
    Output: expected mean absolute difference in bp between the estimate and
            an F17 forward that is itself rounded to 2 decimals, if the
            estimate's formula were the true definition. Uses the normal
            approximation: MAD = sd * sqrt(2 / pi). The rounding errors are
            uniform, not normal; a simulation with uniform errors gives a
            floor about 2-4% higher, which does not change any ranking in
            check 9.
    Why:    the derivative check multiplies rounding error by about m/h, so a
            sizeable MAD at long maturities is expected even when the formula
            is right. This floor says how big.
    """
    sd = np.sqrt(weights_sq_sum * ROUNDING_SD ** 2 + ROUNDING_SD ** 2)
    return sd * np.sqrt(2 / np.pi) * BP


# Check 1 - files and checksums

def check_checksums():
    """Compare every file in data/MANIFEST.csv with the copy in data/raw/.

    For each file on disk: SHA-256 of the raw bytes and, for CSV files, of
    the bytes after CRLF -> LF (git on Windows adds CR to text files on
    checkout; see sha256_of_file). A file passes if either hash equals the
    manifest. For CSVs the table also shows the extra bytes on disk and the
    number of CRLF line endings: if git added one CR per line, the two are
    equal. Files listed but not on disk are reported as missing. F1 and F2
    are gitignored, so a fresh clone lacks them until they are re-downloaded
   ; since 20 Oct 2026 they are on this disk.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    rows, matched, missing, lf_only, cr_explained = [], 0, [], 0, True
    for item in manifest.itertuples(index=False):
        path = RAW_DIR / item.file
        if not path.exists():
            missing.append(item.file)
            rows.append({"file": item.file, "on disk": "no", "manifest bytes": item.bytes,
                         "disk bytes": "", "extra bytes": "", "CRLF count": "", "raw hash": "",
                         "LF-normalised hash": "", "result": "missing"})
            continue
        raw = sha256_of_file(path)
        is_text = path.suffix.lower() == ".csv"
        lf = sha256_of_file(path, normalise_crlf=True) if is_text else raw
        raw_ok, lf_ok = raw == item.sha256, lf == item.sha256
        matched += int(raw_ok or lf_ok)
        lf_only += int((not raw_ok) and lf_ok)  # matches only after CRLF -> LF
        extra = path.stat().st_size - item.bytes
        crlf = path.read_bytes().count(b"\r\n") if is_text else ""
        if is_text and extra != crlf:
            cr_explained = False
        rows.append({
            "file": item.file, "on disk": "yes", "manifest bytes": item.bytes,
            "disk bytes": path.stat().st_size, "extra bytes": extra, "CRLF count": crlf,
            "raw hash": "match" if raw_ok else "differs",
            "LF-normalised hash": ("match" if lf_ok else "differs") if is_text else "n/a (binary)",
            "result": "OK" if (raw_ok or lf_ok) else "MISMATCH",
        })
    table = pd.DataFrame(rows)
    on_disk = len(manifest) - len(missing)
    status = "PASS" if on_disk > 0 and matched == on_disk else "FLAG"
    line = (f"{matched} of {on_disk} files on disk match MANIFEST.csv ({lf_only} only after CRLF -> LF). "
            f"Not on disk: {', '.join(missing) or 'none'}.")
    cr_text = ("For every CSV on disk the extra bytes equal the number of CRLF line endings, so git "
               "(core.autocrlf=true) added exactly one CR per line on checkout. "
               if cr_explained else
               "For at least one CSV the extra bytes do NOT equal the number of CRLF line endings; "
               "look at that file before trusting it. ")
    body = (
        md_table(table) + "\n\n" + cr_text +
        "Hashing with CRLF turned back into LF reproduces the manifest checksum, so the content is the "
        "file recorded in MANIFEST.csv (retrieved 9 Oct 2026; F1 and F2 on 20 Oct 2026, see 'retrieved_utc')."
    )
    return result("Checksums vs MANIFEST.csv", status, line, body)


# Check 2 - F17 file format (encoding and header block)

def check_f17_format():
    """Report the encoding and header block of the two F17 files.

    detect_encoding reads the first bytes ('utf-8-sig' = UTF-8 with a byte
    order mark). PASS if both files have EXPECTED_ENCODING. The loader works with any of
    the encodings detect_encoding returns, so a FLAG here only means the RBA changed
    its file format, not that the data were misread.
    The header block is shown by its labels: the first cell of every line up
    to the 'Series ID' line (found by search, not by a fixed line count).
    """
    rows = []
    for kind, name in F17_FILES.items():
        path = RAW_DIR / name
        encoding = detect_encoding(path)
        lines = path.read_text(encoding=encoding).splitlines()
        labels = [ln.split(",")[0] for ln in lines]
        header_lines = labels.index("Series ID") + 1  # the first 'Series ID' line ends the header
        rows.append({"file": name, "encoding found": encoding,
                     "header lines (incl. blanks)": header_lines,
                     "labels": "; ".join(lb for lb in labels[1:header_lines] if lb)})
    table = pd.DataFrame(rows)
    found = sorted({r["encoding found"] for r in rows})
    ok = found == [EXPECTED_ENCODING]
    status = "PASS" if ok else "FLAG"
    if ok:
        line = f"Both files are {EXPECTED_ENCODING}, as expected; header block found by its labels."
    else:
        line = (f"Encoding found: {', '.join(found)}; expected {EXPECTED_ENCODING}. Parsing is "
                "unaffected (the loader detects the encoding).")
    body = (
        md_table(table) + "\n\n"
        "'utf-8-sig' means UTF-8 that starts with a byte order mark. The 'labels' column is read from "
        "the file: the first cell of each header line, blanks skipped. The loader finds the 'Title' and "
        "'Series ID' rows by these labels, and reads each maturity twice (from the title, e.g. "
        "'0.25 yrs', and from the series ID, e.g. FZCY25D) and requires both to agree."
    )
    return result("F17 encoding and header", status, line, body)


# Check 3 - F17 shape, maturity grid and dates

def check_f17_structure(yields, forwards):
    """Row counts, maturity grid and date order for both F17 files.

    PASS needs: EXPECTED_ROWS rows in each file, the 41-point grid, dates
    strictly increasing (so also no duplicates), no weekend dates, and the
    same dates in both files.
    """
    rows, ok = [], True
    for kind, frame in (("yields", yields), ("forwards", forwards)):
        idx = frame.index
        increasing = bool(idx.is_monotonic_increasing and idx.is_unique)
        weekends = int((idx.dayofweek >= 5).sum())
        grid_ok = list(frame.columns) == MATURITY_GRID
        rows.append({"file": kind, "rows": len(frame), "columns": frame.shape[1],
                     "maturities": f"{frame.columns[0]:g} to {frame.columns[-1]:g} step 0.25" if grid_ok else "UNEXPECTED",
                     "first date": idx[0], "last date": idx[-1],
                     "strictly increasing": increasing, "weekend dates": weekends})
        ok &= len(frame) == EXPECTED_ROWS and grid_ok and increasing and weekends == 0
    same_dates = yields.index.equals(forwards.index)
    ok &= same_dates
    status = "PASS" if ok else "FLAG"
    increasing = all(r["strictly increasing"] for r in rows)
    line = (f"{len(yields):,} / {len(forwards):,} rows ({yields.index[0]:%d %b %Y} to {yields.index[-1]:%d %b %Y}), "
            f"{yields.shape[1]} maturities 0-10y; dates strictly increasing: {'yes' if increasing else 'NO'}; "
            f"same dates in both files: {'yes' if same_dates else 'NO'}.")
    body = md_table(pd.DataFrame(rows)) + f"\n\nSame dates in the yields and forwards files: {'yes' if same_dates else 'NO'}."
    return result("F17 rows, grid and dates", status, line, body)


# Check 4 - gaps in the F17 dates

def explain_gap(start, end):
    """Name the holiday that explains a gap between two F17 dates.

    Inputs: start, end - consecutive F17 dates.
    Output: 'Christmas - New Year' if a 25 Dec or 1 Jan lies strictly
            between them, 'Easter' if Easter Sunday does, else '' (no
            obvious reason). Easter dates come from dateutil.
    """
    inside = pd.date_range(start + pd.Timedelta(days=1), end - pd.Timedelta(days=1))
    if any((d.month, d.day) in ((12, 25), (1, 1)) for d in inside):
        return "Christmas - New Year"
    easter_days = {pd.Timestamp(easter(y)) for y in {start.year, end.year}}
    if any(d in easter_days for d in inside):
        return "Easter"
    return ""


def check_gaps(yields):
    """List every gap of more than MAX_GAP_DAYS calendar days between F17 dates.

    A gap of 3 days is a normal weekend and 4 days a long weekend. Longer gaps
    are listed with the holiday that explains them. FLAG if any gap has no
    holiday explanation (it means business days are missing from F17).
    A single missing weekday sits inside a gap of 4 days or less, so the gap
    rule cannot see it. The second table therefore lists every weekday with
    no F17 row, year by year, so each date can be checked against the NSW
    public holiday list. That comparison was done by hand and is stated as
    judgement in the report; the script does not hold a holiday calendar.
    """
    idx = yields.index
    gap_days = pd.Series(idx[1:] - idx[:-1], index=idx[1:]).dt.days
    long_gaps = gap_days[gap_days > MAX_GAP_DAYS]
    rows = []
    for end, days in long_gaps.items():
        start = idx[idx.get_loc(end) - 1]
        reason = explain_gap(start, end)
        missing = pd.bdate_range(start + pd.Timedelta(days=1), end - pd.Timedelta(days=1))
        rows.append({"last date before": start, "next date": end, "calendar days": int(days),
                     "weekdays missing": ", ".join(d.strftime("%a %d %b") for d in missing),
                     "explanation": reason or "NONE - missing data"})
    gaps = pd.DataFrame(rows)
    unexplained = gaps[gaps["explanation"].str.startswith("NONE")]

    weekdays = pd.bdate_range(idx[0], idx[-1])
    missing_days = weekdays.difference(idx)
    per_year = pd.DataFrame({
        "year": sorted(set(weekdays.year)),
    })
    per_year["F17 rows"] = [int((idx.year == y).sum()) for y in per_year["year"]]
    per_year["weekdays with no row"] = [int((missing_days.year == y).sum()) for y in per_year["year"]]
    per_year["dates"] = [", ".join(d.strftime("%a %d %b") for d in missing_days[missing_days.year == y])
                         for y in per_year["year"]]

    status = "FLAG" if len(unexplained) else "PASS"
    gap_text = "; ".join(f"{r['last date before']:%d %b %Y} -> {r['next date']:%d %b %Y} ({r['weekdays missing']})"
                         for _, r in unexplained.iterrows())
    if len(unexplained):
        line = (f"{len(gaps)} gaps over {MAX_GAP_DAYS} days; {len(gaps) - len(unexplained)} are Easter or Christmas. "
                f"Unexplained: {gap_text} - business days missing from F17.")
    else:
        line = f"{len(gaps)} gaps over {MAX_GAP_DAYS} days, all Easter or Christmas - New Year."
    body = (
        f"Gaps longer than {MAX_GAP_DAYS} calendar days:\n\n" + md_table(gaps) + "\n\n"
        f"Weekdays with no F17 row, by year ({len(missing_days)} in all):\n\n"
        + md_table(per_year) + "\n\n"
        "Checked by hand against NSW public holidays (judgement, not computed by this script; re-check if "
        "the data are refreshed): every date in the 'dates' column is a NSW public holiday (including "
        "weekend substitutes such as Mon 28 Jan 2019), the NSW bank holiday on the first Monday of August "
        "(missing in 2017-2024 only; F17 has rows on it in 2025 and 2026), or Thu 22 Sep 2022 (the National "
        "Day of Mourning), except Mon 20 and Tue 21 Jul 2026.\n\n"
        "An unexplained gap means the RBA published no curve for those business days. Later "
        "milestones work on F17 business days, so a daily change across the gap spans several days; "
        "M3 and M4 should treat it as one observation or drop it."
    )
    return result("Gaps in F17 dates", status, line, body)


# Check 5 - blanks and value ranges

def one_day_reversals(frame):
    """Find one-day reversals: a big move that the next F17 day undoes.

    Input:  frame - F17 yields or forwards (dates x maturities, per cent).
    Output: (moves, mask)
              moves - daily changes in bp; row d = value on d minus value on
                      the previous F17 day (first row NaN). Rounded to 6
                      decimals so that 0.15 * 100 counts as 15, not
                      14.999999999.
              mask  - True at (d, m) when |move on d| >= REVERSAL_BP,
                      |move on the next F17 day| >= REVERSAL_BP, and the two
                      moves have opposite signs, at the same maturity m.
    Why: a jump that reverses the next day is what a typo or a curve-fitting
    glitch looks like. Big market days can do it too, so check_values looks
    each day up (explain_day) instead of treating every one as an error.
    """
    moves = (frame.diff() * BP).round(6)
    nxt = moves.shift(-1)  # the move on the next F17 day, on the same row
    mask = (moves.abs() >= REVERSAL_BP) & (nxt.abs() >= REVERSAL_BP) & (np.sign(moves) == -np.sign(nxt))
    return moves, mask


def check_values(yields, forwards, changes):
    """Blanks, value ranges, largest daily moves and one-day reversals.

    Inputs: yields, forwards - the F17 frames; changes - load_a2() output.
    Part 1 (both files): blank cells, min and max and where they occur, the
      number of negative cells and the years they fall in.
    Part 2 (both files): the largest daily move, with explain_day's reason.
      For the forwards it also shows the yield at the same maturity on the
      day before, the day and the day after, to tell a market move (both
      files move) from a glitch in one file.
    Part 3 (yields): every one-day reversal (one_day_reversals) with
      explain_day's reason. Yields are what M3 and M4 use.
    Part 4: for each SUSPECT_DAYS entry, how much it and the move back on
      the next F17 day weigh in M3's first period (share of the sum of
      squared daily changes, and the standard deviation with and without).
    Part 5 (forwards): reversals at the M2 maturities on M2 forecast dates
      (the last F17 day of each month), which would corrupt an M2 forecast.
    PASS needs no blanks, every value inside VALUE_BOUNDS, and every yield
    reversal day explained as a cash rate change or a KNOWN_MARKET_MOVES
    entry. A SUSPECT or unexplained day gives FLAG.
    """
    ranges, largest, move_at_by_kind, values_ok = [], [], {}, True
    for kind, frame in (("yields", yields), ("forwards", forwards)):
        long = frame.stack()  # one row per (date, maturity)
        low_at, high_at = long.idxmin(), long.idxmax()
        abs_moves = (frame.diff().abs() * BP).round(6).stack()
        move_at = abs_moves.idxmax()  # (date, maturity) of the largest absolute move
        move_at_by_kind[kind] = move_at
        blanks = int(frame.isna().sum().sum())
        negative = long[long < 0]
        neg_years = sorted(set(negative.index.get_level_values(0).year))
        ranges.append({"file": kind, "blank cells": blanks,
                       "min (%)": long.min(), "min at": f"{low_at[0]:%Y-%m-%d}, {low_at[1]:g}y",
                       "max (%)": long.max(), "max at": f"{high_at[0]:%Y-%m-%d}, {high_at[1]:g}y",
                       "negative cells": len(negative),
                       "years with negatives": ", ".join(str(y) for y in neg_years) or "none"})
        largest.append({"file": kind, "largest daily move (bp)": abs_moves.max(),
                        "date": move_at[0], "maturity": f"{move_at[1]:g}y",
                        "reason": explain_day(move_at[0], changes)})
        values_ok &= blanks == 0 and VALUE_BOUNDS[0] <= long.min() and long.max() <= VALUE_BOUNDS[1]
    ranges, largest = pd.DataFrame(ranges), pd.DataFrame(largest)

    # Part 2: the forwards' largest move, next to the yields at the same maturity.
    day, mat = move_at_by_kind["forwards"]
    i = forwards.index.get_loc(day)
    fwd_path, yld_path = forwards[mat].iloc[i - 1:i + 2], yields[mat].iloc[i - 1:i + 2]
    same_way = bool((np.sign(fwd_path.diff().iloc[1:]) == np.sign(yld_path.diff().iloc[1:])).all())
    fwd_text = (f"Forwards: the largest move is at {mat:g}y on {day:%d %b %Y}: the forward went "
                f"{' -> '.join(f'{v:.2f}' for v in fwd_path)} and the yield at the same maturity went "
                f"{' -> '.join(f'{v:.2f}' for v in yld_path)} (day before, the day, the day after). "
                + ("Both files move up and back together, so this is a market move, not a typo in one file."
                   if same_way else "The two files do NOT move the same way; look at this day."))

    # Part 3: one-day reversals in the yields.
    moves, mask = one_day_reversals(yields)
    rev_rows = []
    for d in mask.index[mask.any(axis=1)]:
        hit = [c for c in yields.columns if mask.loc[d, c]]
        worst = moves.loc[d, hit].abs().idxmax()
        next_day = yields.index[yields.index.get_loc(d) + 1]
        rev_rows.append({
            "date": d,
            "maturities hit": ", ".join(f"{c:g}" for c in hit) if len(hit) <= 4
                              else f"{min(hit):g} to {max(hit):g} ({len(hit)} points)",
            "largest move (bp)": moves.loc[d, worst], "at": f"{worst:g}y",
            "next day at same maturity (bp)": moves.loc[next_day, worst],
            "move at 0.25 / 1 / 3 / 10y (bp)": " / ".join(f"{moves.loc[d, c]:+.0f}" for c in (0.25, 1.0, 3.0, 10.0)),
            "explanation": explain_day(d, changes),
        })
    reversals = pd.DataFrame(rev_rows)
    problems = reversals[reversals["explanation"].str.startswith(("SUSPECT", "NONE"))] if len(reversals) else reversals

    # Part 4: weight of the suspect days (and the move back) in M3's first period.
    drop = []
    for key in SUSPECT_DAYS:
        d = pd.Timestamp(key)
        if d in yields.index:
            j = yields.index.get_loc(d)
            drop += list(yields.index[j:j + 2])  # the day and the next F17 day
    hit_suspect = sorted({c for d in drop if d in mask.index for c in yields.columns if mask.loc[d, c]})
    first = moves.loc[M3_FIRST_PERIOD[0]:M3_FIRST_PERIOD[1]]
    kept = first[~first.index.isin(drop)]
    impact = pd.DataFrame({
        "maturity": [f"{c:g}y" for c in hit_suspect],
        "share of sum of squared daily changes (%)": [
            (first.loc[first.index.isin(drop), c] ** 2).sum() / (first[c] ** 2).sum() * 100 for c in hit_suspect],
        "sd all days (bp/day)": [first[c].std() for c in hit_suspect],
        "sd without those days (bp/day)": [kept[c].std() for c in hit_suspect],
    })
    drop_text = ", ".join(f"{d:%d %b %Y}" for d in drop) or "none"
    period_text = f"{M3_FIRST_PERIOD[0][:4]}-{M3_FIRST_PERIOD[1][:4]}"

    # Part 5: forward reversals at the M2 maturities on M2 forecast dates.
    _, f_mask = one_day_reversals(forwards)
    month_ends = forwards.index.to_series().groupby(forwards.index.to_period("M")).max()
    m2_hits = [f"{d:%d %b %Y} at {c:g}y" for d in month_ends for c in M2_MATURITIES if f_mask.loc[d, c]]

    status = "PASS" if values_ok and len(problems) == 0 else "FLAG"
    neg_years = ranges.set_index("file")["years with negatives"]
    line = ("" if values_ok else "Blank cells or values outside VALUE_BOUNDS; see detail. ") + (
        f"Blanks: {int(ranges['blank cells'].sum())}. Yields {ranges.loc[0, 'min (%)']:.2f} to "
        f"{ranges.loc[0, 'max (%)']:.2f}%, forwards {ranges.loc[1, 'min (%)']:.2f} to "
        f"{ranges.loc[1, 'max (%)']:.2f}% (negative values: yields in {neg_years['yields']}; forwards in "
        f"{neg_years['forwards']}). {len(reversals)} one-day reversals of {REVERSAL_BP}+ bp "
        "in the yields; ")
    if len(problems):
        # The short reason: the explanation without its 'SUSPECT - ' / 'NONE - ' prefix, up to any ':'.
        line += "not a known market move: " + "; ".join(
            f"{r['date']:%d %b %Y} at {r['maturities hit']}y ({r['explanation'].split(' - ', 1)[1].split(':')[0]})"
            for _, r in problems.iterrows()) + "."
    else:
        line += "all are cash rate changes or known market moves."

    body = (
        md_table(ranges) + "\n\n"
        f"Bounds used: {VALUE_BOUNDS[0]} to {VALUE_BOUNDS[1]} per cent. Negative values are small and sit in "
        "the years when the cash rate was close to zero, so they are plausible.\n\n"
        "Largest daily move in each file (reason from `explain_day`):\n\n" + md_table(largest) + "\n\n"
        + fwd_text + "\n\n"
        f"**One-day reversals in the yields** (a move of at least {REVERSAL_BP} bp at some maturity that the "
        f"next F17 day reverses by at least {REVERSAL_BP} bp at the same maturity; `one_day_reversals`). The "
        "explanations come from `KNOWN_MARKET_MOVES` and `SUSPECT_DAYS` in the script's settings and are "
        "judgement; the 'move at 0.25 / 1 / 3 / 10y' column shows how broad each move was:\n\n"
        + md_table(reversals) + "\n\n"
    )
    if len(impact):
        body += (
            f"Weight of {drop_text} (the suspect day and the move back) in {period_text} daily yield changes:\n\n"
            + md_table(impact) + "\n\n"
            f"**For M3 and M4:** run {period_text} with and without {drop_text}. These "
            f"{len(drop)} days carry {impact.loc[0, 'share of sum of squared daily changes (%)']:.0f}% of the {impact.loc[0, 'maturity']} variance in "
            "that period, so they can move its short-end loadings on their own.\n\n"
        )
    body += (f"Reversals in the forward file at {', '.join(f'{c:g}' for c in M2_MATURITIES)} years on an M2 "
             f"forecast date (last F17 day of a month): {', '.join(m2_hits) or 'none'}.")
    return result("Blanks, ranges and daily moves", status, line, body)


# Check 6 - A2 parsing

def check_a2(changes):
    """How A2 parsed: row count, text cells, and internal consistency.

    Consistency: for consecutive rows where both targets are numbers, the
    change should equal new_target minus the previous new_target. PASS needs
    no inconsistent rows and every text cell to be a range ('x to y').
    Also reports how far the file can see: 'covers_until' (publication date
    plus one day). target_in_effect returns NaN after it.
    """
    text_rows = changes[changes["has_text"]]
    all_ranges = all(is_range_text(t) for t in pd.concat([text_rows["change_text"], text_rows["target_text"]]).dropna())

    previous = changes["new_target"].shift(1)
    implied = (changes["new_target"] - previous).round(6)
    comparable = implied.notna() & changes["change_pp"].notna()
    bad = changes[comparable & ((implied - changes["change_pp"]).abs() > 1e-6)]
    zero_change = changes[changes["change_pp"] == 0]
    zero_text = ", ".join(d.strftime("%d %b %Y") for d in zero_change["effective_date"]) or "none"
    covers_until = changes["covers_until"].iloc[0]

    status = "PASS" if all_ranges and bad.empty else "FLAG"
    line = (f"{len(changes)} rows, {changes['effective_date'].min():%d %b %Y} to "
            f"{changes['effective_date'].max():%d %b %Y}: {len(changes) - len(zero_change)} target changes plus "
            f"{len(zero_change)} zero-change row ({zero_text}). {len(text_rows)} rows hold text "
            f"({'all ranges, ' if all_ranges else 'NOT all ranges, '}"
            f"{text_rows['effective_date'].min():%Y}), kept as missing and flagged. "
            f"Rows where change differs from the difference in targets: {len(bad)}. "
            f"Targets known to {covers_until:%d %b %Y} (publication date + 1 day).")
    shown = text_rows[["effective_date", "change_pp", "new_target", "change_text", "target_text"]]
    body = (
        "Columns were found by their Title text: 'Change in Cash Rate Target' and 'New Cash Rate Target'. "
        "Dates are the A2 dates, which are effective dates (A2 notes: since February 2008 the decision is "
        "announced on the Board day and takes effect the next day).\n\n"
        f"Rows with text instead of a number ({len(text_rows)}). The number columns stay empty; the text is kept:\n\n"
        + md_table(shown) + "\n\n"
        f"Rows where the change does not equal the difference in targets: {len(bad)}.\n\n"
        f"Rows with a zero change: {zero_text} (the target did not move; A2 lists the row because another "
        f"administered rate changed). The file has {len(changes)} rows, of which "
        f"{len(changes) - len(zero_change)} move the target.\n\n"
        f"Coverage: the file's 'Publication date' is {covers_until - pd.Timedelta(days=1):%d %b %Y}, so it knows "
        f"the target up to {covers_until:%d %b %Y} (a decision announced on the publication date takes effect "
        "the next day). `target_in_effect` returns NaN after that date instead of assuming no change, and "
        f"`target_known_at_close` returns NaN on {covers_until:%d %b %Y} itself."
    )
    return result("A2 parsing", status, line, body)


# Check 7 - cash rate changes inside the F17 sample

def changes_in_sample(changes, yields):
    """A2 changes whose effective date falls between the first and last F17 dates."""
    first, last = yields.index[0], yields.index[-1]
    inside = changes[(changes["effective_date"] >= first) & (changes["effective_date"] <= last)]
    return inside.reset_index(drop=True)


def check_changes(changes, yields):
    """Count and list the cash rate changes inside the F17 sample.

    PASS needs EXPECTED_CHANGES changes, EXPECTED_CUTS cuts and
    EXPECTED_HIKES hikes, and every announcement date to be an F17 business
    day (M2 reads the curve on and around those days).
    """
    inside = changes_in_sample(changes, yields)
    cuts = int((inside["change_pp"] < 0).sum())
    hikes = int((inside["change_pp"] > 0).sum())
    announced_on_f17_day = inside["announcement_date"].isin(yields.index)

    table = inside[["announcement_date", "effective_date", "change_pp", "new_target"]].copy()
    table.insert(0, "#", range(1, len(table) + 1))
    table["direction"] = np.where(table["change_pp"] < 0, "cut", "hike")
    table["announced on"] = inside["announcement_date"].dt.strftime("%a").to_numpy()
    table["announcement day in F17"] = announced_on_f17_day.to_numpy()

    ok = (len(inside), cuts, hikes) == (EXPECTED_CHANGES, EXPECTED_CUTS, EXPECTED_HIKES) and announced_on_f17_day.all()
    status = "PASS" if ok else "FLAG"
    by_year = inside.groupby(inside["effective_date"].dt.year)["change_pp"].agg(
        cuts=lambda s: int((s < 0).sum()), hikes=lambda s: int((s > 0).sum()))
    year_text = ", ".join(f"{y}: {r.cuts} cut{'s' if r.cuts != 1 else ''} / {r.hikes} hike{'s' if r.hikes != 1 else ''}"
                          for y, r in by_year.iterrows())
    quiet_years = [str(y) for y in sorted(set(yields.index.year)) if y not in by_year.index]
    weekdays = table["announced on"].value_counts()
    weekday_text = ", ".join(f"{n} on a {day}" for day, n in weekdays.items())
    expected = (len(inside), cuts, hikes) == (EXPECTED_CHANGES, EXPECTED_CUTS, EXPECTED_HIKES)
    line = (f"{len(inside)} changes ({cuts} cuts, {hikes} hikes), "
            f"{'as expected' if expected else f'EXPECTED {EXPECTED_CHANGES} ({EXPECTED_CUTS} cuts, {EXPECTED_HIKES} hikes)'}. "
            f"All announcement days are F17 business days: {'yes' if announced_on_f17_day.all() else 'NO'}.")
    body = (
        md_table(table) + "\n\n"
        f"By year of effect: {year_text}. Years in the F17 sample with no change: "
        f"{', '.join(quiet_years) or 'none'}. Announcements: {weekday_text}.\n\n"
        "The announcement date is derived, not read: effective date minus one calendar day since February "
        "2008 (A2 notes). This follows the RBA rule that the media release comes at 2.30 pm and any change "
        "takes effect the following day (https://www.rba.gov.au/statistics/cash-rate/, whose date column is "
        "headed 'Effective Date'). That every derived announcement day is an F17 business day "
        f"({'yes' if announced_on_f17_day.all() else 'NO'}) only rules out weekend dates. The real test is in "
        "the data, in check 8: the 0-year point should be unchanged on each derived announcement day and "
        "step on the effective day."
    )
    return result("Cash rate changes in sample", status, line, body)


# Check 8 - the F17 0-year point against the cash rate target

def check_zero_year(yields, changes):
    """Compare F17's 0-year yield with the A2 cash rate target.

    Two alignments:
      - target_in_effect(d): the target that applies on day d;
      - target_known_at_close(d): the target in effect on d + 1, which
        includes a decision announced on d.
    A day "matches" if the 0-year point is within TOL of the target. Days
    with no known target (target_known_at_close is NaN on the last day A2
    covers) are left out of that alignment's match rate. Also:
      - the spread (0-year minus target in effect) by year;
      - the main off-target run: the longest stretch of consecutive F17 days
        on which the 0-year point is off the target, plus any stray
        off-target days outside it;
      - for each of the 26 decisions, the daily change in the 0-year point on
        the announcement day and on the effective day. If the 0-year point
        tracks target_in_effect it moves on the effective day, by the size of
        the change (within STEP_TOL_BP);
      - notes for M2: the month-end spread by year (the bias it adds to the
        M2 error), and month-end forecast dates where "t + h months" and
        "t + h/12 years" fall on opposite sides of a decision.
    Status: PASS only if the 0-year point equals the target in effect on
    more than ZERO_MATCH_SHARE of days. Otherwise FLAG, with the pattern
    explained.
    Returns (result, daily table) - the table is saved for audit and plotted.
    """
    zero = yields[0.0]
    in_effect = target_in_effect(zero.index, changes)
    known = target_known_at_close(zero.index, changes)
    daily = pd.DataFrame({"date": zero.index, "f17_0y": zero.to_numpy(),
                          "target_in_effect": in_effect.to_numpy(),
                          "target_known_at_close": known.to_numpy()})
    daily["spread_vs_in_effect_bp"] = ((daily["f17_0y"] - daily["target_in_effect"]) * BP).round(2)
    daily["matches_in_effect"] = daily["spread_vs_in_effect_bp"].abs() < TOL * BP
    # Nullable boolean: missing (not False) where the target known at close is
    # unknown, so those days drop out of the means below.
    known_ok = daily["target_known_at_close"].notna()
    daily["matches_known_at_close"] = ((daily["f17_0y"] - daily["target_known_at_close"]).abs() < TOL).astype(
        "boolean").where(known_ok, pd.NA)
    unknown_days = daily.loc[~known_ok, "date"]

    rate_in = daily["matches_in_effect"].mean()
    rate_known = daily["matches_known_at_close"].mean()

    by_year = daily.groupby(daily["date"].dt.year).agg(
        days=("f17_0y", "size"),
        match_in_effect=("matches_in_effect", "mean"),
        match_known_at_close=("matches_known_at_close", "mean"),
        mean_spread_bp=("spread_vs_in_effect_bp", "mean"),
        min_spread_bp=("spread_vs_in_effect_bp", "min"),
        max_spread_bp=("spread_vs_in_effect_bp", "max"),
    ).reset_index().rename(columns={"date": "year"})
    by_year["match_in_effect"] = by_year["match_in_effect"] * 100
    by_year["match_known_at_close"] = by_year["match_known_at_close"] * 100
    by_year = by_year.rename(columns={
        "match_in_effect": "% match in effect", "match_known_at_close": "% match known at close",
        "mean_spread_bp": "mean spread (bp)", "min_spread_bp": "min spread (bp)", "max_spread_bp": "max spread (bp)"})

    # The main off-target run: number the runs of consecutive F17 days with
    # the same match / no-match state, keep the off-target runs, take the
    # longest. Off-target days outside it are listed as stray days.
    off_flag = ~daily["matches_in_effect"]
    run_id = (off_flag != off_flag.shift()).cumsum()
    off = daily[off_flag]
    if len(off):
        runs = off.groupby(run_id[off_flag])["date"].agg(["min", "max", "size"])
        main_run = runs.loc[runs["size"].idxmax()]
        first_off, last_off, run_days = main_run["min"], main_run["max"], int(main_run["size"])
        in_run = (daily["date"] >= first_off) & (daily["date"] <= last_off)
        stray = daily[off_flag & ~in_run]
        run_spread = daily.loc[in_run, "spread_vs_in_effect_bp"]
        run_low, run_high = run_spread.min(), run_spread.max()
        stray_text = "; ".join(f"{r['date']:%d %b %Y} ({r['spread_vs_in_effect_bp']:+.0f} bp)"
                               for _, r in stray.iterrows()) or "none"
        run_text = (f"off the target on every F17 day from {first_off:%d %b %Y} to {last_off:%d %b %Y} "
                    f"({run_days} days, {run_high:+.0f} to {run_low:+.0f} bp); on target outside that run "
                    f"except: {stray_text}")
    else:
        run_text = "on the target on every F17 day"
    off_above = int((off["spread_vs_in_effect_bp"] > 0).sum())

    # Around each decision: daily change of the 0-year point (bp).
    zero_change = zero.diff() * BP
    inside = changes_in_sample(changes, yields)
    rows = []
    for item in inside.itertuples(index=False):
        ann, eff = item.announcement_date, item.effective_date
        rows.append({
            "announced": ann, "effective": eff, "change (bp)": item.change_pp * BP,
            "0y on announcement day": zero.get(ann, np.nan),
            "in effect that day": target_in_effect([ann], changes).iloc[0],
            "known at close that day": target_known_at_close([ann], changes).iloc[0],
            "0y change on announcement day (bp)": zero_change.get(ann, np.nan),
            "0y on effective day": zero.get(eff, np.nan),
            "0y change on effective day (bp)": zero_change.get(eff, np.nan),
            "new target": item.new_target,
        })
    around = pd.DataFrame(rows)
    moved_on_effective = (around["0y change on effective day (bp)"] - around["change (bp)"]).abs() <= STEP_TOL_BP
    still_on_announcement = around["0y change on announcement day (bp)"].abs() <= STEP_TOL_BP
    exceptions = around[~moved_on_effective]
    # For each exception, the 0-year point's gap to the target before (on the
    # announcement day, against the old target) and after (on the effective
    # day, against the new target). A step smaller than the change means the
    # gap narrowed.
    exception_text = "; ".join(
        f"{r['effective']:%d %b %Y}: moved {r['0y change on effective day (bp)']:+.0f} bp for a "
        f"{r['change (bp)']:+.0f} bp change, because its gap to the target went from "
        f"{(r['0y on announcement day'] - r['in effect that day']) * BP:+.0f} to "
        f"{(r['0y on effective day'] - r['new target']) * BP:+.0f} bp"
        for _, r in exceptions.iterrows()) or "none"

    # Notes for M2 (1): the 0-year spread on M2 forecast dates (last F17 day
    # of each month), by year. The M2 error is outcome - forecast = target -
    # forward, so a curve that starts below the target adds minus this spread
    # to the mean error, even with perfect foresight.
    month_end = daily.groupby([daily["date"].dt.year, daily["date"].dt.month]).tail(1)
    me_spread = month_end.groupby(month_end["date"].dt.year)["spread_vs_in_effect_bp"].mean()
    bias_text = ", ".join(f"{-v:+.0f} bp in {y}" for y, v in me_spread.items() if abs(v) >= 0.5) or "none"

    # Notes for M2 (2): the forward at maturity h/12 years is for the night
    # t + 365.25 * h / 12 days; the outcome date used is t + h calendar
    # months. List forecast dates where the two dates have different targets.
    last_day = yields.index[-1]
    knife = []
    for t in month_end["date"]:
        for h in M2_HORIZONS:
            by_months = t + pd.DateOffset(months=h)
            by_years = t + pd.Timedelta(days=round(365.25 * h / 12))
            if max(by_months, by_years) > last_day:
                continue
            a, b = target_in_effect([by_months, by_years], changes)
            if not np.isclose(a, b, equal_nan=True):  # equal_nan: two unknown targets are not a split
                knife.append(f"t = {t:%d %b %Y}, h = {h}: {by_months:%d %b %Y} -> {a:.2f}, "
                             f"{by_years:%d %b %Y} -> {b:.2f}")
    knife_text = "; ".join(knife) or "none"

    status = "PASS" if rate_in > ZERO_MATCH_SHARE else "FLAG"
    line = (f"0-year = target in effect on {rate_in:.1%} of days ({rate_known:.1%} vs known at close). It behaves "
            f"as the actual (interbank overnight) cash rate, which is how the RBA's method defines the 1-day "
            f"yield (RDP 2008-09, App. A); check 12 tests this against F1. It is {run_text}. Steps on "
            f"the effective day for {int(moved_on_effective.sum())} of {len(around)} decisions.")
    body = (
        f"Days where the 0-year point equals the target (within {TOL * BP:.1f} bp):\n\n"
        + md_table(pd.DataFrame({
            "alignment": ["target_in_effect(d)", "target_known_at_close(d)"],
            "% of days matching": [rate_in * 100, rate_known * 100],
            "days not matching": [int(off_flag.sum()), int((~daily["matches_known_at_close"]).sum())],
            "days with no known target": [int(daily["target_in_effect"].isna().sum()), len(unknown_days)],
        }), digits=1) + "\n\n"
        + (f"No known target at close on {', '.join(f'{d:%d %b %Y}' for d in unknown_days)}: A2 cannot see a "
           "decision announced on its last covered day, so that day is left out of the second row.\n\n"
           if len(unknown_days) else "")
        + "By year (spread = 0-year minus target in effect, bp):\n\n" + md_table(by_year, digits=1) + "\n\n"
        f"The 0-year point is {run_text}. It is above the target on {off_above} days. "
        "This is what the RBA's method predicts: RBA RDP 2008-09, Appendix A "
        "(https://www.rba.gov.au/publications/rdp/2008/2008-09/appendix-a.html) fits the curve so that "
        "\"the 1-day yield is given by the overnight cash rate\", and Finlay and Olivan 2012 (see check 9) "
        "say the curves 'start at the prevailing cash rate'. The actual interbank cash rate traded below "
        "the target from the March 2020 package until 2025. Check 12 compares the values with F1's interbank "
        "overnight cash rate (the actual cash rate).\n\n"
        "Around each decision in the sample (rates in per cent, changes in bp):\n\n" + md_table(around) + "\n\n"
        f"On the announcement day the 0-year point is unchanged (within {STEP_TOL_BP} bp) for "
        f"{int(still_on_announcement.sum())} of {len(around)} decisions; on the effective day it moves by the "
        f"size of the change (within {STEP_TOL_BP} bp) for {int(moved_on_effective.sum())} of {len(around)}. "
        f"Exceptions: {exception_text}. So the 0-year point lines up with target_in_effect, never with "
        "target_known_at_close: every announcement day is a mismatch against target_known_at_close by "
        "construction, because the 0-year point is an overnight rate for that day. This also confirms, in the "
        "data, the announcement dates derived in check 7.\n\n"
        "Full daily table: `outputs/m1/zero_year_vs_target.csv`. Figure: `zero_year_spread.png`.\n\n"
        "**For M2:**\n\n"
        "- Use the A2 target (target_in_effect) as the outcome, never the F17 0-year point.\n"
        "- F17 forwards are forwards of the actual cash rate, so while it sat below the target they sit below "
        "the target path even with perfect foresight. With error = outcome - forecast this adds a positive "
        f"mean error; on M2 forecast dates (last F17 day of each month) the average is {bias_text}. It works "
        "against the term premium, which pushes the error negative. A forecast that uses only date-t data "
        "removes it: forward(t, h) - (0-year(t) - target_in_effect(t)). Whether to report it next to the raw "
        "forward is an M2 decision.\n"
        "- Outcome date: the forward at h/12 years is for the night t + 365.25 h / 12 days, but the "
        "outcome used is target_in_effect(t + h calendar months). The two dates fall on opposite sides of a "
        f"decision for: {knife_text}. M2 should pick one rule and show these cases both ways."
    )
    return result("F17 0-year point vs cash rate target", status, line, body), daily


def plot_zero_year(daily, path):
    """Line chart of the 0-year point minus the target in effect, in bp."""
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=COLORS["surface"])
    ax.step(daily["date"], daily["spread_vs_in_effect_bp"], where="post", color=COLORS["blue"], linewidth=1.5)
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
    style_axes(ax, "F17 0-year yield minus cash rate target in effect", "bp")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Check 9 - what the F17 forward rates are

def check_forwards(yields, forwards, changes):
    """Decide which kind of forward rate F17 publishes, and check compounding.

    Inputs: the two F17 frames, and load_a2() output (only used to name the
    day of the largest Part B error, via explain_day).

    Part A - derivative check. Build three candidates from the yields and
    compare each with the F17 forward file on maturities 0.25 to 9.75 (where
    all three exist):
      (a) instantaneous forward, f(m) = d/dm [m y(m)], central differences;
      (b) 3-month period forward ending at m, F(m - 0.25, m);
      (c) 3-month period forward starting at m, F(m, m + 0.25).
    Differences of rounded yields amplify rounding, so each candidate gets a
    "rounding floor": the MAD rounding alone would give if it were the true
    definition.
    Part B - integral check. Under continuous compounding y(m) equals the
    average of instantaneous forwards over [0, m]. Rebuild the yields from
    the F17 forwards three ways (trapezoid = instantaneous; left sum =
    period forwards starting at m; right sum = period forwards ending at m)
    and compare with F17 yields on maturities 0.25 to 10 (positive mean =
    rebuilt yield above the F17 yield). Averaging shrinks rounding error
    instead of amplifying it, so this is the sharper test. Also split at
    F17_INPUT_CHANGE (29 Dec 2023).
    Part C - compounding. Repeat the trapezoid check assuming annual
    compounding for both files, and a mixed basis (annual yields, continuous
    forwards; compared in annual units, and also shown in continuous units).
    PASS if the instantaneous reading wins both the derivative check (smallest
    absolute mean difference) and the integral check (smallest MAD).
    Returns (result, by-maturity table for the figure).
    """
    m, h = grid_step(yields)
    inst = instantaneous_forwards(yields)
    ending = period_forwards(yields, "ending")
    starting = period_forwards(yields, "starting")
    common = [c for c in yields.columns if 0 < c < 10]  # 0.25 ... 9.75

    # Rounding floors. Weights on the rounded yields for each estimate:
    #   (a) (m+h)/2h and (m-h)/2h, (b) m/h and (m-h)/h, (c) (m+h)/h and m/h.
    mm = np.asarray(common)
    floors = {
        "instantaneous": rounding_floor_mad(((mm + h) ** 2 + (mm - h) ** 2) / (2 * h) ** 2),
        "period ending at m": rounding_floor_mad((mm ** 2 + (mm - h) ** 2) / h ** 2),
        "period starting at m": rounding_floor_mad(((mm + h) ** 2 + mm ** 2) / h ** 2),
    }
    candidates = {"instantaneous": inst, "period ending at m": ending, "period starting at m": starting}
    part_a = []
    for label, est in candidates.items():
        stats = diff_stats(est, forwards, common)
        part_a.append({"candidate built from yields": label, **stats,
                       "rounding floor MAD (bp)": float(np.mean(floors[label]))})
    part_a = pd.DataFrame(part_a)
    part_a["cells"] = part_a["cells"].astype(int)

    # By maturity, for the figure and a short table.
    by_maturity = pd.DataFrame({"maturity": common})
    for label, est in candidates.items():
        by_maturity[label] = [np.nanmean(np.abs(est[c] - forwards[c])) * BP for c in common]
    by_maturity["rounding floor (instantaneous)"] = floors["instantaneous"]
    show = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.75]
    by_maturity_short = by_maturity[by_maturity["maturity"].isin(show)].copy()

    # Part B - integral check.
    all_m = [c for c in yields.columns if c > 0]
    rules = {"trapezoid (instantaneous forwards)": "trapezoid",
             "left sum (period forwards starting at m)": "left",
             "right sum (period forwards ending at m)": "right"}
    rebuilt = {label: average_forward(forwards, rule) for label, rule in rules.items()}
    part_b = pd.DataFrame([{"reading of the F17 forward file": label, **diff_stats(est, yields, all_m)}
                           for label, est in rebuilt.items()])
    part_b["cells"] = part_b["cells"].astype(int)
    b_by_m = pd.DataFrame({"maturity": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]})
    for label, est in rebuilt.items():
        short = label.split(" (")[0]
        b_by_m[short + " MAD (bp)"] = [np.nanmean(np.abs(est[c] - yields[c])) * BP for c in b_by_m["maturity"]]

    # Part B, largest error: where and when it happens (check 5 explains it).
    trap = rebuilt["trapezoid (instantaneous forwards)"]
    trap_err = ((trap[all_m] - yields[all_m]) * BP).abs().stack()
    worst_day, worst_m = trap_err.idxmax()

    # Before and from the 29 Dec 2023 input change.
    before = yields.index < F17_INPUT_CHANGE
    periods = pd.DataFrame([
        {"dates": f"before {F17_INPUT_CHANGE:%d %b %Y}", **diff_stats(trap, yields, all_m, before)},
        {"dates": f"from {F17_INPUT_CHANGE:%d %b %Y}", **diff_stats(trap, yields, all_m, ~before)},
    ])
    periods["cells"] = periods["cells"].astype(int)

    # Part C - compounding. All in per cent; ln(1 + r/100) turns an annual
    # rate into the continuous one, exp(r/100) - 1 turns it back.
    log_fwd = 100 * np.log1p(forwards / 100)
    annual_both = 100 * np.expm1(average_forward(log_fwd, "trapezoid") / 100)
    mixed = 100 * np.expm1(trap / 100)  # rebuilt yield, converted to annual
    part_c = pd.DataFrame([
        {"compounding assumed": "continuous, both files", **diff_stats(trap, yields, all_m)},
        {"compounding assumed": "annual, both files", **diff_stats(annual_both, yields, all_m)},
        {"compounding assumed": "annual yields, continuous forwards", **diff_stats(mixed, yields, all_m)},
    ])
    part_c["cells"] = part_c["cells"].astype(int)
    # The mixed row again in continuous units: rebuilt yield vs the F17 yield
    # (read as annual) converted to continuous.
    mixed_cont = diff_stats(trap, 100 * np.log1p(yields / 100), all_m)

    # Verdict, computed from the numbers. Results are looked up by their row
    # label, not their position, so the order of the tables does not matter.
    a_best = part_a.loc[part_a["mean (bp)"].abs().idxmin(), "candidate built from yields"]
    b_best = part_b.loc[part_b["MAD (bp)"].idxmin(), "reading of the F17 forward file"]
    inst_wins = a_best == "instantaneous" and b_best.startswith("trapezoid")
    f0_equal = bool((forwards[0.0] == yields[0.0]).all())
    mad_b = part_b.set_index("reading of the F17 forward file")["MAD (bp)"]
    mean_a = part_a.set_index("candidate built from yields")["mean (bp)"]
    mad_c = part_c.set_index("compounding assumed")["MAD (bp)"]
    mean_c = part_c.set_index("compounding assumed")["mean (bp)"]
    gap_c = abs(mad_c["continuous, both files"] - mad_c["annual, both files"])
    mad_trap = mad_b["trapezoid (instantaneous forwards)"]
    mad_left = mad_b["left sum (period forwards starting at m)"]
    mad_right = mad_b["right sum (period forwards ending at m)"]
    # Largest gap between continuous and simple (actual/365) overnight rates
    # in the sample: r_simple - r_cont is about r^2 / (2 x 365), r a decimal.
    r_max = float(yields[0.0].max())
    overnight_gap_bp = (r_max / 100) ** 2 / (2 * 365) * 1e4

    status = "PASS" if inst_wins else "FLAG"
    verdict = ("Instantaneous forwards; both files share one compounding basis (continuous per the RBA "
               "definition; the data cannot tell continuous from annual)." if inst_wins
               else f"Forward type NOT settled (derivative check favours {a_best}, integral check {b_best}).")
    line = (f"{verdict} Integral check MAD {mad_trap:.2f} bp vs {mad_left:.2f} / {mad_right:.2f} bp for period "
            f"forwards; derivative check mean {mean_a['instantaneous']:+.2f} bp vs "
            f"{mean_a['period ending at m']:+.2f} / {mean_a['period starting at m']:+.2f} bp.")
    body = (
        "**What the RBA says.**\n\n"
        "- RBA Bulletin, March 2012, Finlay and Olivan, 'Extracting Information from Financial Market "
        "Instruments' (the method behind F17), https://www.rba.gov.au/publications/bulletin/2012/mar/6.html. "
        "It defines the zero-coupon yield from the discount curve as z(t) = -log(d(t)) / t, which is "
        "continuous compounding (the page does not use the words 'continuously compounded'; the log formula "
        "implies it), and the forward curve as f(t) = z(t) + t z'(t), which is the instantaneous "
        "forward. In words, the forward curve is \"the interest rate agreed today for overnight borrowing at "
        "a date in the future\". It also says \"The forward and yield curves start at the prevailing cash "
        "rate.\"\n"
        "- RBA RDP 2008-09, Appendix A, https://www.rba.gov.au/publications/rdp/2008/2008-09/appendix-a.html "
        "(the fitting method): inputs are OIS rates \"for maturities extending up to 1 year\" and bond yields "
        "\"for maturities greater than or equal to 18 months\"; the fit ensures \"the 1-day yield is given by "
        "the overnight cash rate\"; at 1 year the average absolute error against OIS is about 3.5 bp and the "
        "largest 12 bp. Whether today's F17 (built from F16 inputs since Dec 2023) still uses OIS at the short "
        "end is not verified.\n"
        "- F17 table notes (Notes sheet of https://www.rba.gov.au/statistics/tables/xls/f17hist.xlsx): a web "
        "search summary reports that they call the yield and forward curves continuously compounded and say "
        "no adjustment is made for risk premia. That file is not on disk and was not downloaded, so this "
        "wording is not verified here.\n"
        "- RBA, Changes to Statistical Tables, https://www.rba.gov.au/statistics/tables/changes-to-tables.html: "
        "on 29 Dec 2023 \"all series in F17 will be replaced with series calculated using the new series\" "
        "of table F16 (a change of inputs); on 31 Mar 2023 F17 data before 1 Jan 2021 were deleted for "
        "commercial reasons; on 8 Aug 2016 F17 was revised for a data correction and a change in method. "
        "The note does not say whether past dates were recalculated. This file starts in 2017, which "
        "suggests the history was republished from the new inputs; if so, the split at 29 Dec 2023 below "
        "compares date ranges, not methods. Either way F17 is a revised dataset, not what was published at "
        "the time, but each day's curve is fitted to that day's prices only, so no later information leaks "
        "into a forecast made from it.\n\n"
        "**Part A - derivative check** (forwards built from F17 yields vs the F17 forward file, maturities "
        "0.25 to 9.75, every date; positive mean = built forward above F17 forward):\n\n"
        + md_table(part_a) + "\n\n"
        "The yields have 2 decimals, and a derivative multiplies their rounding error by about m / h, so "
        "MADs grow with maturity whatever the definition. The 'rounding floor' is the MAD that rounding alone "
        "would give if the candidate were the true definition (normal approximation; the exact floor for "
        "uniform rounding errors is about 2-4% higher, which changes no ranking). Only the instantaneous "
        "candidate sits near its floor and has a mean near zero; the period forwards are biased by about 2 bp "
        "in opposite directions, which is what a one-step offset on an upward-sloping curve produces.\n\n"
        "MAD by maturity (bp):\n\n" + md_table(by_maturity_short) + "\n\n"
        "At short maturities the instantaneous estimate is a little above its floor because the forward "
        "curve bends sharply near 0 (a priced rate change); a central difference over 6 months smooths that.\n\n"
        "**Part B - integral check** (F17 yields rebuilt as the average of F17 forwards over [0, m], "
        "maturities 0.25 to 10; positive mean = rebuilt yield above the F17 yield):\n\n" + md_table(part_b) + "\n\n"
        "By maturity:\n\n" + md_table(b_by_m) + "\n\n"
        f"Reading the F17 forwards as instantaneous rebuilds the yields to {mad_trap:.2f} bp on average. "
        "The yields alone are rounded to 2 decimals, an error uniform on +/-0.5 bp, whose mean absolute value "
        "is 0.25 bp; that is the floor here. Reading the forwards as 3-month period forwards misses by "
        "several times more at every maturity. The larger error at 0.25 years is mostly the trapezoid rule on "
        f"the bend at the short end. The largest single error ({trap_err.max():.0f} bp) is on "
        f"{worst_day:%d %b %Y} at {worst_m:g}y, a day check 5 explains as: {explain_day(worst_day, changes)}.\n\n"
        f"Trapezoid check before and from the 29 Dec 2023 replacement of F17 with series calculated from new "
        "F16 inputs:\n\n" + md_table(periods) + "\n\n"
        "**Part C - compounding:**\n\n" + md_table(part_c) + "\n\n"
        "Units: the first two rows compare in the units assumed for both files. The mixed row converts the "
        "rebuilt (continuous) yield to annual and compares it with the F17 yield read as annual, so it is in "
        f"annual units; in continuous units the same test gives MAD {mixed_cont['MAD (bp)']:.2f} bp, mean "
        f"{mixed_cont['mean (bp)']:+.2f} bp.\n\n"
        f"Annual yields with continuous forwards miss by {mad_c['annual yields, continuous forwards']:.1f} bp, "
        f"so both files share one compounding basis. Continuous and annual for both files differ by only "
        f"{gap_c:.3f} bp in MAD: the difference is second order (it scales with the squared slope of the "
        "curve), so these two files cannot separate them. The means are small too: "
        f"{mean_c['continuous, both files']:+.2f} bp (continuous) against {mean_c['annual, both files']:+.2f} bp "
        f"(annual), a gap of {abs(mean_c['continuous, both files'] - mean_c['annual, both files']):.2f} bp. "
        "Both are far below the 1 bp rounding step of the published numbers, and the mean also moves with the "
        "integration rule used, so it is not evidence for either basis: the integral check cannot settle "
        "the compounding on its own. Continuous compounding rests "
        "on the RBA's definition z = -log(d) / t above and a discount-factor check on 3 Jan 2017. "
        "The F17 discount-factor file on the RBA site would settle it on every date; it is not on disk.\n\n"
        f"F17 forward at 0 years equals the 0-year yield on every date: {'yes' if f0_equal else 'NO'}.\n\n"
        "**Conclusion.** The F17 forward rate at maturity m is the instantaneous (overnight) forward rate for "
        "a date m years ahead, continuously compounded, from the same curve as the yields. For M2, the forecast "
        "of the cash rate h months ahead is the F17 forward at maturity h/12 years (0.25, 0.5, 1.0) read "
        "directly, with no conversion: the cash rate is a simple overnight rate, and continuous and simple "
        "overnight rates differ by about r^2 / (2 x 365), at most "
        f"{overnight_gap_bp:.3f} bp at the sample's highest 0-year rate ({r_max:.2f}%).\n\n"
        "Reading the short end (RDP 2008-09, App. A, if the method still applies): the 3 and 6-month "
        "forwards come from the OIS-fitted segment, so they are close to OIS-implied forwards of the actual "
        "cash rate, and the M5 'OIS benchmark next to F17' would largely repeat them. The 12-month "
        "forward sits at the end of the OIS inputs, next to the 1-year to 18-month stretch with no inputs, "
        "where the paper reports the largest fitting errors; treat 12-month results as less reliable. "
        "Figure: `forward_check_by_maturity.png`."
    )
    return result("What the F17 forward rates are", status, line, body), by_maturity


def plot_forward_check(by_maturity, path):
    """Line chart: MAD (bp) by maturity for each forward candidate, plus the rounding floor."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=COLORS["surface"])
    m = by_maturity["maturity"]
    series = [("instantaneous", COLORS["blue"]), ("period ending at m", COLORS["orange"]),
              ("period starting at m", COLORS["aqua"])]
    for label, color in series:
        ax.plot(m, by_maturity[label], color=color, linewidth=2, label=label)
    ax.plot(m, by_maturity["rounding floor (instantaneous)"], color=COLORS["muted"], linewidth=1.2,
            linestyle="--", label="rounding floor (instantaneous)")
    ax.set_xlabel("maturity (years)", color=COLORS["muted"])
    ax.set_ylim(bottom=0)
    style_axes(ax, "Forwards built from F17 yields vs F17 forward file: mean absolute difference", "bp")
    ax.legend(frameon=False, fontsize=9, labelcolor=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Check 10 - yield curve control

def ycc_frame(yields):
    """Daily table for the YCC chart and statistics, Jan 2020 to Mar 2022.

    Columns:
      zero_3y      - F17 3-year zero yield (constant maturity);
      target_bond  - name of the bond the target applied to that day;
      bond_years   - that bond's remaining maturity in years (days / 365.25);
      zero_at_bond - F17 zero yield interpolated (linear) at bond_years: the
                     zero-coupon rate for the target bond's own maturity. It is
                     a zero rate, not the bond's coupon-bearing yield, but at
                     these maturities and rates the two are close;
      ycc_target   - the YCC target in force at the close (NaN outside YCC).
    """
    window = yields.loc["2020-01-01":"2022-03-31"]
    table = pd.DataFrame({"zero_3y": window[3.0]})
    table["target_bond"] = np.where(table.index < YCC_BOND_SWITCH, "April 2023", "April 2024")
    maturity = table["target_bond"].map(YCC_BOND_MATURITY)
    table["bond_years"] = (maturity - table.index).dt.days / 365.25
    grid = np.asarray(window.columns, dtype=float)
    table["zero_at_bond"] = [np.interp(t, grid, row) for t, row in zip(table["bond_years"], window.to_numpy())]
    table["ycc_target"] = np.nan
    for regime in YCC_REGIMES:
        inside = (table.index >= regime["first_close"]) & (table.index <= regime["last_close"])
        table.loc[inside, "ycc_target"] = regime["target"]
    return table


def check_ycc(yields):
    """The 3-year yield against the YCC target, per target regime.

    For each regime: days, mean of the 3-year zero, mean / sd / min / max of
    (3-year zero minus target) in bp, and the share of days within
    YCC_BAND_BP; the same for the zero yield at the target bond's maturity;
    and the RBA's own average yield of the target bond for comparison.
    PASS if the median absolute gap at the target bond's maturity is
    YCC_BAND_BP or less in both regimes (F17 shows the target holding on most
    days). This describes the data, it does not test the RBA.
    Returns (result, daily table for the figure).
    """
    table = ycc_frame(yields)
    band = YCC_BAND_BP
    rows, ok = [], True
    for regime in YCC_REGIMES:
        part = table.loc[regime["first_close"]:regime["last_close"]]
        gap3 = (part["zero_3y"] - regime["target"]) * BP
        gapb = (part["zero_at_bond"] - regime["target"]) * BP
        rows.append({
            "target": regime["label"], "first close": pd.Timestamp(regime["first_close"]),
            "last close": pd.Timestamp(regime["last_close"]), "days": len(part),
            "3y zero mean (%)": part["zero_3y"].mean(),
            "3y gap mean (bp)": gap3.mean(), "3y gap sd": gap3.std(), "3y gap min": gap3.min(),
            "3y gap max": gap3.max(), f"3y % days within {band}bp": (gap3.abs() <= band).mean() * 100,
            "F17 at bond maturity mean (%)": part["zero_at_bond"].mean(),
            "RBA: target bond mean (%)": regime["rba_bond_avg"],
            "bond-maturity gap median abs (bp)": gapb.abs().median(),
            f"bond-maturity % within {band}bp": (gapb.abs() <= band).mean() * 100,
        })
        ok &= gapb.abs().median() <= band
    stats = pd.DataFrame(rows)

    # The October 2021 break: largest 3-year gap in the 0.10 regime.
    second = YCC_REGIMES[1]
    late = table.loc[second["first_close"]:second["last_close"]]
    worst_day = ((late["zero_3y"] - second["target"]) * BP).idxmax()
    key_days = table.loc[["2020-03-18", "2020-03-19", "2020-03-20", "2020-11-02", "2020-11-03",
                          "2020-11-04", str(worst_day.date()), "2021-11-01", "2021-11-02"]].reset_index()
    key_days = key_days[["date", "zero_3y", "target_bond", "bond_years", "zero_at_bond", "ycc_target"]]

    # First close on which the target bond (after the switch) had less than
    # 3 years left: from then on the 3-year point lies beyond the bond.
    after_switch = table[table.index >= YCC_BOND_SWITCH]
    under_3y = after_switch.index[after_switch["bond_years"] < 3][0]
    zero_before, zero_on = table.loc["2020-03-18", "zero_3y"], table.loc["2020-03-19", "zero_3y"]

    status = "PASS" if ok else "FLAG"
    s0, s1 = stats.iloc[0], stats.iloc[1]
    within = f"bond-maturity % within {band}bp"
    line = (f"At the target bond's maturity F17 is within {band} bp of the 0.25 target on {s0[within]:.0f}% of "
            f"days and of the 0.10 target on {s1[within]:.0f}% (F17 averages "
            f"{s1['F17 at bond maturity mean (%)']:.2f}% there vs {s1['RBA: target bond mean (%)']:.2f}% for the "
            f"bond itself per the RBA): the fitted curve does not follow the one bond the RBA held down. Break in "
            f"late Oct 2021 (3y {late['zero_3y'].max():.2f}% on {worst_day:%d %b %Y}).")
    body = (
        "Dates and levels are from RBA, Review of the Yield Target, "
        "https://www.rba.gov.au/monetary-policy/reviews/yield-target/index.html: a target of around 0.25 per "
        "cent for the 3-year AGS yield introduced on 19 Mar 2020, lowered to around 0.10 per cent on 3 Nov "
        "2020, discontinued on 2 Nov 2021; the target bond was the April 2023 bond, then the April 2024 bond "
        "from 21 Oct 2020. There was no 0.50 per cent yield target: 0.50 was the cash rate target until "
        "the cut that took effect on 20 Mar 2020.\n\n"
        "Timing: a YCC announcement on day d applies from the F17 close on d (the same rule as "
        "target_known_at_close: 2.30 pm release, 4:30 pm close). The 19 Mar 2020 package was off-cycle; its "
        "media release (2020-08, https://www.rba.gov.au/media-releases/2020/mr-20-08.html) announced a "
        "press conference at 4.00 pm AEDT that day, so it came out before the close, and said bond purchases "
        f"would \"commence tomorrow\". The data agree: the 3-year zero went from {zero_before:.2f} to "
        f"{zero_on:.2f} on the close of 19 Mar 2020, before the cash rate cut took effect on 20 Mar.\n\n"
        + md_table(stats) + "\n\n"
        "Key days (rates in per cent; bond_years = remaining maturity of the target bond):\n\n"
        + md_table(key_days) + "\n\n"
        "How to read this. The 3-year F17 point is a constant-maturity zero rate, but the target was on one "
        "bond whose maturity shortened every day. From the switch to the April 2024 bond "
        f"({YCC_BOND_SWITCH:%d %b %Y}) until {under_3y - pd.Timedelta(days=1):%d %b %Y} the target bond had "
        "more than 3 years left, so the 3-year point sat a little short of it on the curve. From "
        f"{under_3y:%d %b %Y} the bond was shorter than 3 years (about "
        f"{table.loc[second['last_close'], 'bond_years']:.1f} years at the last close under the target), so "
        "the 3-year point sits beyond it and drifts above the target as the curve steepens. (The April 2023 "
        "bond, targeted before the switch, also went below 3 years in April 2020; see bond_years.) The F17 "
        "zero rate at the bond's own "
        "maturity is the fairer comparison. Under the 0.25 target it is within "
        f"{abs(s0['F17 at bond maturity mean (%)'] - s0['RBA: target bond mean (%)']) * BP:.0f} bp of the RBA's "
        "average for the bond. Under the 0.10 target it sits "
        f"{(s1['F17 at bond maturity mean (%)'] - s1['RBA: target bond mean (%)']) * BP:.0f} bp above the bond on "
        "average: F17 is a smooth curve fitted across all bonds, and the RBA's review notes that the spread "
        "between the April 2024 target bond and the next bond widened, so the target bond traded out of line "
        "with the curve. At the end of October 2021 the market tested the target and the RBA did not defend "
        "it: the review reports the April 2024 yield rising from 24 to 77 bp between the 11:15 am bond "
        "purchase announcement time on Thursday 28 Oct and the 4:30 pm close on Friday 29 Oct 2021. Figure: "
        "`ycc_3y_vs_target.png`.\n\n"
        "This FLAG is a finding about F17, not a data error: a fitted curve only partly shows a policy that "
        "pins one bond.\n\n"
        "**For M3 and M4:** in the YCC period the 3-year point was held down by policy, so its daily changes "
        "are small and not driven by the usual factors, so M3 treats Mar 2020 - Nov 2021 as its own "
        "period."
    )
    return result("Yield curve control (YCC)", status, line, body), table


def plot_ycc(table, path):
    """Line chart: F17 3-year zero, zero at the target bond's maturity, and the YCC target."""
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=COLORS["surface"])
    ax.plot(table.index, table["zero_3y"], color=COLORS["blue"], linewidth=1.8, label="F17 3-year zero yield")
    ax.plot(table.index, table["zero_at_bond"], color=COLORS["orange"], linewidth=1.8,
            label="F17 zero yield at target bond maturity")
    ax.step(table.index, table["ycc_target"], where="post", color=COLORS["ink"], linewidth=1.5,
            linestyle="--", label="YCC target")
    start = pd.Timestamp(YCC_REGIMES[0]["first_close"])
    end = pd.Timestamp(YCC_REGIMES[-1]["last_close"])
    ax.axvspan(start, end, color=COLORS["grid"], alpha=0.5, linewidth=0)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Yield curve control: 3-year yield vs target (shaded: target in force)", "per cent")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=COLORS["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Check 11 - F17 10-year zero yield against the F2 10-year bond yield
# (added in M5)
#
# F2 is gitignored (licence unchecked). Everything below keeps F2 values in
# memory; the report gets statistics and the chart gets the gap only.

def f2_10y_comparison(yields, f2):
    """Line up the F17 10-year zero yield and the F2 10-year bond yield.

    Inputs: yields - F17 frame from load_f17('yields');
            f2     - F2 frame from data.load_f2() (columns 2.0, 3.0, 5.0, 10.0).
    Output: (levels, changes)
              levels  - DataFrame on the common dates (F17 dates on which F2
                        has a 10-year value), columns 'f17' and 'f2' (per
                        cent) and 'gap_bp' = (F2 - F17) * 100;
              changes - daily changes in bp, columns 'f17' and 'f2', one row
                        per common date after the first.
    Why common dates first: a daily change must cover the same interval in
    both files. F2 has rows on some days F17 skips (NSW Bank Holidays,
    20-21 Jul 2026), so each change runs from one common date to the next,
    never from each file's own previous row.
    The changes come from data.daily_changes_bp (the M3/M4 helper: value
    minus previous row, times 100). drop_suspect=False: the two
    SUSPECT_DAYS are short-end fitting artefacts (0.25-0.5y), not 10-year
    moves, so all days are kept here.
    """
    f2_10y = f2[10.0].dropna()
    common = yields.index.intersection(f2_10y.index)
    levels = pd.DataFrame({"f17": yields.loc[common, 10.0], "f2": f2_10y.loc[common]})
    levels["gap_bp"] = (levels["f2"] - levels["f17"]) * BP
    changes = daily_changes_bp(levels, ["f17", "f2"], drop_suspect=False)
    return levels, changes


def check_f2(yields):
    """F17 10-year zero yield against the F2 10-year bond yield.

    The two should differ in level: F17 is a zero-coupon yield from a
    fitted curve, F2 the yield of a coupon-paying bond (interpolated by the
    RBA to a fixed 10-year maturity). A coupon bond yield is roughly an
    average of zero yields over the bond's cash flows, so when the curve
    slopes up it sits below the 10-year zero yield. They should move
    together day to day. Reported:
      - the correlation of daily changes on common dates, and the sd of the
        difference between the two changes next to the floor that F17's
        2-decimal rounding alone gives (sqrt(2) x ROUNDING_SD, because a
        change is the difference of two rounded numbers);
      - the level gap F2 - F17: mean, median absolute, sd, min and max with
        dates, and its correlation with the F17 10y - 2y slope (a check of
        the coupon reading above: expected negative);
      - the same by calendar year;
      - F2 dates inside the F17 span that F17 does not have.
    Status: PASS if the correlation is at least F2_CORR_MIN and every gap is
    within F2_GAP_MAX_BP. NOT RUN if F2 is not on disk (gitignored).
    Returns (result, levels or None). 'levels' only feeds the gap chart; it
    holds F2 values, so it is never saved as a file.
    """
    name = "F17 vs F2 10-year cross-check"
    if not (RAW_DIR / F2_FILE).exists():
        line = "F2 is not on disk (gitignored). Re-download from MANIFEST.csv to run it."
        return result(name, "NOT RUN", line, line), None

    f2 = load_f2()
    levels, changes = f2_10y_comparison(yields, f2)
    gap = levels["gap_bp"]
    corr = changes["f17"].corr(changes["f2"])
    change_diff = changes["f2"] - changes["f17"]
    rounding_floor = np.sqrt(2) * ROUNDING_SD * BP  # sd (bp) of the rounding error in one F17 daily change
    slope_bp = (yields.loc[levels.index, 10.0] - yields.loc[levels.index, 2.0]) * BP
    slope_corr = gap.corr(slope_bp)

    # By calendar year: correlation of changes (a change belongs to the year
    # of the date it ends on) and the gap's mean / min / max.
    by_year = pd.DataFrame({
        "year": sorted(set(levels.index.year)),
    })
    by_year["days"] = [int((levels.index.year == y).sum()) for y in by_year["year"]]
    # Formatted here with 3 decimals (md_table would round every float to 2, showing 0.996 as 1.00).
    by_year["corr of daily changes"] = [
        f"{changes.loc[changes.index.year == y, 'f17'].corr(changes.loc[changes.index.year == y, 'f2']):.3f}"
        for y in by_year["year"]]
    gap_year = gap.groupby(gap.index.year).agg(["mean", "min", "max"])
    by_year["mean gap (bp)"] = gap_year["mean"].to_numpy()
    by_year["min gap (bp)"] = gap_year["min"].to_numpy()
    by_year["max gap (bp)"] = gap_year["max"].to_numpy()

    # Calendar coverage. F17 dates without an F2 10-year value drop out of
    # the comparison; F2 dates inside the F17 span with no F17 row are listed
    # (they show the bond market traded on a day F17 skips).
    f17_only = yields.index.difference(levels.index)
    f2_span = f2[10.0].dropna().loc[yields.index[0]:yields.index[-1]].index
    f2_only = f2_span.difference(yields.index)
    f2_only_text = ", ".join(f"{d:%a %d %b %Y}" for d in f2_only) or "none"
    july_note = ""
    if {pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21")} <= set(f2_only):
        july_note = (" F2 has values on Mon 20 and Tue 21 Jul 2026, the two weekdays missing from F17 "
                     "(check 4), so the bond market traded on them: that gap is in F17 only.")

    ok = corr >= F2_CORR_MIN and gap.abs().max() <= F2_GAP_MAX_BP
    status = "PASS" if ok else "FLAG"
    line = (f"Daily 10-year changes correlate {corr:.3f} over {len(changes):,} common-date changes "
            f"({levels.index[0]:%d %b %Y} to {levels.index[-1]:%d %b %Y}). Level gap F2 - F17 averages "
            f"{gap.mean():+.1f} bp (range {gap.min():+.1f} to {gap.max():+.1f} bp) and is more negative when "
            f"the curve is steeper (correlation {slope_corr:+.2f} with the 10y - 2y slope), the coupon effect "
            "expected between a bond yield and a zero yield.")
    body = (
        "F2's 10-year series is the yield on Australian Government bonds interpolated to a 10-year maturity; "
        "F17's is a zero-coupon yield from a fitted curve. They should move together but differ in level. "
        "F2 values are kept in memory only (gitignored, licence unchecked): this section holds statistics and "
        "one chart of the gap.\n\n"
        + md_table(pd.DataFrame({
            "statistic": [
                "common dates", "F17 dates with no F2 10-year value",
                "correlation of daily changes", "sd of F17 daily change (bp)", "sd of F2 daily change (bp)",
                "sd of (F2 change - F17 change) (bp)", "floor from F17 2-decimal rounding alone (bp)",
                "largest abs (F2 change - F17 change) (bp)",
                "gap F2 - F17: mean (bp)", "gap: median abs (bp)", "gap: sd (bp)",
                "gap: min (bp)", "gap: max (bp)", "correlation of gap with F17 10y - 2y slope",
            ],
            "value": [
                f"{len(levels):,}", f"{len(f17_only):,}",
                f"{corr:.4f}", f"{changes['f17'].std():.2f}", f"{changes['f2'].std():.2f}",
                f"{change_diff.std():.2f}", f"{rounding_floor:.2f}",
                f"{change_diff.abs().max():.1f} on {change_diff.abs().idxmax():%d %b %Y}",
                f"{gap.mean():+.2f}", f"{gap.abs().median():.2f}", f"{gap.std():.2f}",
                f"{gap.min():+.1f} on {gap.idxmin():%d %b %Y}", f"{gap.max():+.1f} on {gap.idxmax():%d %b %Y}",
                f"{slope_corr:+.2f}",
            ],
        })) + "\n\n"
        "By year (a change belongs to the year of the date it ends on):\n\n" + md_table(by_year) + "\n\n"
        f"How to read this. The day-to-day correlation is {corr:.3f} and the two changes differ by "
        f"{change_diff.std():.2f} bp (sd), close to the {rounding_floor:.2f} bp that rounding F17 to 2 decimals "
        "would give on its own, so almost all of the day-to-day difference is rounding. The level gap is "
        f"more negative when the curve is steeper (correlation {slope_corr:+.2f} with the 10y - 2y slope), "
        "which is the coupon effect described above; the rest is the difference between a smooth fitted "
        f"curve and the bonds themselves. PASS needs a correlation of at least {F2_CORR_MIN} and every gap "
        f"within +/- {F2_GAP_MAX_BP} bp (judgement thresholds, in the settings at the top of the file).\n\n"
        f"Changes are taken between consecutive common dates (`f2_10y_comparison`). F2 dates inside the F17 "
        f"span that F17 does not have: {f2_only_text}. The August Mondays are NSW Bank Holidays, which F17 "
        f"skips.{july_note}\n\n"
        "Figure: `f2_10y_gap.png`."
    )
    return result(name, status, line, body), levels


def plot_f2_gap(levels, path):
    """Chart of the 10-year level gap, F2 bond yield minus F17 zero yield, in bp, as a MONTHLY
    mean with its monthly min-max band.

    Only monthly statistics of the gap are drawn, never the F2 yield or the daily gap: F17 is
    public, so a daily gap would let anyone read back the F2 series, and F2 is not ours to
    publish.
    """
    month = levels["gap_bp"].resample("ME")
    stats = pd.DataFrame({"mean": month.mean(), "lo": month.min(), "hi": month.max()}).dropna()
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=COLORS["surface"])
    ax.fill_between(stats.index, stats["lo"], stats["hi"], color=COLORS["blue"], alpha=0.2, linewidth=0,
                    label="monthly min-max")
    ax.plot(stats.index, stats["mean"], color=COLORS["blue"], linewidth=1.2, label="monthly mean")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
    style_axes(ax, "10-year: F2 bond yield minus F17 zero-coupon yield", "bp")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Check 12 - F17 0-year point against F1's interbank overnight cash rate
# (added in M5)
#
# Confirms check 8's reading that the 0-year point is the actual cash rate,
# not the target. F1 is gitignored (its OIS columns are FENICS data); only
# the cash rate column is used here, and only statistics are written.

def check_f1_cash_rate(yields, changes):
    """Compare F17's 0-year yield with F1's interbank overnight cash rate.

    Inputs: yields  - F17 frame from load_f17('yields');
            changes - the A2 change list from load_a2() (for the target
                      comparison row).
    Day alignment: F1's 'Interbank Overnight Cash Rate' on day d is the rate
    on overnight loans made on d, and check 8 found the 0-year point is an
    overnight rate for the day of the close. So the main comparison is the
    same day. Two contrast rows show what a wrong reading would give:
      - F1's cash rate on the previous F1 date (as if F17 lagged a day);
      - the A2 target in effect (check 8's comparison).
    A day matches if the two differ by less than TOL (0.5 bp), the same rule
    as check 8. F17 dates missing from F1 count as not matching and are
    reported.
    Also reported: the match rate on the days check 8 found off the target,
    the largest absolute differences (dates and bp), and why the
    previous-date contrast misses (days the cash rate moved).
    Status: PASS if the same-day match rate is above ZERO_MATCH_SHARE.
    NOT RUN if F1 is not on disk (gitignored).
    """
    name = "F17 0-year point vs F1 actual cash rate"
    if not (RAW_DIR / F1_FILE).exists():
        line = "F1 is not on disk (gitignored). Re-download from MANIFEST.csv to run it."
        return result(name, "NOT RUN", line, line)

    f1 = load_f1()
    zero = yields[0.0]
    # F1 value on each F17 date (NaN where F1 has no row for that date).
    cash_same = f1["cash_rate"].reindex(zero.index)
    cash_prev = f1["cash_rate"].shift(1).reindex(zero.index)  # previous F1 row, a contrast only
    in_effect = target_in_effect(zero.index, changes)
    missing_f1 = int(cash_same.isna().sum())

    diff_bp = ((zero - cash_same) * BP).round(2)        # 0-year minus same-day cash rate, bp
    match_same = diff_bp.abs() < TOL * BP                # NaN (no F1 row) compares as False
    match_prev = ((zero - cash_prev) * BP).abs() < TOL * BP
    match_target = ((zero - in_effect) * BP).abs() < TOL * BP
    off_target = ~match_target                           # the days check 8 found off the target
    share = match_same.mean()

    table = pd.DataFrame({
        "compared with": ["F1 cash rate, same day (main)", "F1 cash rate, previous F1 date (contrast)",
                          "A2 target in effect (check 8)"],
        "% of F17 days matching": [share * 100, match_prev.mean() * 100, match_target.mean() * 100],
        "days not matching": [int((~match_same).sum()), int((~match_prev).sum()), int((~match_target).sum())],
    })

    # Largest absolute differences: the days that do not match, biggest
    # first (at most 10). Only the date and the difference are shown.
    worst = diff_bp[~match_same].abs().sort_values(ascending=False).head(10)
    worst_text = ("; ".join(f"{d:%d %b %Y} ({diff_bp[d]:+.2f} bp)" for d in worst.index)
                  if len(worst) else "none: every F17 day matches")
    max_abs = diff_bp.abs().max()

    # Why the previous-date contrast misses: the F17 days on which the cash
    # rate moved since the previous F1 date, split into A2 effective dates and
    # other days (moves of the actual rate with no decision).
    moved = zero.index[~match_prev]
    n_effective = int(moved.isin(changes["effective_date"]).sum())
    other = moved[~moved.isin(changes["effective_date"])]
    other_years = (f"years {other.year.min()}-{other.year.max()}" if len(other) else "none")

    status = "PASS" if share > ZERO_MATCH_SHARE else "FLAG"
    line = (f"0-year point = F1 interbank overnight cash rate on {share:.1%} of {len(zero):,} F17 days (largest "
            f"difference {max_abs:.2f} bp), including all {int(off_target.sum()):,} days it is off the target; "
            f"{match_prev.mean():.1%} against the previous day's rate, {match_target.mean():.1%} against the "
            "target. Confirms check 8: the 0-year point is the actual cash rate.")
    body = (
        "F1's 'Interbank Overnight Cash Rate' (Series ID FIRMMCRID, found by `data.load_f1` from its title) is "
        "the actual cash rate. F1 is kept in memory only; this section holds statistics.\n\n"
        f"A day matches if the two differ by less than {TOL * BP:.1f} bp. F17 dates with no F1 row: "
        f"{missing_f1}.\n\n" + md_table(table, digits=1) + "\n\n"
        f"On the {int(off_target.sum()):,} F17 days where the 0-year point is off the A2 target (check 8), it "
        f"equals F1's same-day cash rate on {match_same[off_target].mean():.1%} of them.\n\n"
        f"Largest absolute differences (0-year minus same-day F1 cash rate): {worst_text}. Largest overall: "
        f"{max_abs:.2f} bp.\n\n"
        "How to read this. The same-day match settles check 8: the F17 curve starts at the actual cash rate, "
        "so the 0-year point's gap to the target in check 8 is the cash rate trading below the target, not "
        f"a fitting error. The previous-date row misses on the {len(moved)} F17 days on which the cash rate "
        f"differs from its previous F1 value: {n_effective} effective dates of target changes and {len(other)} "
        f"small moves of the actual rate between decisions ({other_years}). The 0-year point moves with it on "
        "the same day every time. For M2 this supports the gap adjustment (forward plus "
        "target minus 0-year point) and the outcome rule (A2 target, never the 0-year point). For M5's OIS "
        "benchmark, whose outcome is the realised average of F1's cash rate, it means F17 starts at the overnight "
        "rate that OIS references, so both can be judged against that average; beyond 0 years F17 is still a "
        "government-bond curve (see M5 section 1c)."
    )
    return result(name, status, line, body)


# Outputs

def write_report(results):
    """Assemble outputs/m1/validation_report.md from the check results.

    Layout: header, summary table (one line per check), one
    section per check, then the list of files written to outputs/m1/.
    """
    counts = pd.Series([r["status"] for r in results]).value_counts()
    summary = pd.DataFrame({
        "#": range(1, len(results) + 1),
        "check": [r["name"] for r in results],
        "result": [r["status"] for r in results],
        "interpretation": [r["line"] for r in results],
    })
    parts = [
        "# M1 - data validation report",
        "",
        f"Generated by `src/m1_validate.py` on {date.today():%d %b %Y}. Inputs: `data/raw/` "
        f"({F17_FILES['yields']}, {F17_FILES['forwards']}, {A2_FILE}; checks 11-12 also {F2_FILE} and "
        f"{F1_FILE}). Rates in per cent per annum, differences in basis points (bp).",
        "",
        "F1 and F2 are gitignored third-party data (F1's OIS columns are FENICS data; F2's licence is "
        "unchecked). This report and its charts hold only statistics derived from them, never their series.",
        "",
        "PASS = as expected. FLAG = a finding to read before the next step, not necessarily an error. "
        "NOT RUN = input missing.",
        "",
        f"Totals: {', '.join(f'{k} {v}' for k, v in counts.items())}.",
        "",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
    ]
    for number, item in enumerate(results, start=1):
        parts += [f"## {number}. {item['name']} - {item['status']}", "", item["body"], ""]
    parts += [
        "## Files written",
        "",
        "- `outputs/m1/validation_report.md` (this file)",
        "- `outputs/m1/zero_year_vs_target.csv` - daily 0-year point, both target alignments, spread",
        "- `outputs/m1/forward_check_by_maturity.png`, `outputs/m1/ycc_3y_vs_target.png`, "
        "`outputs/m1/zero_year_spread.png`",
        "- `outputs/m1/f2_10y_gap.png` - check 11, the F2 - F17 10-year gap (written only when F2 is on disk)",
        "",
    ]
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main():
    """Run every M1 check, write the report and figures, print a summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yields = load_f17("yields")
    forwards = load_f17("forwards")
    changes = load_a2()

    results = [
        check_checksums(),
        check_f17_format(),
        check_f17_structure(yields, forwards),
        check_gaps(yields),
        check_values(yields, forwards, changes),
        check_a2(changes),
        check_changes(changes, yields),
    ]
    zero_result, zero_daily = check_zero_year(yields, changes)
    forward_result, forward_by_m = check_forwards(yields, forwards, changes)
    ycc_result, ycc_table = check_ycc(yields)
    f2_result, f2_levels = check_f2(yields)  # check 11 (M5); f2_levels is None if F2 is missing
    results += [zero_result, forward_result, ycc_result, f2_result,
                check_f1_cash_rate(yields, changes)]  # check 12 (M5)

    # zero_year_vs_target.csv stays F17 + A2 only: F1 and F2 series are never written to disk.
    zero_daily.assign(date=zero_daily["date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUT_DIR / "zero_year_vs_target.csv", index=False)
    plot_zero_year(zero_daily, OUT_DIR / "zero_year_spread.png")
    plot_forward_check(forward_by_m, OUT_DIR / "forward_check_by_maturity.png")
    plot_ycc(ycc_table, OUT_DIR / "ycc_3y_vs_target.png")
    if f2_levels is not None:
        plot_f2_gap(f2_levels, OUT_DIR / "f2_10y_gap.png")

    write_report(results)

    print(f"M1 validation: report written to {REPORT_PATH.relative_to(PROJECT_DIR)}")
    for number, item in enumerate(results, start=1):
        print(f"{number:>2}. [{item['status']}] {item['name']}: {item['line']}")


if __name__ == "__main__":
    main()
