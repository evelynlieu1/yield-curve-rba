"""Loaders and small helpers for the yield curve vs RBA case study.

Every later milestone imports from this file, so the rules that decide how the
raw RBA files are read live in one place:

- F17 (zero-coupon yields and forward rates) is read from the two CSV files in
  data/raw/. Values are per cent per annum. Columns are named by maturity in
  years (0.0, 0.25, ... 10.0). Blanks become NaN (missing), never zero.
- A2 (cash rate target changes) is read from the xlsx file in data/raw/.
  Columns are found from their header text, not their position. Early-1990s
  rows hold ranges such as "-0.50 to -1.00"; these become NaN and are flagged,
  so no number is ever invented for them.
- target_in_effect and target_known_at_close turn the A2 change list into a
  target for any set of dates (conventions in the README, "Conventions used
  everywhere"). Dates the A2 file cannot know about (after its publication
  date plus one day) get NaN, never an invented "no change".
- F1 (cash rate target, actual cash rate, OIS) and F2 (bond yields), added
  in M5, are read by load_f1 and load_f2. Both are gitignored third-party
  data: keep them in memory, never write their series to data/processed/ or
  outputs/ (details in those functions).

Nothing here writes to data/raw/. The raw RBA headers contain long dashes; the
code finds maturities with number patterns, so it never needs to type one.
"""

from pathlib import Path
import csv
import hashlib
import io
import re

import numpy as np
import pandas as pd


# Paths and fixed facts about the inputs

# Project folder = the parent of src/. All paths are built from it so the code
# works no matter which folder Python is started from.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MANIFEST_PATH = PROJECT_DIR / "data" / "MANIFEST.csv"

# The two F17 files. 'kind' is the short name used by load_f17.
F17_FILES = {
    "yields": "rba_f17_zero_coupon_yields.csv",
    "forwards": "rba_f17_forward_rates.csv",
}
A2_FILE = "rba_a02_monetary_policy_changes.xlsx"

# The F17 maturity grid the README expects: 0 to 10 years in quarter-year steps,
# 41 points. Built from integers so there is no floating point drift
# (0.25 * k is exact in binary floating point).
MATURITY_GRID = [0.25 * k for k in range(41)]

# A2 notes: "Commencing February 2008, the announcement is made on the day of
# the Reserve Bank Board meeting, effective the following day. Prior to this
# time, the announcement and effective dates were the same."
# Changes with an effective date on or after this day were announced the
# calendar day before.
# Timing sources used by target_known_at_close (and the YCC regimes in
# m1_validate.py):
#   - RBA, Cash Rate Target, https://www.rba.gov.au/statistics/cash-rate/:
#     "A media release is issued at 2.30 pm after each Monetary Policy Board
#     meeting", with any change "taking effect the following day". The date
#     column of that table is headed "Effective Date".
#   - RBA, Review of the Yield Target,
#     https://www.rba.gov.au/monetary-policy/reviews/yield-target/index.html,
#     refers to the bond market's "4:30 pm close".
#   - So a 2.30 pm decision on day d is already in the 4:30 pm close on d.
#     The one off-cycle decision in the sample (Thu 19 Mar 2020, media release
#     2020-08, https://www.rba.gov.au/media-releases/2020/mr-20-08.html)
#     announced a "press conference ... at 4.00pm AEDT today", so it too came
#     out before the close.
NEXT_DAY_EFFECT_FROM = pd.Timestamp("2008-02-01")

# F17 dates look like 03-Jan-2017.
F17_DATE_FORMAT = "%d-%b-%Y"


# Small helpers

def detect_encoding(path):
    """Work out which text encoding a raw CSV file uses.

    Input:  path to a file.
    Output: 'utf-8-sig' if the file starts with a UTF-8 byte order mark,
            'utf-8' if it decodes as UTF-8 without one, otherwise 'cp1252'
            (Windows-1252, which can decode any byte).
    Why:    the README expected Windows-1252, but the files on disk start with a
            UTF-8 byte order mark. Detecting it means the loader works for
            either, and the validation report can say which one it found.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def sha256_of_file(path, normalise_crlf=False):
    """SHA-256 checksum of a file, as a 64-character hex string.

    Inputs:
        path           - file to hash.
        normalise_crlf - if True, every Windows line ending (CR LF, bytes
                         0D 0A) is turned into a Unix one (LF, byte 0A)
                         before hashing.
    Output: the hex digest.
    Why the option: git on Windows (core.autocrlf=true) rewrites LF to CRLF
    when it checks out text files. The bytes then differ from the download
    recorded in data/MANIFEST.csv, but the content is the same. Hashing the
    LF-normalised bytes recovers the original checksum. Only use it on text
    files: in a binary file (xlsx) the bytes 0D 0A are data.
    """
    data = Path(path).read_bytes()
    if normalise_crlf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _to_float(cell, where):
    """Turn one F17 text cell into a float.

    Inputs:  cell  - the text from the CSV.
             where - a short description of the cell, used in the error.
    Output:  float, or NaN if the cell is blank (blank = missing, never 0).
    Raises:  ValueError if the cell holds text that is not a number. F17 is a
             purely numeric table, so text there means the file has changed
             and should be looked at, not silently skipped.
    """
    text = cell.strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Non-numeric value {text!r} at {where}") from None


def maturity_from_title(title):
    """Read the maturity in years from an F17 column title.

    Input:  a title such as 'Zero-coupon yield <long dash> 0.25 yrs' or
            '... 1 yr'.
    Output: the maturity as a float (0.25, 1.0, ...).
    How:    the pattern looks for a number followed by 'yr' or 'yrs'. It
            never touches the long dash, so the code never has to type one.
    Raises: ValueError if no maturity is found.
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*yrs?\b", title)
    if not matches:
        raise ValueError(f"No maturity found in F17 title {title!r}")
    return float(matches[-1])


def maturity_from_series_id(series_id):
    """Read the maturity in years from an F17 series ID.

    Input:  an ID such as 'FZCY25D' (yields) or 'FZCF250D' (forwards). The
            number is the maturity in hundredths of a year.
    Output: the maturity as a float (0.25, 2.5, ...).
    Why:    a second, independent source for the maturity, so the loader can
            check that the title and the ID agree.
    Raises: ValueError if the ID does not have that shape.
    """
    match = re.fullmatch(r"FZC[YF](\d+)D", series_id.strip())
    if match is None:
        raise ValueError(f"Unexpected F17 series ID {series_id!r}")
    return int(match.group(1)) / 100


# F17 loader

def load_f17(kind):
    """Load F17 zero-coupon yields or forward rates from data/raw/.

    Input:  kind - 'yields' or 'forwards'.
    Output: DataFrame
              index   - 'date', datetime, one row per F17 business day, in
                        file order (the validation script checks the order);
              columns - maturity in years as floats 0.0, 0.25, ..., 10.0;
              values  - per cent per annum, float, NaN where the file is blank.
    File layout (both files): a metadata block of labelled rows ('Title',
    'Description', 'Frequency', 'Type', 'Units', two blank rows, 'Source',
    'Publication date', 'Series ID'), then one row per date. The first cell of
    each metadata row is its label, so the block is found by label, not by
    counting lines.
    Checks done here (the loader stops with an error if any fails):
      - the maturity read from each title equals the one read from its
        series ID;
      - the maturities are exactly MATURITY_GRID (41 points, 0 to 10 years);
      - every data row has one cell per column;
      - every non-blank cell is a number.
    """
    if kind not in F17_FILES:
        raise ValueError(f"kind must be one of {sorted(F17_FILES)}, got {kind!r}")
    path = RAW_DIR / F17_FILES[kind]
    text = path.read_text(encoding=detect_encoding(path))
    rows = list(csv.reader(io.StringIO(text)))

    # Find the labelled metadata rows by their first cell. Stop at the
    # 'Series ID' row: it is the last header row, and everything after it is
    # data (whose first cell is a date, not a label).
    labels = {}
    for i, row in enumerate(rows):
        label = row[0].strip() if row else ""
        if label:
            labels[label] = i
        if label == "Series ID":
            break
    if "Title" not in labels or "Series ID" not in labels:
        raise ValueError(f"F17 {kind}: header block without 'Title' and 'Series ID' rows")
    title_row = rows[labels["Title"]]
    id_row_index = labels["Series ID"]
    id_row = rows[id_row_index]

    # Maturities from two independent sources; they must agree.
    from_titles = [maturity_from_title(t) for t in title_row[1:]]
    from_ids = [maturity_from_series_id(s) for s in id_row[1:]]
    if from_titles != from_ids:
        raise ValueError(f"F17 {kind}: titles and series IDs disagree on maturities")
    if from_titles != MATURITY_GRID:
        raise ValueError(f"F17 {kind}: maturities are not the 0 to 10 year quarter grid")

    # Data rows are every non-empty row after the 'Series ID' row.
    n_cols = len(title_row)
    dates, values = [], []
    for line_no, row in enumerate(rows[id_row_index + 1:], start=id_row_index + 2):
        if not row or not row[0].strip():
            continue
        if len(row) != n_cols:
            raise ValueError(f"F17 {kind}: line {line_no} has {len(row)} cells, expected {n_cols}")
        dates.append(row[0].strip())
        values.append([_to_float(c, f"{kind} line {line_no}") for c in row[1:]])

    index = pd.DatetimeIndex(pd.to_datetime(dates, format=F17_DATE_FORMAT), name="date")
    frame = pd.DataFrame(np.array(values, dtype=float), index=index, columns=from_titles)
    frame.columns.name = "maturity_years"
    return frame


# A2 loader

# A range written as text, such as '-0.50 to -1.00' or '17.00 to 17.50'.
_RANGE_PATTERN = re.compile(r"^\s*-?\d+(?:\.\d+)?\s+to\s+-?\d+(?:\.\d+)?\s*$", re.IGNORECASE)


def _a2_number(cell):
    """Turn one A2 cell into (number, text).

    Input:  a cell as read from the xlsx (float, int, str or empty).
    Output: (value, text)
              value - float, or NaN if the cell is empty or is text that is
                      not a single number (for example a range);
              text  - the original text when the cell was text that could not
                      be read as one number, otherwise None.
    Why:    ranges such as '-0.50 to -1.00' must not crash the loader and must
            not become a number (not the midpoint, not either end). Keeping the
            text lets the report show exactly what was skipped.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan, None
    if isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, bool):
        return float(cell), None
    text = str(cell).strip()
    if text == "":
        return np.nan, None
    try:
        return float(text), None
    except ValueError:
        return np.nan, text


def is_range_text(text):
    """True if text looks like a range such as '-0.50 to -1.00'."""
    return text is not None and bool(_RANGE_PATTERN.match(text))


def load_a2():
    """Load every cash rate target change from the A2 xlsx in data/raw/.

    Input:  none (reads data/raw/rba_a02_monetary_policy_changes.xlsx, first
            sheet).
    Output: DataFrame, one row per A2 data row, sorted by effective date.
            Most rows are target changes; a few have a zero change because
            A2 also lists moves in other administered rates (the validation
            report counts them). Columns:
              effective_date    - datetime, the day the new target applies;
              announcement_date - datetime, derived: the day before the
                                  effective date from February 2008, the same
                                  day before that (A2 notes);
              change_pp         - change in the target, percentage points,
                                  NaN if the cell was a range;
              new_target        - the target after the change, per cent,
                                  NaN if the cell was a range;
              change_text       - the original text when change_pp is NaN
                                  because the cell held text, else missing;
              target_text       - the same for new_target;
              has_text          - True if either cell held text;
              covers_until      - datetime, the same on every row: the last
                                  day whose target this file can know, which
                                  is the 'Publication date' plus one day. A
                                  decision announced on the publication date
                                  takes effect the next day; nothing later is
                                  known. target_in_effect returns NaN after it.
    How the columns are found: the sheet has labelled header rows ('Title',
    ..., 'Publication date', 'Series ID') in its label column. The label
    column is the one that holds the 'Series ID' label; data rows are below
    that row and their dates sit in the label column. The change and
    new-target columns are the ones whose 'Title' cell reads 'Change in Cash
    Rate Target' and 'New Cash Rate Target' (case and spacing ignored).
    Nothing depends on column positions.
    Numbers are rounded to 6 decimals only to remove binary floating point
    noise (for example 0.15000000000000002); A2 rates have 2 decimals.
    """
    path = RAW_DIR / A2_FILE
    sheet = pd.read_excel(path, sheet_name=0, header=None, dtype=object)

    # Locate the 'Series ID' label: its column is the label/date column and
    # its row is the last header row.
    label_col, id_row = None, None
    for col in sheet.columns:
        hits = [i for i, v in sheet[col].items() if isinstance(v, str) and v.strip() == "Series ID"]
        if hits:
            label_col, id_row = col, hits[0]
            break
    if label_col is None:
        raise ValueError("A2: no 'Series ID' row found")

    def find_label_row(label):
        """Row number of the first cell in the label column that reads 'label'."""
        hits = [i for i, v in sheet[label_col].items() if isinstance(v, str) and v.strip() == label]
        if not hits:
            raise ValueError(f"A2: no {label!r} row found")
        return hits[0]

    # The 'Title' row holds the column names.
    titles = sheet.loc[find_label_row("Title")]

    def find_column(wanted):
        """Return the one column whose Title matches 'wanted' (case-insensitive)."""
        pattern = re.compile(r"^\s*" + r"\s+".join(wanted.split()) + r"\s*$", re.IGNORECASE)
        found = [c for c, v in titles.items() if isinstance(v, str) and pattern.match(v)]
        if len(found) != 1:
            raise ValueError(f"A2: expected one column titled {wanted!r}, found {len(found)}")
        return found[0]

    change_col = find_column("Change in Cash Rate Target")
    target_col = find_column("New Cash Rate Target")

    # The 'Publication date' row says how far the file can see (covers_until
    # below). Read it from the change column, the column this loader uses.
    publication = pd.Timestamp(sheet.loc[find_label_row("Publication date"), change_col]).normalize()

    # Data rows: below the Series ID row, with something in the date column.
    body = sheet.loc[id_row + 1:]
    body = body[body[label_col].notna()]

    records = []
    for _, row in body.iterrows():
        change, change_text = _a2_number(row[change_col])
        target, target_text = _a2_number(row[target_col])
        records.append({
            "effective_date": pd.Timestamp(row[label_col]),
            "change_pp": change,
            "new_target": target,
            "change_text": change_text,
            "target_text": target_text,
        })
    changes = pd.DataFrame(records)

    changes["change_pp"] = changes["change_pp"].round(6)
    changes["new_target"] = changes["new_target"].round(6)
    changes["has_text"] = changes["change_text"].notna() | changes["target_text"].notna()
    changes["effective_date"] = pd.to_datetime(changes["effective_date"]).dt.normalize()

    # Announcement date, derived from the A2 note (see NEXT_DAY_EFFECT_FROM).
    next_day = changes["effective_date"] >= NEXT_DAY_EFFECT_FROM
    changes["announcement_date"] = changes["effective_date"].where(
        ~next_day, changes["effective_date"] - pd.Timedelta(days=1)
    )

    # A decision announced on the publication date takes effect the next day;
    # nothing later is known.
    changes["covers_until"] = publication + pd.Timedelta(days=1)

    changes = changes.sort_values("effective_date").reset_index(drop=True)
    if changes["effective_date"].duplicated().any():
        raise ValueError("A2: two changes share an effective date")
    if changes["effective_date"].max() > changes["covers_until"].iloc[0]:
        raise ValueError("A2: a change takes effect after the file's publication date + 1 day")
    columns = ["effective_date", "announcement_date", "change_pp", "new_target",
               "change_text", "target_text", "has_text", "covers_until"]
    return changes[columns]


# Cash rate target on any date

def _as_ns(values):
    """Convert dates to a numpy datetime64[ns] array (one unit for comparisons)."""
    return np.asarray(pd.DatetimeIndex(values).normalize(), dtype="datetime64[ns]")


def target_in_effect(dates, changes=None):
    """The cash rate target that applies on each calendar day in 'dates'.

    Inputs:
        dates   - anything pandas can turn into a DatetimeIndex (a
                  DatetimeIndex, a list of dates, a Series of dates). Times of
                  day are dropped.
        changes - the output of load_a2(). Loaded from disk if None. Passing
                  it in avoids re-reading the file and lets tests use a small
                  made-up table.
    Output: Series of targets (per cent), indexed by the normalised dates,
            named 'target_in_effect'.
    Convention (the README): A2 dates are effective dates, so the target on day d
    is the new_target of the latest change whose effective_date is on or
    before d. Dates before the first change get NaN. A change whose
    new_target is NaN (an early-1990s range) gives NaN until the next change.
    Dates after the file's coverage (column 'covers_until' from load_a2) also
    get NaN: the file cannot know whether the target changed after it was
    published, so it must not be read as "no change". A table without that
    column (for example a made-up table in a test) has no end date.
    How: np.searchsorted finds, for every date at once, how many effective
    dates are <= d (side='right'); that count minus one is the row of the
    change in force.
    """
    if changes is None:
        changes = load_a2()
    changes = changes.sort_values("effective_date")
    effective = _as_ns(changes["effective_date"])
    targets = changes["new_target"].to_numpy(dtype=float)

    index = pd.DatetimeIndex(dates).normalize()
    position = np.searchsorted(effective, _as_ns(index), side="right") - 1
    values = np.where(position >= 0, targets[np.clip(position, 0, None)], np.nan)
    if "covers_until" in changes.columns:
        beyond = _as_ns(index) > _as_ns([changes["covers_until"].max()])[0]
        values = np.where(beyond, np.nan, values)
    return pd.Series(values, index=index, name="target_in_effect")


def target_known_at_close(dates, changes=None):
    """The cash rate target known at the F17 close on each day in 'dates'.

    Inputs: as for target_in_effect.
    Output: Series of targets (per cent), indexed by the normalised input
            dates (not shifted), named 'target_known_at_close'.
    Convention (the README): target_known_at_close(d) = target_in_effect(d + 1
    calendar day). Since February 2008 a decision is announced on the Board
    day (2.30 pm) and takes effect the next day, so a decision announced on d
    is already known (and priced) at the 4:30 pm close on d. Sources are
    listed next to NEXT_DAY_EFFECT_FROM at the top of this file.
    The last covered day gives NaN: target_in_effect(covers_until + 1 day) is
    unknown, because the file cannot see a decision announced on that day.
    Raises: ValueError for any date before February 2008. Before then
    announcement and effective dates were the same day, so d + 1 would look
    one day ahead. The curve sample starts in 2017; the error stops anyone
    reusing this helper on older data without thinking about it.
    """
    index = pd.DatetimeIndex(dates).normalize()
    if (index < NEXT_DAY_EFFECT_FROM).any():
        raise ValueError("target_known_at_close assumes next-day effect (A2 notes, from Feb 2008); "
                         "earlier dates would look one day ahead")
    shifted = target_in_effect(index + pd.Timedelta(days=1), changes)
    return pd.Series(shifted.to_numpy(), index=index, name="target_known_at_close")


# Daily yield changes, with the suspect F17 days left out

# F17 days whose short-end move looks like a curve-fitting artefact, not a
# market move. Found by M1: outputs/m1/validation_report.md, section 5
# ("Blanks, ranges and daily moves", one-day reversals). Keys are dates
# (YYYY-MM-DD), values the M1 note. Decision taken after M1 (the README M3):
# analyses of daily yield CHANGES leave out the change into and the change
# out of each day; a robustness variant keeps them. m1_validate.py imports
# this dict, so the M1 report and the later milestones use the same list.
SUSPECT_DAYS = {
    "2019-02-14": "probable fitting artefact: 0.25y falls 15 bp (0.5y 11 bp) while the 0-year point is fixed "
                  "and the 1-year rises; no RBA decision near; back the next day",
    "2019-12-13": "probable fitting artefact: 0.25y and 0.5y jump while the 0-year point is fixed and the "
                  "1-year falls; no RBA decision near; back the next day",
}


def daily_changes_bp(yields, maturities, drop_suspect=True):
    """Daily changes in basis points of F17 yields at some maturities.

    Inputs:
        yields       - F17 frame from load_f17('yields'): rows are F17
                       business days in date order, columns maturities in
                       years, values per cent.
        maturities   - list of the maturity columns to keep (e.g. 0.25 to 10).
        drop_suspect - if True (the main analysis), leave out the change INTO
                       and the change OUT OF each day in SUSPECT_DAYS.
    Output: DataFrame of changes in bp, indexed by the date the change ends
            on, one column per maturity. The first F17 day has no change and
            is not in the output.
    Conventions:
        - change on day d = (yield on d - yield on the previous F17 day) * 100.
          Across a weekend or holiday the change covers several calendar days.
        - the change into suspect day d is the row dated d; the change out of
          it is the row dated the next F17 day. Both rows are dropped, so the
          artefact and its reversal leave together.
        - values are rounded to 6 decimals only to remove binary floating
          point noise (yields have 2 decimals, so changes are whole bp).
        - the rule uses hindsight (a day is suspect because the curve came
          back the next day). In out-of-sample tests (M4) it may be applied
          to estimation windows; residuals on evaluation days must be
          reported both with and without these days.
    """
    frame = yields.loc[:, list(maturities)]
    changes = (frame.diff() * 100.0).round(6).iloc[1:]
    if drop_suspect:
        drop = []
        for key in SUSPECT_DAYS:
            day = pd.Timestamp(key)
            if day in frame.index:
                j = frame.index.get_loc(day)
                drop += list(frame.index[j:j + 2])  # the day itself and the next F17 day
        changes = changes[~changes.index.isin(drop)]
    return changes


# F1 (money market) and F2 (government bond yields) loaders, added in M5

# Both files were downloaded on 20 Oct 2026 with the user's OK (sizes and
# checksums in data/MANIFEST.csv). Both are gitignored because they hold
# third-party data (licensing notes in load_f1 and load_f2). Rule for every
# script: keep what these loaders return in memory. Never write the series,
# or processed copies of them, to data/processed/ or outputs/. Only
# statistics and charts derived from them may be written.
F1_FILE = "rba_f1_money_market_daily.csv"
F2_FILE = "rba_f2_govt_bond_yields_daily.csv"

# The F1 columns this project uses. Key = output column name. Value =
# (pattern that must match the WHOLE 'Title' cell, case ignored; the Series ID
# that column must have). Matching the whole cell matters: 'Cash Rate Target'
# must not pick up 'Change in the Cash Rate Target', and 'Interbank Overnight
# Cash Rate' must not pick up 'Highest Interbank Overnight Cash Rate'. The
# Series ID is a second, independent check that the right column was found.
F1_COLUMNS = {
    "cash_target": (r"cash\s+rate\s+target", "FIRMMCRTD"),
    "cash_rate": (r"interbank\s+overnight\s+cash\s+rate", "FIRMMCRID"),
    "ois_1m": (r"1\s*-\s*month\s+ois", "FIRMMOIS1D"),
    "ois_3m": (r"3\s*-\s*month\s+ois", "FIRMMOIS3D"),
    "ois_6m": (r"6\s*-\s*month\s+ois", "FIRMMOIS6D"),
}

# F2 nominal bond titles read 'Australian Government 10 year bond'. The
# indexed bond is titled 'Australian Government Indexed Bond': it has no
# number before 'year bond', so this pattern never matches it.
_F2_TITLE_PATTERN = re.compile(r"australian\s+government\s+(\d+)\s+year\s+bond", re.IGNORECASE)
F2_MATURITIES = [2.0, 3.0, 5.0, 10.0]


def _read_rba_table(path):
    """Split an RBA statistical-table CSV (the F1 / F2 layout) into header and data rows.

    Input:  path to a CSV in data/raw/.
    Output: (header, rows)
              header - dict: label of each metadata row ('Title',
                       'Description', 'Units', 'Source', 'Publication date',
                       'Series ID', ...) -> the cells after the label, one per
                       data column, in file order;
              rows   - list of (line number, date text, cells), one per data
                       row, where cells has one entry per data column.
    File layout: line 1 names the table (it contains a long dash and is not
    used), then labelled metadata rows ending with 'Series ID', then one row
    per date. The encoding is found by detect_encoding (both files start with
    a UTF-8 byte order mark). Empty lines are skipped (F2 ends with about
    61,000 of them).
    Short rows: the RBA leaves out trailing blank cells. For example, an F1
    row from 2023 stops after the bank bill columns, because the OIS and
    Treasury note columns are blank by then. Short rows are padded with blank
    cells to the width of the Title row, so blank stays blank (NaN later).
    A row LONGER than the Title row stops the loader: it would mean the
    columns have shifted.
    """
    path = Path(path)
    lines = list(csv.reader(io.StringIO(path.read_text(encoding=detect_encoding(path)))))

    # Metadata rows: first cell is the label. Stop at 'Series ID', the last one.
    header, id_line = {}, None
    for i, row in enumerate(lines):
        label = row[0].strip() if row else ""
        if label:
            header[label] = row[1:]
        if label == "Series ID":
            id_line = i
            break
    if id_line is None or "Title" not in header:
        raise ValueError(f"{path.name}: header block without 'Title' and 'Series ID' rows")

    n_cols = len(header["Title"])
    rows = []
    for line_no, row in enumerate(lines[id_line + 1:], start=id_line + 2):
        if not row or not row[0].strip():
            continue
        cells = row[1:]
        if len(cells) > n_cols:
            raise ValueError(f"{path.name}: line {line_no} has {len(cells)} values, the Title row has {n_cols}")
        rows.append((line_no, row[0].strip(), cells + [""] * (n_cols - len(cells))))
    return header, rows


def _rba_frame(rows, positions, file_name):
    """Build a date-indexed float DataFrame from _read_rba_table rows.

    Inputs:
        rows      - the 'rows' output of _read_rba_table.
        positions - dict: output column name -> position of that column among
                    the data cells (0 = first column after the date).
        file_name - used in error messages.
    Output: DataFrame, index 'date' (datetime), one float column per entry of
            positions, in that order. Values are per cent as in the file;
            blank cells are NaN, never zero.
    Rules:
        - every non-blank kept cell must be a number (_to_float raises
          otherwise; text such as 'N/A' only appears in F1 columns this
          project does not read);
        - a date whose kept cells are ALL blank is left out: it carries no
          information for these columns, and keeping it would only put empty
          rows into daily changes (F2 NSW bank holidays, the F1 publication
          day). Rows with SOME blanks are kept, with NaN in the blank cells;
        - dates must be strictly increasing (no repeats, no reordering).
    """
    dates, values = [], []
    for line_no, date_text, cells in rows:
        row = [_to_float(cells[j], f"{file_name} line {line_no}") for j in positions.values()]
        if all(np.isnan(row)):
            continue
        dates.append(date_text)
        values.append(row)
    index = pd.DatetimeIndex(pd.to_datetime(dates, format=F17_DATE_FORMAT), name="date")  # same 03-Jan-2017 format as F17
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{file_name}: dates are not strictly increasing")
    return pd.DataFrame(np.array(values, dtype=float), index=index, columns=list(positions))


def load_f1():
    """Load the cash rate target, the actual cash rate and 1, 3, 6-month OIS from F1.

    Input:  none (reads data/raw/rba_f1_money_market_daily.csv).
    Output: DataFrame
              index   - 'date', datetime, F1 dates in file order (Sydney
                        business days; F1 also has rows on NSW bank holidays,
                        where only the cash rate columns are filled);
              columns - floats, per cent per annum:
                  cash_target - 'Cash Rate Target' on the date;
                  cash_rate   - 'Interbank Overnight Cash Rate', the actual
                                cash rate (equal to the target 2011-2019,
                                0-12 bp below it 2020-2025);
                  ois_1m, ois_3m, ois_6m - '1-month OIS', '3-month OIS',
                                '6-month OIS' (overnight indexed swap rates);
              values  - NaN where the file is blank, never zero.
    How the columns are found: by their Title text (F1_COLUMNS), each checked
    against its Series ID and against a 'Per cent' unit. Positions are never
    assumed. Short rows are padded and all-blank dates dropped
    (_read_rba_table, _rba_frame).

    Date convention of cash_target (checked against A2 in
    tests/test_data_f1_f2.py): F1's target on day d is the target IN EFFECT
    on d, the same as target_in_effect(d). On a Board (announcement) day F1
    still shows the old target; the new one appears from the effective date,
    the next day. So it is NOT target_known_at_close(d), which already
    includes a decision announced on d.

    Coverage in the 20 Oct 2026 download:
      - cash_target and cash_rate: 4 Jan 2011 to 19 Oct 2026, no blanks.
        (The 20 Oct 2026 row holds only the total return index, so it is
        dropped as all-blank.)
      - OIS: from 4 Jan 2011, the first F1 date, to 1 Dec 2022. Blank after
        that; the file does not say why. Inside that span only 4 dates are
        blank (NSW bank holidays 1 Aug 2011, 3 Oct 2011, 6 Aug 2012,
        5 Aug 2013), the same dates for all three maturities. 985 NaN per
        OIS column in all (4 inside the span, 981 after it).

    Licensing (data/MANIFEST.csv, README "Licensing"): the 'Source' row of
    F1 names FENICS for the three OIS columns and ASX for the bank bill
    columns (not read here). Those are third-party data, not RBA's, so F1 is
    gitignored and its OIS values must not be published. The cash rate
    columns are RBA data under the RBA's own cash rate terms. Keep this
    frame in memory: no series or processed copies in data/processed/ or
    outputs/, only statistics and charts derived from them.
    """
    header, rows = _read_rba_table(RAW_DIR / F1_FILE)
    titles, ids, units = header["Title"], header["Series ID"], header["Units"]
    positions = {}
    for name, (pattern, series_id) in F1_COLUMNS.items():
        found = [j for j, t in enumerate(titles) if re.fullmatch(pattern, t.strip(), re.IGNORECASE)]
        if len(found) != 1:
            raise ValueError(f"F1: expected one column titled like {pattern!r}, found {len(found)}")
        j = found[0]
        if ids[j].strip() != series_id:
            raise ValueError(f"F1: column {titles[j]!r} has Series ID {ids[j]!r}, expected {series_id!r}")
        if not units[j].strip().lower().startswith("per cent"):
            raise ValueError(f"F1: column {titles[j]!r} has units {units[j]!r}, expected per cent")
        positions[name] = j
    return _rba_frame(rows, positions, "F1")


def load_f2():
    """Load the 2, 3, 5 and 10-year Australian Government bond yields from F2.

    Input:  none (reads data/raw/rba_f2_govt_bond_yields_daily.csv).
    Output: DataFrame
              index   - 'date', datetime, F2 dates in file order;
              columns - maturity in years as floats 2.0, 3.0, 5.0, 10.0
                        (columns.name 'maturity_years', as in load_f17);
              values  - per cent per annum, NaN where the file is blank,
                        never zero.
    What the yields are: the file's Description calls them yields on
    Australian Government bonds 'interpolated' to a fixed maturity. They are
    yields on actual bonds, not a fitted zero-coupon curve like F17, so the
    levels differ from F17 at the same maturity.
    How the columns are found: the maturity is read from each Title
    ('Australian Government 10 year bond') and checked against its Series ID
    ('FCMYGBAG10D'). The indexed (inflation-linked) bond has no maturity in
    its title and is left out. The maturities found must be exactly
    F2_MATURITIES. Short rows are padded and all-blank dates dropped
    (_read_rba_table, _rba_frame).

    Coverage in the 20 Oct 2026 download (published 16 Oct 2026):
      - 10 years: 20 May 2013 to 14 Oct 2026, no blanks once all-blank
        dates (21 NSW bank holidays) are dropped.
      - 2, 3, 5 years: 2 Sep 2013 to 14 Oct 2026. NaN on the 73 earlier
        dates, where only the 10-year has a value, and on 5 later dates
        where the 10-year has one (3 Jan, 29 Apr, 5 May, 15 Aug and
        15 Sep 2014). 78 NaN in each of these columns in all.
      - The 2-year yield is slightly negative on some days (lowest -0.001);
        that is a real value, not an error.

    Licensing (data/MANIFEST.csv): the file names the RBA as its source, but
    a web summary says the inputs come from Yieldbroker. That is not checked,
    so F2 is gitignored. Keep this frame in memory: no series or processed
    copies in data/processed/ or outputs/, only statistics and charts.
    """
    header, rows = _read_rba_table(RAW_DIR / F2_FILE)
    titles, ids, units = header["Title"], header["Series ID"], header["Units"]
    positions = {}
    for j, title in enumerate(titles):
        match = _F2_TITLE_PATTERN.fullmatch(title.strip())
        if match is None:
            continue  # the indexed bond
        maturity = float(match.group(1))
        if ids[j].strip() != f"FCMYGBAG{match.group(1)}D":
            raise ValueError(f"F2: column {title!r} has Series ID {ids[j]!r}, which does not match its maturity")
        if not units[j].strip().lower().startswith("per cent"):
            raise ValueError(f"F2: column {title!r} has units {units[j]!r}, expected per cent")
        positions[maturity] = j
    if sorted(positions) != F2_MATURITIES:
        raise ValueError(f"F2: found maturities {sorted(positions)}, expected {F2_MATURITIES}")
    positions = {m: positions[m] for m in F2_MATURITIES}
    frame = _rba_frame(rows, positions, "F2")
    frame.columns.name = "maturity_years"
    return frame
