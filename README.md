# Does the Australian yield curve price the RBA? - Report

Independent case study on public RBA data, January 2017 to September 2026. Not advice. Data to September 2026.

Terms used throughout:

- **bp** - basis point, 0.01 percentage points.
- **RBA tables** - F17 (the zero-coupon yield curve), F1 (money market), F2 (bond yields), A2 (cash rate decisions).
- **Acronyms** - each is spelled out the first time it appears.

## Summary

**1. Does the curve predict the cash rate? Mostly yes, but not with statistical confidence.**

- **Accuracy.** The curve's forward rate is about half as wrong as assuming "no change".
    - Typical error (RMSE, root-mean-square error) 3 months ahead: 16 bp against 36 bp.
    - 6 months ahead: 31 bp against 67 bp.
    - At 6 months the forward moves one-for-one with what the RBA then did: the slope of realised changes on priced changes is 1.01.
- **Not statistically significant.** The gain does not pass a 5% significance test over the full sample.
    - A Diebold-Mariano (DM) test gives p = 0.081 at 3 months and 0.114 at 6.
    - Most of the gain comes from the 2022-23 hiking cycle.
- **Where it was most wrong.** At turning points:
    - forecasts made in Jan-Mar 2022 missed the coming hikes by up to 131 bp;
    - forecasts made in Aug-Dec 2025 missed the 2026 hikes.
- **Against OIS.** Overnight index swaps (OIS) are the market's direct bet on the average cash rate. The curve forecasts as well as OIS over 2017-2022, but that is partly built in: the RBA fits the short end of its curve to OIS.

**2. Does a 3-factor hedge beat a duration hedge? Not in the planned test.**

- **The test.** A 7-year zero-coupon position is hedged out of sample for 1,943 days. The 3-factor hedge (level, slope, curvature) leaves more risk than duration: 1.04 against 0.93 bp per day.
- **Why.** The whole verdict rests on one 60-day period in mid-2019. There the curvature factor was so poorly estimated that the hedge took extreme positions.
    - In the other 32 of 33 periods the 3-factor hedge is the best (0.60 bp per day).
    - A 2-factor hedge (level and slope) beats duration throughout (0.74 bp per day).
- **On real bond yields** (F2) no factor hedge beats duration.
- **The honest reading.** A 3-factor hedge is better when its factors are well estimated. In practice it needs a guard against the times they are not.

The rest of this report shows the evidence and what does not hold up.

## Questions

1. **Expectations.** How well do the curve's forward rates anticipate changes in the RBA's cash rate target, and where were they most wrong?
2. **Hedging.** Does hedging with a 3-factor model leave less unhedged risk than hedging with duration alone, on data the model has not seen?

## Data

| RBA table | What it holds | Used for |
|---|---|---|
| F17 | Zero-coupon yields and instantaneous forward rates, 0-10 years in 0.25-year steps, 2,448 days (3 Jan 2017 - 30 Sep 2026) | Both questions |
| A2 | Every cash rate target change: 26 in the sample (9 cuts, 17 hikes) | Outcomes for question 1 |
| F1 | Cash rate target, actual overnight cash rate, 1, 3 and 6-month OIS (OIS ends 1 Dec 2022) | OIS benchmark |
| F2 | 2, 3, 5 and 10-year government bond yields | Cross-check, bond-yield hedging test |

All files were downloaded from rba.gov.au. Sizes and checksums are in `data/MANIFEST.csv`. No data files are stored in this repository: download them from the URLs in `data/MANIFEST.csv`. F1 and F2 carry third-party or unchecked data, so only statistics from them are published here.

**What the M1 checks found** ([validation report](outputs/m1/validation_report.md)):

- **Forward rates.** F17's forward rates are *instantaneous* forwards, so they can be read directly as the expected cash rate on a future date. Rebuilding the yields from them misses by 0.40 bp, against about 3 bp for 3-month period forwards.
- **The 0-year point** is the actual overnight cash rate, not the target. It matches F1 on all 2,448 days. From March 2020 to September 2025 it sat 1-12 bp below the target.
- **F17 against F2.** F17's 10-year yield moves with the F2 10-year bond yield: daily changes correlate 0.996.
- **Two fitting errors.** On 14 Feb 2019 and 13 Dec 2019 the short end jumps and fully reverses the next day. They are left out of the factor and hedging analysis.

**What the data cannot do:**

- **F17 is the RBA's fitted, revised curve**, not traded prices seen in real time.
- **26 decisions in four clusters** is a small sample.
- **A forward rate is not a pure forecast.** It also carries a term premium.

## Method

**Question 1** ([M2 report](outputs/m2/expectations_report.md)):

- **Forecast dates.** One per month: the last F17 day of each month, Jan 2017 onwards. Every month counts, holds as well as moves.
- **Forecast.** The F17 forward rate for 3, 6 or 12 months ahead.
- **Benchmark.** "No change": the target known at the close that day.
- **Outcome.** The target in effect 3, 6 or 12 months later.
- **Timing.**
    - A decision is announced at 2:30 pm and takes effect the next day.
    - F17 is the 4:30 pm close, so it already prices a same-day decision.
    - The benchmark therefore uses the target known at the close.
    - Tests check that no forecast uses data dated after its forecast date.
- **Statistics.**
    - Bias, MAE (mean absolute error) and RMSE.
    - The Diebold-Mariano test, which compares squared errors.
    - The Fama regression of realised on priced changes.
    - Both use Newey-West standard errors, because overlapping horizons make errors correlated.

**OIS benchmark** ([M5 report](outputs/m5/robustness_report.md), section 1). OIS prices the *average* cash rate over its term, so three forecasts of the average actual cash rate over the next 1, 3 and 6 months are compared:

- OIS;
- the F17 zero yield, which is the average of its forwards;
- no change.

**Question 2** ([M3 report](outputs/m3/factors_report.md), [M4 report](outputs/m4/hedging_report.md)):

- **Factors.** PCA (principal component analysis) of daily yield changes at 40 maturities from 0.25 to 10 years. The first three components are level, slope and curvature.
- **Position.** Long a 7-year zero, scaled to one unit of DV01 (the value change for a 1 bp move in yield).
- **Hedges:**

| Hedge | Instruments | How the weights are set |
|---|---|---|
| H0 | none | unhedged |
| H1 duration | 2y, 10y | DV01-neutral, split by maturity: 0.375 and 0.625 (nothing estimated) |
| H2 level + slope | 2y, 10y | zero exposure to factors 1-2 |
| H3 level + slope + curvature | 2y, 5y, 10y | zero exposure to factors 1-3 (three factors need three instruments) |

- **Out of sample.**
    - Estimate the factors on the trailing 500 days.
    - Fix the weights for the next 60 days.
    - Record the leftover daily P&L, then roll forward 60 days. That gives 33 periods.
- **Comparison.** The standard deviation of leftover P&L, and a paired test on squared residuals. The paired test clusters errors by 60-day period, because the weights are fixed within each one.

## Results

### 1. The curve as a forecaster of the cash rate

| Horizon | Forward RMSE (bp) | No-change RMSE (bp) | DM p (Newey-West / small-sample) | Fama slope b (p for b = 1) |
|---|---|---|---|---|
| 3 months | 16.0 | 36.3 | 0.081 / 0.131 | 0.87 (0.082) |
| 6 months | 30.9 | 66.6 | 0.114 / 0.203 | 1.01 (0.914) |

Source: [M2 report](outputs/m2/expectations_report.md), [M5 report](outputs/m5/robustness_report.md).

![Realised target and the 6-month forward](outputs/m2/target_vs_forward_6m.png)

- **The forward is far better than "no change", but not significantly.**
    - The 2022-23 Tightening forecasts supply 83% of the gain at 3 months.
    - Without them the 3-month p is 0.016. That row was added after seeing the results, so treat it with caution.
    - From 2022 onwards alone, p = 0.097.
- **The largest misses are turning points priced late.** The three largest 6-month misses (+131, +120, +78 bp) were forecasts made on 28 Feb, 31 Mar and 31 Jan 2022, before the 2022 hikes.
- **Bias.** On average the target ended above the forward (+4.6 bp at 3 months), the opposite sign to a term premium.
    - Only the 3-month bias is significant (p = 0.012).
    - About half of it is the actual cash rate sitting below the target.
    - This sample cannot sign a term premium.

![Forecast errors by episode](outputs/m2/errors_by_episode.png)

**Against OIS** (forecasting the average actual cash rate; 71 month-ends, Jan 2017 - Nov 2022, where F17 and OIS overlap):

| Horizon | OIS RMSE | F17 zero RMSE | No-change RMSE | OIS vs F17 p |
|---|---|---|---|---|
| 1 month | 4.8 | 6.1 | 12.5 | 0.187 |
| 3 months | 7.9 | 7.6 | 25.5 | 0.574 |
| 6 months | 14.0 | 13.6 | 43.2 | 0.418 |

Source: [M5 report](outputs/m5/robustness_report.md), section 1.

- **OIS and the curve cannot be told apart.** That is expected rather than reassuring: the RBA fits the 0-1 year part of its curve to OIS.
- **At 1 month OIS is better on every measure.** Its RMSE is 4.8 against 6.1 bp, and its bias is -0.1 against +1.1 bp. The curve has no 1-month point and cannot show a hike priced for the coming days.
- **After OIS ends.** To Aug 2026, the curve still beats "no change" at 1, 3 and 6 months, but significantly only at 1 month (p = 0.015, 0.066 and 0.109).

### 2. Level, slope and curvature

- **Three factors** explain 98.9% of daily curve moves:
    - level 90.1%;
    - slope 7.3%;
    - curvature 1.5%.
- **By period** the share ranges from 97.7% to 99.4%.
- **Stability.** Level keeps its shape outside yield curve control (YCC), with cosine similarity 0.987 or more. Slope and curvature vary more.
- **Under YCC** (2020-21, when the RBA capped the 3-year yield) slope takes 11.9% of the variance, against 4.4-5.6% elsewhere.
- **Curvature is weakly identified in some periods.** Its variance is only 2.2-2.7 times the next factor's in 2017-2019 and under YCC. That is what breaks the hedge below.

Source: `outputs/m3/factors_report.md`.

![Factor loadings](outputs/m3/loadings_full_sample.png)

### 3. Hedging

| Hedge | Leftover risk (bp of DV01 per day) | Risk removed |
|---|---|---|
| H0 unhedged | 5.46 | - |
| H1 duration | 0.93 | 97.1% |
| H2 level + slope | 0.74 | 98.1% |
| H3 level + slope + curvature | 1.04 | 96.4% |

Source: `outputs/m4/hedging_report.md` (1,943 out-of-sample days, Jan 2019 - Sep 2026).

- **H3 does not beat duration** (block-clustered p = 0.762).
- **One period decides it.** For 26 Jun - 18 Sep 2019 the curvature factor was almost indistinguishable from the next factor.
    - The equations for H3's weights were ill-conditioned: condition number 111, against 7.6 or less in every other period.
    - The weights were -2.66 / +6.15 / -3.00.
    - H3 then did worse than no hedge (4.93 against 3.96 bp per day).
- **In every other period H3 is the best hedge.** It has the lower mean squared residual than duration in 32 of 33 periods, and leaves 0.60 bp per day on the other 32.
- **H2 (level and slope) is the steady choice.** It beats duration in every 7-year run, with weights that stay in a narrow range.

**Robustness** ([M5 report](outputs/m5/robustness_report.md), sections 3-4). Each row changes one thing:

| Run | H1 | H2 | H3 | H3 vs H1 p |
|---|---|---|---|---|
| Planned design | 0.93 | 0.74 | 1.04 | 0.762 |
| Estimation window 250 days | 0.93 | 0.78 | 0.65 | <0.001 |
| Window 750 days | 0.95 | 0.79 | 7.55 | 0.331 |
| Re-estimate every 20 days | 0.93 | 0.74 | 0.79 | 0.282 |
| Re-estimate every 120 days | 0.93 | 0.75 | 1.20 | 0.614 |
| 5-year target | 1.06 | 1.03 | 0.80 | 0.003 |
| PCA on 1-10 years (found after seeing the results) | 0.93 | 0.74 | 0.59 | <0.001 |

Source: [M5 report](outputs/m5/robustness_report.md).

- **The planned verdict holds in 5 of 7 planned variants, but each time it rests on one ill-conditioned period.**
    - Four of them still contain the mid-2019 period.
    - The 750-day window has its own bad period in March-June 2020, with an H3 weight of 15.
    - On the same days as the 750-day run, the planned design has H3 beating duration (0.60 against 0.95).

![Leftover risk by variant](outputs/m5/hedge_sd_by_variant.png)

**On real bond yields (F2, a 5-year bond):**

| Run | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| F2 bond yields | 5.29 | 1.12 | 1.23 | 6.75 |
| F17 at the same four maturities | 5.26 | 1.06 | 1.01 | 1.12 |

Source: [M5 report](outputs/m5/robustness_report.md).

- **No factor hedge is significantly better than duration on F2** (p = 0.189 for H3, 0.162 for H2).
- **H3 blew up in two 2021 periods.** Its equations were ill-conditioned there, with condition numbers 51 and 8.9 against 3.0 or less elsewhere.
    - Both estimation windows ended inside YCC, when the 3-year yield barely moved.
    - Without those periods (ex post) H3 leaves 0.82 against duration's 0.91 (p = 0.48).
- **The limit of this check.** With only four maturities, H3 shows no edge on F17 either. So this check cannot test the 40-maturity result above.

## What does not hold up

- **Significance.** The curve's forecasting gain over "no change" is large but not significant at 5% over the full sample. The one significant row (excluding Tightening) was chosen after seeing the results.
- **The OIS comparison.** It cannot show that bond prices carry the same information as OIS, because the curve's short end is fitted to OIS.
- **The hedging headline turns on one period.** "H3 does not beat duration" rests on one ill-conditioned period in 2019, or one in 2020 for the 750-day window.
    - The opposite verdict comes from a 250-day window, a 5-year target or a 1-10 year PCA.
    - The 1-10 year PCA was found after seeing the results, so it is not claimed as the answer.
- **Smoothness may flatter H3.** The fitted curve is smooth, which may flatter factor hedges at 40 maturities. The bond-yield check could not settle this, because with four maturities neither curve shows a factor edge.
- **Hindsight in the data cleaning.** The two days dropped as fitting errors were identified after looking at the data.

## Limitations

- **Revised data.** F17 is the RBA's fitted, revised curve, not prices a trader saw in real time. Its inputs changed in Dec 2023.
- **Small sample.**
    - 26 decisions in four clusters.
    - About 18 independent 6-month forecasts.
    - 71 overlapping months with OIS (OIS ends Dec 2022).
    - Every p-value is rough, and no adjustment is made for testing many variants.
- **Term premium.** Forward rates mix expectations with a term premium, so a forecast error is not purely a forecasting mistake.
- **Hedging simplifications.** Constant-maturity points, no carry or roll-down, and first-order (DV01) P&L. F2 yields are coupon-bond yields, not zeros.
- **No futures.** There is no free daily history of ASX bond futures, so nothing is said about futures basis or trading.

## How to reproduce

From the repository root, create the environment once:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

**Download the data first.** Save each file listed in `data/MANIFEST.csv` into `data/raw/` under the name in its first column. Without F1 and F2, M1 reports checks 11-12 as NOT RUN and M5 stops with a message.

Then:

```bash
.venv/Scripts/python src/m1_validate.py
.venv/Scripts/python src/m2_expectations.py
.venv/Scripts/python src/m3_factors.py
.venv/Scripts/python src/m4_hedging.py
.venv/Scripts/python src/m5_robustness.py
.venv/Scripts/python -m pytest tests
```

**Where each step is documented:**

- Each script writes its own report in `outputs/mN/`.
- Each report opens with its answer.
- The full CSV tables behind the reports are not stored in the repo; running the scripts writes them to `outputs/mN/`.

Source: RBA 2026 (tables F1, F2, F17, A2).
