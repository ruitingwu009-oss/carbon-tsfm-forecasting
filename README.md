# Do time series foundation models forecast carbon prices? Replication code

Code and intermediate results for the paper *Do Time Series Foundation Models Forecast Carbon Prices? Leakage-Free Evidence from the EU ETS and China’s Carbon Markets* (Wu, Jamaludin and Selvaraj, submitted to *Energies*).

## 1. Data

The raw price series are not redistributed because of the terms of the data providers. To rebuild the input file `carbon_prices_clean.xlsx`:

| Sheet | Market | Source | Field | Sample |
| --- | --- | --- | --- | --- |
| `EUA` | EU ETS | Investing.com, “Carbon Emissions Futures” (https://www.investing.com/commodities/carbon-emissions-historical-data; currently symbol CFI2Z6, the ICE EUA December 2026 contract; continuous front-December series as published, no back-adjustment) | field “Price” of the Investing.com download, i.e. the daily settlement price (stored as column `close`) | 2 Jan 2018 – 25 Sep 2026 (earlier rows are dropped by the code) |
| `CEA_national` | China national ETS | Shanghai Environment and Energy Exchange, daily market report | composite closing price (COMCEA) | 16 Jul 2021 – 24 Sep 2026 |
| `Hubei` | Hubei pilot | Hubei Emission Exchange (China Hubei Emission Exchange website) | daily closing price of HBEA | 2 Apr 2014 – 5 Aug 2026 |
| `Guangdong` | Guangdong pilot | China Emissions Exchange (Guangzhou) | daily closing price of GDEA | 19 Dec 2013 – 5 Aug 2026 |

Investing.com does not document the roll dates of its continuous series. Script 19 lists the largest daily return in each December: the only roll within the EU test period (December 2025) produced at most a 1.9% return (15 December 2025), below all jump thresholds; the December 2024 roll (4.2% on 19 December 2024) precedes the test period.

Each sheet has two columns, `date` and `close`, sorted by date. Non-trading days are dropped, so each series is indexed by trading day. One missing national-market day (13 Oct 2025) was filled from an independent compilation of exchange data and checked against the next day’s reported previous close.

## 2. Forecast origins

`origins/forecast_origins_<market>.csv` lists, for every market, the index (row number in the sheet, 0-based) and date of each forecast origin. The test period is the last 20% of observations; origin t uses observations t−255 … t.

## 3. Scripts

| Script | Purpose | Runs on |
| --- | --- | --- |
| `code/01_main_experiment.py` | All models, all markets (L = 256): point forecasts, DM tests, sub-period analysis, Table 1 statistics; saves `forecasts_<market>.pkl` | Google Colab, T4 GPU (~1 h) |
| `code/02_quantiles_and_environment.py` | Per-origin quantile forecasts of both foundation models and the empirical random walk; package versions and checkpoint revisions | Google Colab, T4 GPU (~15 min) |
| `code/03_leakage_variants.py` | VMD-AR with full-series, expanding-history and window-wise decomposition (Table A9) | CPU (~30 min) |
| `code/04_jump_regressions.py` | Jump regressions (Table 7), including the split into jumps reversed and not reversed on the next day | CPU |
| `code/05_jump_sensitivity.py` | Jump counts and threshold/window sensitivity of the jump regressions (Table A5); the reported intervals for Tables 6 and A4 come from script 14 | CPU |
| `code/06_dm_alternative_losses.py`, `06b_dm_bartlett.py` | DM tests under squared, absolute and percentage-error losses (Table A3) | CPU |
| `code/07_common_period.py`, `08_post_release.py` | Common calendar period and evaluation of forecasts made from 31 October 2025, after the publication of the model weights (Table A8) | CPU |
| `code/09_subperiod_dynamics.py` | Price dynamics by sub-period (Section 5.2) | CPU |
| `code/10_coverage_by_horizon.py` | Interval coverage and width by horizon with Newey–West intervals (Tables A6 and A7) | CPU |
| `code/11_post_release_holm.py` | Repeats the post-publication evaluation of script 08 (same 31 October 2025 cut-off) and adds the Holm-adjusted DM results | CPU |
| `code/12_interval_score_alignment.py` | Interval scores (Table 4) and the forecast-alignment audit (`alignment_audit.csv`) | CPU |
| `code/13_mlp_chronological_validation.py` | MLP refitted with a chronological validation set (Appendix B) | CPU |
| `code/14_reversal_ratio_month_bootstrap.py` | Next-day reversal ratios with month-clustered bootstrap intervals (Tables 6 and A4) | CPU |
| `code/15_mlp_chronological_main.py` | MLP with chronological early-stopping validation, used in the main tables; writes `results5/forecasts_<market>.pkl` | CPU |
| `code/16_metrics_dm_holm_both_estimators.py` | Table 2, Table A1 and Holm-adjusted DM tests with rectangular and Bartlett weights (Section 4.2, Table A3) | CPU |
| `code/17_common_period_and_post_release.py` | Common calendar period with origins and targets restricted to the window, origins by horizon, post-publication sample (Table A8) | CPU |
| `code/18_pooled_regression_driscoll_kraay.py` | Pooled jump regressions with within-market Newey–West and Driscoll–Kraay standard errors (Table 7, Table A5) | CPU |
| `code/19_bds_filtered_and_eua_rolls.py` | BDS test on raw and autoregression-filtered returns; EUA returns around December contract rolls (Section 3.1) | CPU |

Run all scripts from the `code/` folder with `carbon_prices_clean.xlsx` placed in the same folder. Scripts 03–10 read the outputs of scripts 01 and 02 from `results3/` (point forecasts) and `results4/` (quantile forecasts); the files in `results/` can be copied into these two folders to reproduce every table without rerunning the foundation models. Scripts 16–17 read `results5/` (written by script 15; a copy is in `results/results5/`).

## 4. Environment and reproducibility

- `requirements.txt` pins the package versions used for the paper.
- Model checkpoints: `amazon/chronos-2` at Hugging Face commit `29ec3766d36d6f73f0696f85560a422f50e8498c` and `google/timesfm-2.5-200m-pytorch` at commit `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` (also in `results/environment.json`). The Chronos-2 weights were added on 30 October 2025; all later commits to that repository change only `README.md`. To load exactly these revisions, pass `revision=<commit hash>` when downloading the checkpoints from Hugging Face.

- Python 3, Google Colab with an NVIDIA T4 GPU for scripts 01–02. Package versions and Hugging Face commit hashes of the checkpoints are recorded in `results/environment.json` (Appendix B of the paper).
- Foundation models: `amazon/chronos-2` (package `chronos-forecasting`) and `google/timesfm-2.5-200m-pytorch` (package `timesfm`), zero-shot, no fine-tuning. Neither model draws random samples; two independent runs produced identical forecasts.
- Random seeds: `numpy.random.seed(42)`; `random_state=0` for the GBDT and MLP models.
- MLP in the main tables (Table 2, Table A1): early stopping on a chronological validation set (last 10% of training origins, separated by a 60-origin gap), script 15. The random validation split of the originally submitted version (below) is retained for comparison in Appendix B.
- MLP of the originally submitted version (script 01; Appendix B): scikit-learn (validation set drawn at random by scikit-learn; script 13 compares it with the chronological validation set) `MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=0)`, i.e. 10% of training origins held out for validation, stopping after 10 epochs without an improvement of at least 1e-4 in the validation score.

## 5. Saved results

`results/` contains the saved point forecasts (`forecasts_<market>.pkl`), quantile forecasts (`quantiles_<market>.pkl`) and the CSV files behind every table, so that all tables can be reproduced without rerunning the foundation models.

## 6. Running scripts 01 and 02 in Google Colab

1. Open a new notebook at colab.research.google.com and select *Runtime → Change runtime type → T4 GPU*.
2. Upload `requirements.txt`, the script and `carbon_prices_clean.xlsx`:
   ```
   from google.colab import files
   files.upload()
   ```
3. Install the pinned package versions used for the paper:
   ```
   !pip -q install -r requirements.txt
   ```
4. Run the script, e.g. `exec(open('01_main_experiment.py').read())`. Script 01 writes `results3.zip` (about 1 hour); script 02 writes `results4.zip` (about 15 minutes). Both files are downloaded automatically.

## 7. Versions

The version of the code and results corresponding to the published tables and figures is tagged as release `v1.1` (https://github.com/ruitingwu009-oss/carbon-tsfm-forecasting/releases/tag/v1.1).
