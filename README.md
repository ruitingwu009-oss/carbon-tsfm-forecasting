# Do time series foundation models forecast carbon prices? Replication code

Code and intermediate results for the paper *Do Time Series Foundation Models Forecast Carbon Prices? Leakage-Free Evidence from the EU ETS and China’s Carbon Markets* (Wu, Jamaludin and Selvaraj, submitted to *Energies*).

## 1. Data

The raw price series are not redistributed because of the terms of the data providers. To rebuild the input file `carbon_prices_clean.xlsx`:

| Sheet | Market | Source | Field | Sample |
| --- | --- | --- | --- | --- |
| `EUA` | EU ETS | Investing.com, “Carbon Emissions Futures” (ICE EUA, nearest December contract, continuous series as published, no back-adjustment) | daily settlement price (column `close`) | 2 Jan 2018 – 25 Sep 2026 (earlier rows are dropped by the code) |
| `CEA_national` | China national ETS | Shanghai Environment and Energy Exchange, daily market report | composite closing price (COMCEA) | 16 Jul 2021 – 24 Sep 2026 |
| `Hubei` | Hubei pilot | Hubei Emission Exchange (China Hubei Emission Exchange website) | daily closing price of HBEA | 2 Apr 2014 – 5 Aug 2026 |
| `Guangdong` | Guangdong pilot | China Emissions Exchange (Guangzhou) | daily closing price of GDEA | 19 Dec 2013 – 5 Aug 2026 |

Each sheet has two columns, `date` and `close`, sorted by date. Non-trading days are dropped, so each series is indexed by trading day. One missing national-market day (13 Oct 2025) was filled from an independent compilation of exchange data and checked against the next day’s reported previous close.

## 2. Forecast origins

`origins/forecast_origins_<market>.csv` lists, for every market, the index (row number in the sheet, 0-based) and date of each forecast origin. The test period is the last 20% of observations; origin t uses observations t−255 … t.

## 3. Scripts

| Script | Purpose | Runs on |
| --- | --- | --- |
| `code/01_main_experiment.py` | All models, all markets (L = 256): point forecasts, DM tests, sub-period analysis, Table 1 statistics; saves `forecasts_<market>.pkl` | Google Colab, T4 GPU (~1 h) |
| `code/02_quantiles_and_environment.py` | Per-origin quantile forecasts of both foundation models and the empirical random walk; package versions and checkpoint revisions | Google Colab, T4 GPU (~15 min) |
| `code/03_leakage_variants.py` | VMD-AR with full-series, expanding-history and window-wise decomposition | CPU (~30 min) |
| `code/04_jump_regressions.py` | Jump regressions (Table 7), including the reversed/persistent split | CPU |
| `code/05_jump_sensitivity.py` | Jump counts, bootstrap intervals and threshold/window sensitivity (Tables 6, A4, A5) | CPU |
| `code/06_dm_alternative_losses.py`, `06b_dm_bartlett.py` | DM tests under squared, absolute and percentage-error losses (Table A3) | CPU |
| `code/07_common_period.py`, `08_post_release.py` | Common calendar period and post-release evaluation (Table A7) | CPU |
| `code/09_subperiod_dynamics.py` | Price dynamics by sub-period (Section 5.2) | CPU |
| `code/10_coverage_by_horizon.py` | Interval coverage and width by horizon with Newey–West intervals (Table A6) | CPU |
| `code/11_post_release_holm.py` | Forecasts made after the publication of the model weights; Holm-adjusted DM results | CPU |
| `code/12_cluster_bootstrap_interval_score_alignment.py` | Month-clustered bootstrap of reversal ratios (Table 6), interval scores (Table 4) and the forecast-alignment audit (`alignment_audit.csv`) | CPU |
| `code/13_mlp_chronological_validation.py` | MLP refitted with a chronological validation set (Appendix B) | CPU |

Run all scripts from the `code/` folder with `carbon_prices_clean.xlsx` placed in the same folder. Scripts 03–10 read the outputs of scripts 01 and 02 from `results3/` (point forecasts) and `results4/` (quantile forecasts); the files in `results/` can be copied into these two folders to reproduce every table without rerunning the foundation models.

## 4. Environment and reproducibility

- Python 3, Google Colab with an NVIDIA T4 GPU for scripts 01–02. Package versions and Hugging Face commit hashes of the checkpoints are recorded in `results/environment.json` (Appendix B of the paper).
- Foundation models: `amazon/chronos-2` (package `chronos-forecasting`) and `google/timesfm-2.5-200m-pytorch` (package `timesfm`), zero-shot, no fine-tuning. Neither model draws random samples; two independent runs produced identical forecasts.
- Random seeds: `numpy.random.seed(42)`; `random_state=0` for the GBDT and MLP models.
- MLP: scikit-learn (validation set drawn at random by scikit-learn; script 13 checks a chronological validation set) `MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=0)`, i.e. 10% of training origins held out for validation, stopping after 10 epochs without an improvement of at least 1e-4 in the validation score.

## 5. Saved results

`results/` contains the saved point forecasts (`forecasts_<market>.pkl`), quantile forecasts (`quantiles_<market>.pkl`) and the CSV files behind every table, so that all tables can be reproduced without rerunning the foundation models.

## 6. Running scripts 01 and 02 in Google Colab

1. Open a new notebook at colab.research.google.com and select *Runtime → Change runtime type → T4 GPU*.
2. Install the packages:
   ```
   !pip -q install "chronos-forecasting>=2.0" statsmodels openpyxl huggingface_hub
   !pip -q install "timesfm[torch]"
   ```
3. Upload the script and `carbon_prices_clean.xlsx`:
   ```
   from google.colab import files
   files.upload()
   ```
4. Run the script, e.g. `exec(open('01_main_experiment.py').read())`. Script 01 writes `results3.zip` (about 1 hour); script 02 writes `results4.zip` (about 15 minutes). Both files are downloaded automatically.
