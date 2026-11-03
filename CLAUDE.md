# FYP: Explainable AI for Cross-Border Stock Anomaly Detection

## Goal
Test whether lagged cross-market features (S&P 500, Nikkei 225, FTSE 100) improve
anomaly detection compared with domestic-only features, and use SHAP/LIME to show
which foreign signals and lags matter. Stretch goal: 60-day rolling SHAP share of
foreign features plotted against VIX (2019-2024).

## Compute
- The school GPU is not being used. All training runs locally on CPU.
  Do not add CUDA/GPU code paths or assume GPU availability.

## Data rules
- The data date range (2000-01-01 to 2024-12-31) must never be changed.
- No future information may enter any label, feature, threshold or standardisation
  statistic. Fits and calibration use data up to 2015, or trailing data only.
- Do not re-run data_pipeline.py unless explicitly asked. The panel in use is
  data/processed/panel_full_2026-10-27.csv: identical to panel_full.csv (2026-09-16,
  kept unchanged) except EPU_US, which is misaligned there (load_epu did not sort the
  US file; wrong on 6094 of 6288 days). All configs and run_all.sh use the new file.
- TED is excluded from all features. FRED discontinued TEDRATE on 2022-01-21 and
  the forward-fill leaves it stale at 0.09 for the remaining 738 rows. The exclusion
  is listed in configs/features.yaml and asserted in code/features.py. Its planned
  replacement, HY OAS (BAMLH0A0HYM2), is on hold: since April 2026 FRED serves only a
  rolling 3-year window (first observation 2023-10-27). Never backfill, interpolate or
  forward-fill missing years. Check any candidate with code/check_fred_coverage.py.
- Deviations from the brief, approved by the user when the feature build was planned:
  1. EPU lag: EPU for month M is published in early month M+1, so features use the
     value for month M-1, then lag 1 day (`epu.publication_lag_months: 1`). The brief
     specified a lag-1 day only, which would leak the rest of the month.
  2. Volume: lag-1 log volume is replaced by lag-1 log volume minus its trailing
     252-day mean, computed from data up to t-1 (`volume.normalise_trailing_window`).
     Raw log volume is non-stationary.
  3. MACD signal is divided by close, and Bollinger width is divided by the middle
     band (`technical.macd_normalise_by_close`, `bollinger_normalise_by_middle`).
     Both are scale-free; raw values are in index points, which range from about
     1,400 to 6,000 for the S&P 500.

## Environment
- Windows with Git Bash. Quote any path containing spaces; use forward slashes.
- Run scripts from the repo root, e.g. `python code/labelling.py --index GSPC`.
- `FRED_API_KEY` env var is needed only by data_pipeline.py. Never commit keys.
- The repo is public on GitHub.

## Commits (automated)
- Laptop: Task Scheduler task "FYP weekly commit" runs scripts/weekly_commit.sh every
  Sunday 21:00 SGT (a missed run starts after the next sign-in). It commits all changes,
  rebases onto origin/main and pushes. Log: ~/fyp_weekly_commit.log.
- Cloud sessions (claude.ai web/app): the Stop hook .claude/hooks/cloud_autocommit.sh
  commits after every turn, since idle cloud VMs are reclaimed, and pushes to the session's
  claude/ branch (claude/autosave-<session> when on another branch). It never pushes to main.
- Routine "FYP weekly cloud commit" is disabled (2026-11-03). Cloud branches reach main
  only after the user reviews them and merges them as a pull request.
- Cloud sessions have no data: data/raw/ and data/processed/*.csv are gitignored.

## Layout
code/, configs/, data/raw, data/processed, results/, logs/, thesis_tables/, data_readme.txt

## Done
- code/data_pipeline.py builds the panel (6288 rows, 18 columns: close, volume, logret,
  rv21 for GSPC/N225/FTSE; VIX, USDJPY, DGS10, TED; EPU_US, EPU_JP); `--out` sets the
  output path. Kaggle cross-check showed 0 mismatches for all three indices.
- code/labelling.py + code/regimes.py (configs/labelling.yaml): three causal schemes,
  evaluated on the same days: rolling z-score baseline (2.5); regime-dependent threshold
  (2.5/3.0 by the filtered regime on day t); regime-standardised (last 252 same-regime
  returns before t, regime predicted from data up to t-1, at least 60, threshold 2.5).
  HMM on rv21 fit on 2000-2015 and frozen; explicit forward recursion with a
  perturbation test. The full-sample Viterbi HMM survives only as label hmm_fullsample
  (sensitivity). Undefined days get no label. Writes results/labels_{IDX}.csv,
  labelling_evaluation_{IDX}.csv, hmm_sensitivity_{IDX}.csv, hmm_params_{IDX}.json.
- code/features.py (configs/features.yaml): lag-0 foreign features only where the
  foreign session closes before the target's on every date (close times with DST):
  GSPC gets N225 and FTSE, FTSE gets N225, N225 none. Sets: domestic_only (16),
  cross_market (28, lag 1), cross_market_lag0 (32 GSPC, 30 FTSE, = cross_market for
  N225), ablation_foreign (24), ablation_macro (20), rv_only (5). The no-lookahead test
  also checks the same-day rules. Rows: GSPC 6036, FTSE 6035, N225 5427 (from
  2003-06-10, because N225 volume is zero before).
- code/baseline_models.py (configs/baseline_models.yaml): XGBoost and L2 logistic
  regression, all three targets, three label sets x six feature sets plus the
  hmm_fullsample sensitivity, 5 seeds (42-46); block-bootstrap 95% CIs (21-day blocks,
  anomaly clusters kept whole); saves seed-mean test predictions.
- code/dm_tests.py (configs/dm_tests.yaml; Brier loss, Newey-West, HLN correction),
  code/figures.py (configs/figures.yaml; data panel table, S&P 500 regime figure),
  code/report_tables.py (results/tables/, one settings file each in thesis_tables/).
  Earlier results are in results/archive/, logs/archive/, thesis_tables/archive/.
- code/train.py and code/test.py are stubs that import modules not yet written
  (data_utils, model_factory, metrics, explain).
- Known input issues, not yet addressed: N225 volume is zero on 615 days, mostly
  2000-2002; N225 and FTSE forward-filled holidays create zero-return rows (5.9% and
  2.0% of N225/FTSE days). Test-period EPU_US and DGS10 are outside the train range
  (EPU_US max 504 vs 284 in train; DGS10 min 0.52 vs 1.43). The panel ends 2024-12-30
  (yfinance's end date is exclusive).

## Supervisor decisions (reply to the October update)
- TED excluded (agreed). A credit spread replaces it once a series covering 2000-2024
  is chosen (HY OAS fails the coverage check; see Data rules).
- Causal HMM for all labels: fit on 2000-2015 only, frozen, filtered probabilities
  applied forward; the full-sample fit is only a sensitivity analysis.
- Crisis events: Dot-com bust, Lehman 2008, European Debt Crisis, Flash Crash 2010,
  COVID March 2020, 2022 rate-hike cycle. The date windows are the student's choice
  (configs/labelling.yaml).
- The regime-standardised labelling scheme is the core novelty.
- Same-day foreign features only when the foreign session closes before the target's;
  the lag-1-only set stays as a separate variant.
- Report block-bootstrap 95% CIs (block length stated, clusters intact), the prevalence
  baseline for PR-AUC, and DM tests with the loss and small-sample correction stated.

## Open items
- Credit spread for the macro group: supervisor to choose (candidate BAA10Y, on FRED
  1986-2026); then rerun the feature sets that contain macro features.
- Confirm the six crisis date windows.
- Decide whether panel_full_2026-10-27.csv replaces panel_full.csv.
- Results to the supervisor by about 3 Nov; interim report due 10 Nov.

## Planned modelling
Walk-forward split: train 2000-2015, validation 2016-2018, test 2019-2024.
Models: XGBoost, L2 Logistic Regression, Isolation Forest. SMOTE on train only.
Metrics: PR-AUC, F1, MCC. Diebold-Mariano tests with exact statistics and p-values.
Benchmark against Pandey & Sharma RF-LSTM in one consolidated table.

## Supervisor rules
- Commit to git at least weekly. Hyperparameters in YAML configs, not hard-coded.
- One settings file per thesis table/figure in thesis_tables/.
- Figures: no colour, no grids, at most 6 ticks per axis, caption instead of title.
- Report mean and standard deviation over multiple seeds. Never fake results.
- Prioritise running experiments over planning and writing.
