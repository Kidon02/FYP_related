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
- Do not re-run data_pipeline.py unless explicitly asked; panel_full.csv is fixed.
- TED is excluded from all features. FRED discontinued TEDRATE on 2022-01-21 and
  the forward-fill leaves it stale at 0.09 for the remaining 738 rows. The exclusion
  is listed in configs/features.yaml and asserted in code/features.py.
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
- Routine "FYP weekly cloud commit" (claude.ai/code/routines) merges unmerged claude/*
  branches into main every Sunday 22:07 SGT. Delete a cloud branch on GitHub before then
  to keep it out of main.
- Cloud sessions have no data: data/raw/ and data/processed/*.csv are gitignored.

## Layout
code/, configs/, data/raw, data/processed, results/, logs/, thesis_tables/, data_readme.txt

## Done
- code/data_pipeline.py builds data/processed/panel_full.csv (6288 rows, 18 columns:
  close, volume, logret, rv21 for GSPC/N225/FTSE; VIX, USDJPY, DGS10, TED; EPU_US,
  EPU_JP). Kaggle cross-check showed 0 mismatches for all three indices.
- code/labelling.py: baseline rolling-252 z-score (threshold 2.5) vs 2-state HMM
  regime labels (default thresholds: low-vol 2.5, high-vol 3.0, set with
  --regime-thresholds). Writes results/labelling_evaluation_{IDX}.csv and
  results/labels_{IDX}.csv.
- code/train.py and code/test.py are stubs that import modules not yet written
  (data_utils, model_factory, metrics, explain).
- code/features.py (`python code/features.py --index GSPC`): builds
  data/processed/features_{IDX}.csv (wide) and feature_groups_{IDX}.csv. Configured
  in configs/features.yaml. Runs a no-lookahead test (perturb rows from T onward,
  check features at or before T are unchanged) and asserts no label-derived names.
  GSPC: 6036 rows, Domestic-Only 16 features, Cross-Market 28. Constant features
  are dropped and printed. FTSE volume is not constant, and GSPC has one zero-volume
  day (2023-05-24, treated as missing and carried forward).
- code/baseline_models.py (`python code/baseline_models.py --index GSPC`): XGBoost and
  L2 logistic regression on both feature sets and both label sets, 5 seeds (42-46).
  Configured in configs/baseline_models.yaml. Writes results/baseline_models_{IDX}.csv,
  logs/baseline_models_{IDX}_grid.csv and thesis_tables/baseline_models_{IDX}_settings.txt.
  Grid and threshold use validation only. Asserts that labels reproduce from labelling.py.
- Known input issues, not yet addressed: N225 volume is zero on 615 days, mostly
  2000-2002; N225 and FTSE forward-filled holidays create zero-return rows (5.9% and
  2.0% of N225/FTSE days). Test-period EPU_US and DGS10 are outside the train range
  (EPU_US max 504 vs 284 in train; DGS10 min 0.52 vs 1.43).
- labelling.py has a known warm-up issue: rows where the z-score is undefined are
  counted as normal. features.py drops them, but labelling.py and its results have
  not been changed yet.

## Open items
- Confirm the six crisis events in CRISIS_EVENTS against the supervisor's email.
- HMM is fit on the full sample: state as a limitation or refit on 2000-2015.
- Next: N225 and FTSE runs of features.py and baseline_models.py, then decide how to
  handle the labelling warm-up and the out-of-range macro features.

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
