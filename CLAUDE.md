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

## Environment
- Windows with Git Bash. Quote any path containing spaces; use forward slashes.
- Run scripts from the repo root, e.g. `python code/labelling.py --index GSPC`.
- `FRED_API_KEY` env var is needed only by data_pipeline.py. Never commit keys.
- The repo is public on GitHub.

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

## Open items
- Confirm the six crisis events in CRISIS_EVENTS against the supervisor's email.
- HMM is fit on the full sample: state as a limitation or refit on 2000-2015.
- Next: feature engineering (domestic, foreign, macro, technical groups; build
  Domestic-Only and Cross-Market feature sets), then modelling.

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
