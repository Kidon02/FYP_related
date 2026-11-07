# FYP: Explainable AI for Cross-Border Stock Anomaly Detection

Read STATUS.md first.

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
- Do not re-run data_pipeline.py unless explicitly asked (never with --refresh). The panel
  in use is data/processed/panel_full_2026-11-07.csv: panel_full_2026-10-27.csv plus BAA10Y
  (its 18 columns verified identical). The 2026-10-27 panel equals panel_full.csv
  (2026-09-16, kept unchanged) except EPU_US, which is misaligned there (load_epu did not
  sort the US file; wrong on 6094 of 6288 days). All configs and run_all.sh use the
  2026-11-07 file. `--fetch-series NAME` downloads one FRED series only.
- TED is excluded from all features. FRED discontinued TEDRATE on 2022-01-21 and
  the forward-fill leaves it stale at 0.09 for the remaining 738 rows. The exclusion
  is listed in configs/features.yaml and asserted in code/features.py. Its planned
  replacement is BAA10Y (approved by the supervisor, Nov 2026; fetched 2026-11-07, full
  2000-2024 coverage). HY OAS (BAMLH0A0HYM2) was rejected: since April 2026 FRED serves only
  a rolling 3-year window. Never backfill, interpolate or forward-fill missing years. Check
  any candidate with code/check_fred_coverage.py.
- Macro features are stationary (`macro.transform: stationary`, Nov 2026, supervisor):
  DGS10 and BAA10Y daily change, USDJPY log change, EPU log z-score over a trailing 252-day
  window. `levels` keeps the old definitions. Rows: GSPC 6016, FTSE 6016, N225 5427.
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
- The repo must be private on GitHub: it contains the input data (Yahoo, FRED, EPU files).

## Commits (automated)
- Laptop: Task Scheduler task "FYP weekly commit" runs scripts/weekly_commit.sh every
  Sunday 21:00 SGT (a missed run starts after the next sign-in). It commits all changes,
  rebases onto origin/main and pushes. Log: ~/fyp_weekly_commit.log.
- Cloud sessions (claude.ai web/app): the Stop hook .claude/hooks/cloud_autocommit.sh
  commits after every turn, since idle cloud VMs are reclaimed, and pushes to the session's
  claude/ branch (claude/autosave-<session> when on another branch). It never pushes to main.
- Routine "FYP weekly cloud commit" is disabled (2026-11-03). Cloud branches reach main
  only after the user reviews them and merges them as a pull request.
- Input data are tracked (force-added despite .gitignore): data/raw/ (except the Kaggle file) and
  data/processed/panel_full_2026-10-27.csv, so cloud sessions can run the pipeline from cache.

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
- code/train.py (labels, features, baseline models, Isolation Forest) and code/test.py (DM tests,
  significance, SHAP, figures, report tables) are the two entry points; run_all.sh runs both.
- code/isolation_forest.py (configs/isolation_forest.yaml): unsupervised benchmark from the brief, fit
  on 2000-2015 features only; same rows, seeds, selection rules and bootstrap resamples as the
  supervised models. results/isolation_forest_{IDX}.csv.
- code/significance.py: Wilcoxon signed-rank (Holm) and Friedman tests over 9 blocks (3 targets x 3
  label sets) of seed-mean test PR-AUC, as the supervisor requires. results/tables/table_significance.csv.
- code/shap_analysis.py (configs/shap.yaml): global TreeSHAP for XGBoost on the foreign-containing
  feature sets, with validation-chosen hyperparameters; checked against XGBoost pred_contribs.
  results/shap_group_shares.csv, results/figures/fig_shap_top_GSPC.*.
- Cross-machine reproducibility: labels and logistic regression identical (Linux vs Windows);
  XGBoost PR-AUC differs by about 0.004 (multithreaded hist summation order).
- Known input issues, not yet addressed: N225 volume is zero on 615 days, mostly
  2000-2002; N225 and FTSE forward-filled holidays create zero-return rows (5.9% and
  2.0% of N225/FTSE days). Test-period EPU_US and DGS10 are outside the train range
  (EPU_US max 504 vs 284 in train; DGS10 min 0.52 vs 1.43). The panel ends 2024-12-30
  (yfinance's end date is exclusive).

## Interpretation notes
- Same-day foreign features (cross_market_lag0) are nowcasting inputs, available a few
  hours before the target's close; they are not a next-day early warning.
- Causal vs full-sample HMM matters: S&P 500 XGBoost Domestic-Only PR-AUC is 0.170 with
  the causal HMM vs 0.268 with the full-sample HMM (3 Nov 2026 run, regime-dependent
  threshold labels). After the Part A rerun (7 Nov, 6016 rows) it is 0.157 vs 0.268.
- Reference [16] is Mostafavi and Hooman (2025).
- Attachments sent to the supervisor always get a distinct dated or versioned file name.

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
- Confirm the six crisis date windows.
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
