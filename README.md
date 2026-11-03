# Deciphering International Market Connections: Explainable AI for Cross-Border Stock Anomaly Detection

Final year project. Labels abnormal daily returns in the S&P 500, Nikkei 225 and FTSE 100
(2000-2024), tests whether foreign-market features improve anomaly detection compared with
domestic-only features, and explains the models with SHAP.

## Environment

- Python 3.12; packages pinned in `requirements.txt` (`pip install -r requirements.txt`).
- CPU only. No GPU and no API keys are needed: all inputs are in `data/` and nothing is downloaded
  (`code/data_pipeline.py` downloads only with `--refresh`, which needs `FRED_API_KEY`).
- Tested on Windows 11 (Git Bash) and Ubuntu 24.04.

## Data

See `data_readme.txt` for sources, identifiers and file locations. The panel used by every
step is `data/processed/panel_full_2026-10-27.csv` (6,288 trading days, 2000-01-03 to 2024-12-30).

## How to train

```bash
python code/train.py              # all three targets
python code/train.py --index GSPC # one target
```

Runs, in order: labels (`labelling.py`, causal HMM fit on 2000-2015), features (`features.py`,
with a no-lookahead test), supervised models (`baseline_models.py`: XGBoost and L2 logistic
regression, 5 seeds, block-bootstrap CIs) and the unsupervised benchmark (`isolation_forest.py`).
Split: train 2000-2015, validation 2016-2018 (grid search and thresholds), test 2019-2024 (scored once).

## How to test / evaluate

```bash
python code/test.py
```

Runs Diebold-Mariano tests (`dm_tests.py`), Wilcoxon and Friedman tests (`significance.py`),
global SHAP (`shap_analysis.py`), figures (`figures.py`) and the report tables (`report_tables.py`).
Tables are written to `results/tables/`, figures to `results/figures/`, and one settings file per
table or figure to `thesis_tables/`.

`bash run_all.sh` runs training and evaluation end to end.

## Configuration

Every setting is in `configs/`: `labelling.yaml`, `features.yaml`, `baseline_models.yaml`,
`isolation_forest.yaml`, `dm_tests.yaml`, `shap.yaml`, `figures.yaml`.

## Approximate runtime (single CPU core)

Labels and features: about 1 minute. Supervised models: about 5-10 minutes per target.
Isolation Forest: under 1 minute per target. Evaluation (`test.py`): about 2 minutes.

## Reproducibility

Seeds 42-46 throughout. Re-running on the same machine reproduces every result file exactly.
Across machines, labels and logistic-regression results are identical; XGBoost results can
differ slightly (about 0.004 in PR-AUC) because of floating-point summation order in its
multithreaded histogram method.
