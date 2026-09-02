# Deciphering International Market Connections: Explainable AI for Cross-Border Stock Anomaly Detection

Final year project. Detects abnormal return activity in the S&P 500, Nikkei 225,
and FTSE 100, tests whether cross-market lagged features improve detection versus
domestic-only features, and applies SHAP/LIME to interpret the results.

## Environment

- Python 3.10+
- Key libraries: `yfinance`, `fredapi`, `pandas`, `numpy`, `scikit-learn`, `xgboost`,
  `hmmlearn`, `imbalanced-learn` (SMOTE), `shap`, `lime`, `matplotlib`
- No GPU required — all models (XGBoost, Logistic Regression, Isolation Forest) run
  on CPU. A modern laptop is sufficient; full walk-forward training on ~24 years of
  daily data across 3 indices should take well under an hour.

Install dependencies:

```bash
pip install -r requirements.txt
```

You will need a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
set as an environment variable:

```bash
export FRED_API_KEY=your_key_here
```

## Data

See `data_readme.txt` for dataset sources and expected file locations. Datasets are
not redistributed in this repo — the pipeline retrieves them directly from
yfinance/FRED, or you can download the Kaggle CSVs and place them as described there.

## How to train

```bash
python code/train.py --config configs/xgboost_crossmarket.yaml
```

Swap `--config` to any file in `configs/` to train a different model / feature-set
combination (e.g. `logreg_domestic_only.yaml`, `isolation_forest_crossmarket.yaml`).
Each config specifies the model type, feature set (`domestic_only` or
`cross_market`), labelling strategy (`baseline` or `regime_hmm`), and hyperparameters.
Trained models are written to `results/checkpoints/`.

## How to test / evaluate

```bash
python code/test.py --config configs/xgboost_crossmarket.yaml --checkpoint results/checkpoints/<name>.pkl
```

This runs the held-out 2019–2024 test window and writes PR-AUC, F1, and MCC to
`logs/`, plus (for XGBoost) triggers the SHAP explainability outputs used in the
XAI notebook.

## Approximate runtime

- Data retrieval + preprocessing: ~5–10 minutes (network-bound)
- Training one model/feature-set combination: 1–5 minutes on CPU
- Full SHAP analysis (global + local + rolling): ~10–20 minutes

## Reproducibility notes

Every table/figure in the dissertation has a matching settings file in
`thesis_tables/` recording the exact config, seed, and checkpoint used, so any
result can be regenerated. See `Project-Code and Reproducibility Requirements`
from the supervisor's brief for the full standard this repo follows.
