"""
Rank-based significance tests across datasets (Wilcoxon signed-rank and Friedman),
as the supervisor requires, complementing the per-dataset Diebold-Mariano tests.

A "dataset" (block) is one target index x one label set: 3 x 3 = 9 blocks. The score
per block is the seed-mean test PR-AUC from results/baseline_models_{IDX}.csv and
results/isolation_forest_{IDX}.csv. This follows the usual procedure for comparing
methods over several datasets (Demsar, 2006).

Tests:
  1. Wilcoxon signed-rank (exact, two-sided), per model: Domestic-Only vs each other
     feature set over the 9 blocks, with Holm's correction within each model.
  2. Friedman test, per model: all feature sets together; mean rank per set (1 = best).
  3. Friedman test, per feature set: the three models together; mean rank per model.

Caveat: the 9 blocks share the same test period and the indices are correlated, so
they are not fully independent; with n = 9 the smallest attainable Wilcoxon p-value
is 2/512 = 0.0039.

Where a feature set has the same columns as another (Nikkei 225 has no same-day
foreign data, so cross_market_lag0 = cross_market), the twin's score is used.

Usage:
    python code/significance.py

Writes:
    results/significance_tests.csv
    results/tables/table_significance.csv
    thesis_tables/table_significance_settings.txt
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

INDICES = ["GSPC", "N225", "FTSE"]
LABELS = ["baseline", "hmm", "regime_std"]
MODELS = ["xgboost", "logistic_regression", "isolation_forest"]
MODEL_NAMES = {"xgboost": "XGBoost, 2016", "logistic_regression": "LR, 1958", "isolation_forest": "iForest, 2008"}
SET_NAMES = {"domestic_only": "Domestic-Only", "cross_market": "Cross-Market (lag 1)",
             "cross_market_lag0": "Cross-Market (lag 0 + 1)", "ablation_foreign": "Domestic + foreign",
             "ablation_macro": "Domestic + macro", "rv_only": "Realised volatility only"}


def load_scores():
    frames = []
    for idx in INDICES:
        for f in [f"results/baseline_models_{idx}.csv", f"results/isolation_forest_{idx}.csv"]:
            if os.path.exists(f):
                frames.append(pd.read_csv(f).assign(index=idx))
    df = pd.concat(frames)
    df = df[df["label"].isin(LABELS)]
    # Fill twins (identical columns) with the twin's score.
    for i, r in df[df["pr_auc_mean"].isna()].iterrows():
        twin = str(r["note"]).split("same columns as ")[-1].split(":")[0].strip()
        src = df[(df["index"] == r["index"]) & (df["label"] == r["label"]) & (df["model"] == r["model"])
                 & (df["feature_set"] == twin)]
        df.loc[(df["index"] == r["index"]) & (df["label"] == r["label"]) & (df["model"] == r["model"])
               & (df["feature_set"] == r["feature_set"]), "pr_auc_mean"] = src["pr_auc_mean"].iloc[0]
    # One row per block, one column per (model, feature_set).
    return df.pivot_table(index=["index", "label"], columns=["model", "feature_set"], values="pr_auc_mean")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    wide = load_scores()
    n_blocks = len(wide)
    rows = []

    for model in MODELS:
        sets = [fs for fs in SET_NAMES if (model, fs) in wide.columns]
        if len(sets) < 2:
            continue
        ref = wide[(model, "domestic_only")]
        tests = []
        for fs in sets:
            if fs == "domestic_only":
                continue
            d = wide[(model, fs)] - ref
            stat, p = wilcoxon(wide[(model, fs)], ref, alternative="two-sided", method="exact")
            tests.append({"test": "Wilcoxon signed-rank", "model": model, "comparison": f"{fs} vs domestic_only",
                          "n_blocks": n_blocks, "statistic": float(stat), "p_value": float(p),
                          "median_diff": float(d.median()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())})
        for t, p_adj in zip(tests, holm([t["p_value"] for t in tests])):
            t["p_holm"] = float(p_adj)
        rows += tests

        block_scores = wide[[(model, fs) for fs in sets]].to_numpy()
        stat, p = friedmanchisquare(*block_scores.T)
        ranks = np.mean([rankdata(-r) for r in block_scores], axis=0)
        rows.append({"test": "Friedman", "model": model, "comparison": f"{len(sets)} feature sets",
                     "n_blocks": n_blocks, "statistic": float(stat), "p_value": float(p),
                     "mean_ranks": "; ".join(f"{fs} {r:.2f}" for fs, r in zip(sets, ranks))})

    for fs in ["domestic_only", "cross_market", "cross_market_lag0"]:
        cols = [(m, fs) for m in MODELS if (m, fs) in wide.columns]
        block_scores = wide[cols].to_numpy()
        stat, p = friedmanchisquare(*block_scores.T)
        ranks = np.mean([rankdata(-r) for r in block_scores], axis=0)
        rows.append({"test": "Friedman", "model": "all three models", "comparison": f"models on {fs}",
                     "n_blocks": n_blocks, "statistic": float(stat), "p_value": float(p),
                     "mean_ranks": "; ".join(f"{m} {r:.2f}" for (m, _), r in zip(cols, ranks))})

    out = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    out.to_csv("results/significance_tests.csv", index=False)

    table = out.copy()
    table["model"] = table["model"].map(lambda m: MODEL_NAMES.get(m, m))
    # Longest keys first, so cross_market_lag0 is not partly replaced by cross_market.
    for fs, name in sorted(SET_NAMES.items(), key=lambda kv: -len(kv[0])):
        table["comparison"] = table["comparison"].str.replace(fs, name, regex=False)
        table["mean_ranks"] = table["mean_ranks"].fillna("").astype(str).str.replace(fs, name, regex=False)
    for m, name in MODEL_NAMES.items():
        table["mean_ranks"] = table["mean_ranks"].str.replace(m, name, regex=False)
    for c in ["statistic", "median_diff"]:
        table[c] = table[c].round(3)
    for c in ["p_value", "p_holm"]:
        table[c] = table[c].round(4)
    table.to_csv("results/tables/table_significance.csv", index=False)
    with open("thesis_tables/table_significance_settings.txt", "w") as f:
        f.write("\n".join([
            "Dataset: S&P 500, Nikkei 225, FTSE 100; blocks = 3 indices x 3 label sets (baseline, hmm, regime_std)",
            "Feature set: all sets in results/baseline_models_{IDX}.csv and results/isolation_forest_{IDX}.csv",
            "Label strategy: baseline, hmm, regime_std (hmm_fullsample excluded)",
            "Model: XGBoost (2016), L2 logistic regression (1958), Isolation Forest (2008)",
            "Seed: 42-46; score per block = seed-mean test PR-AUC",
            "Config file: none (reads result files); code/significance.py",
            "Checkpoint: n/a",
            "Test window: 2019-01-01 to 2024-12-31",
            "Notes: Wilcoxon signed-rank, exact two-sided, Holm-corrected within each model; Friedman chi-square "
            "with mean ranks (1 = best). Blocks share the test period and are not fully independent; with 9 "
            "blocks the smallest attainable Wilcoxon p is 0.0039. Twins (identical columns) take the twin's score.",
        ]) + "\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(table.to_string(index=False))
    print("\nSaved results/significance_tests.csv and results/tables/table_significance.csv")


if __name__ == "__main__":
    main()
