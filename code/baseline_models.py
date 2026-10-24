"""
Baseline models: XGBoost and L2 logistic regression.

Runs both models on both feature sets (domestic_only, cross_market) and both
label sets (baseline, hmm), with 5 seeds each.

Usage:
    python code/baseline_models.py --index GSPC

Protocol per combination:
  1. Fit a StandardScaler on train only.
  2. Apply SMOTE on train only (after scaling), with the seed.
  3. For each grid point and seed, fit and score validation PR-AUC.
  4. Pick the grid point with the highest mean validation PR-AUC.
  5. For each seed, pick the F1-maximising threshold on validation.
  6. Apply that threshold, unchanged, to test. Report mean and std.

Writes:
    results/baseline_models_{IDX}.csv
    logs/baseline_models_{IDX}_grid.csv       all grid points, validation only
    thesis_tables/baseline_models_{IDX}_settings.txt
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from labelling import baseline_label, regime_conditioned_label, rolling_zscore

LABEL_TOKENS = {"baseline", "hmm", "regime"}
LABEL_FILE_COLUMN = {"baseline": "baseline", "hmm": "hmm"}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def f1_threshold(y, p):
    """Threshold maximising F1 on the given (validation) data."""
    assert y.sum() > 0, "validation set has no positives"
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    denom = prec + rec
    f1s = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    return float(thr[int(np.argmax(f1s))])


def make_model(name, params, seed):
    if name == "xgboost":
        return XGBClassifier(random_state=seed, **params)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    raise ValueError(name)


def grid_points(grid):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def load_inputs(index, feat_cfg):
    features = pd.read_csv(f"data/processed/features_{index}.csv", index_col=0, parse_dates=True)
    groups = pd.read_csv(f"data/processed/feature_groups_{index}.csv", index_col=0)["group"]
    labels = pd.read_csv(f"results/labels_{index}.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(feat_cfg["panel_path"], index_col=0, parse_dates=True)
    return features, groups, labels, panel


def check_inputs(index, features, groups, labels, panel):
    assert list(features.columns) == list(groups.index), "feature file and group map disagree"
    for col in features.columns:
        assert not (set(col.lower().split("_")) & LABEL_TOKENS), f"label-derived feature: {col}"
    assert features.index.isin(labels.index).all(), "feature dates missing from labels"
    assert features.notna().all().all(), "features contain NaN"

    r = panel[f"{index}_logret"]
    # The saved labels must be reproducible from the panel with labelling.py's own functions.
    recomputed_base = baseline_label(r)
    assert (recomputed_base.loc[features.index] == labels.loc[features.index, "baseline"]).all(), \
        "baseline labels do not match labelling.py"
    recomputed_hmm = regime_conditioned_label(r, labels["regime"])
    assert (recomputed_hmm.loc[features.index] == labels.loc[features.index, "hmm"]).all(), \
        "hmm labels do not match labelling.py"
    z = rolling_zscore(r)
    assert z.loc[features.index].notna().all(), "z-score undefined on a kept row"
    print("Input checks passed: no label-derived features, labels reproduce, z-scores defined.")


def split_index(index, split):
    def within(bounds):
        lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        return (index >= lo) & (index <= hi)
    return within(split["train"]), within(split["validation"]), within(split["test"])


def run_combination(X, y, masks, model_name, model_cfg, seeds, cfg):
    tr, va, te = masks
    scaler = StandardScaler().fit(X[tr])          # train only
    Xs = scaler.transform(X)
    Xtr, ytr = Xs[tr], y[tr]
    Xva, yva = Xs[va], y[va]
    Xte, yte = Xs[te], y[te]

    # --- grid search on validation PR-AUC, averaged over seeds ---
    grid_rows, fitted = [], {}
    for params in grid_points(model_cfg["grid"]):
        full = {**model_cfg["fixed"], **params}
        val_scores = []
        for seed in seeds:
            sm = SMOTE(sampling_strategy=cfg["imbalance"]["sampling_strategy"],
                       k_neighbors=cfg["imbalance"]["k_neighbors"], random_state=seed)
            Xr, yr = sm.fit_resample(Xtr, ytr)
            model = make_model(model_name, full, seed).fit(Xr, yr)
            p_val = model.predict_proba(Xva)[:, 1]
            val_scores.append(average_precision_score(yva, p_val))
            fitted[(json.dumps(params, sort_keys=True), seed)] = (model, p_val)
        grid_rows.append({
            "params": json.dumps(params, sort_keys=True),
            "val_pr_auc_mean": float(np.mean(val_scores)),
            "val_pr_auc_std": float(np.std(val_scores, ddof=1)),
        })

    grid_df = pd.DataFrame(grid_rows)
    best_key = grid_df.sort_values("val_pr_auc_mean", ascending=False).iloc[0]["params"]
    best = grid_df.loc[grid_df["params"] == best_key].iloc[0]

    # --- per-seed threshold on validation, then one test evaluation ---
    pr, f1s, mccs, thrs = [], [], [], []
    for seed in seeds:
        model, p_val = fitted[(best_key, seed)]
        thr = f1_threshold(yva, p_val)
        p_te = model.predict_proba(Xte)[:, 1]
        pred = (p_te >= thr).astype(int)
        pr.append(average_precision_score(yte, p_te))
        f1s.append(f1_score(yte, pred, zero_division=0))
        mccs.append(matthews_corrcoef(yte, pred))
        thrs.append(thr)

    summary = {
        "best_params": best_key,
        "val_pr_auc_mean": best["val_pr_auc_mean"],
        "val_pr_auc_std": best["val_pr_auc_std"],
        "threshold_mean": float(np.mean(thrs)),
        "threshold_std": float(np.std(thrs, ddof=1)),
        "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr, ddof=1)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
        "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs, ddof=1)),
        "test_n": int(te.sum()), "test_positives": int(yte.sum()),
        "test_prevalence": float(yte.mean()),
    }
    return summary, grid_df


def settings_text(index, summary_df, n_seeds, seed_base):
    prev = summary_df["test_prevalence"].iloc[0]
    return "\n".join([
        f"Dataset: {index} (panel_full.csv, 2000-01-01 to 2024-12-31)",
        "Feature set: domestic_only and cross_market (both rows reported)",
        "Label strategy: baseline and hmm (both rows reported)",
        "Model: xgboost and logistic_regression (both rows reported)",
        f"Seed: {seed_base}-{seed_base + n_seeds - 1} ({n_seeds} seeds; mean and std reported)",
        "Config file: configs/baseline_models.yaml, configs/features.yaml",
        "Checkpoint: not saved (models are refit inside code/baseline_models.py)",
        "Test window: 2019-01-01 to 2024-12-31",
        f"Notes: test anomaly prevalence {100 * prev:.2f}% (PR-AUC baseline). "
        "Generated by code/baseline_models.py; grid search on validation only.",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    feat_cfg = load_yaml(args.features_config)
    seeds = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["n"])]

    features, groups, labels, panel = load_inputs(args.index, feat_cfg)
    check_inputs(args.index, features, groups, labels, panel)
    tr, va, te = split_index(features.index, cfg["split"])
    print(f"Rows: train {tr.sum()}, validation {va.sum()}, test {te.sum()}")

    results, grid_all = [], []
    for fs, label, model_name in itertools.product(
        cfg["feature_set_names"], cfg["label_sets"], cfg["models"]
    ):
        cols = [c for c in features.columns if groups[c] in feat_cfg["feature_sets"][fs]]
        X = features[cols].to_numpy(float)
        y = labels.loc[features.index, LABEL_FILE_COLUMN[label]].astype(int).to_numpy()
        print(f"Running {model_name} | {fs} ({len(cols)} features) | {label} ...", flush=True)

        summary, grid_df = run_combination(
            X, y, (tr, va, te), model_name, cfg["models"][model_name], seeds, cfg
        )
        results.append({"model": model_name, "feature_set": fs, "label": label,
                        "n_features": len(cols), **summary})
        grid_df.insert(0, "model", model_name)
        grid_df.insert(1, "feature_set", fs)
        grid_df.insert(2, "label", label)
        grid_all.append(grid_df)

    out = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("thesis_tables", exist_ok=True)

    out.to_csv(f"results/baseline_models_{args.index}.csv", index=False)
    pd.concat(grid_all).to_csv(f"logs/baseline_models_{args.index}_grid.csv", index=False)
    with open(f"thesis_tables/baseline_models_{args.index}_settings.txt", "w") as f:
        f.write(settings_text(args.index, out, len(seeds), seeds[0]))

    show = out[["model", "feature_set", "label", "pr_auc_mean", "pr_auc_std",
                "f1_mean", "f1_std", "mcc_mean", "mcc_std", "test_prevalence"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results/baseline_models_{args.index}.csv")


if __name__ == "__main__":
    main()
