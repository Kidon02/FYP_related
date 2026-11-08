"""
Global SHAP feature attribution for XGBoost (research question 2): which foreign
features, and at which lags, drive the anomaly predictions.

For each target, label set and foreign-containing feature set, refits XGBoost with the
hyperparameters already chosen on validation (results/baseline_models_{IDX}.csv), on
the same rows, split, scaler, SMOTE and seeds as code/baseline_models.py. TreeSHAP
values (path-dependent Tree SHAP, Lundberg et al., 2020, Nat. Mach. Intell. 2(1), 56-67,
doi:10.1038/s42256-019-0138-9; SHAP values as defined by Lundberg & Lee, 2017;
shap.TreeExplainer without background data) are computed on the 2019-2024 test
days; mean |SHAP| per feature is averaged over the 5 seeds. Group shares are each
group's share of the total mean |SHAP| (domestic, foreign lag 1+, foreign lag 0, macro,
technical). The first model's SHAP values are checked against XGBoost's own exact
contributions (pred_contribs).

Which feature set to emphasise is decided by validation PR-AUC (flag
`best_on_validation`), never by test results.

Usage:
    python code/shap_analysis.py

Writes:
    results/shap_global_{IDX}.csv          mean |SHAP| per feature
    results/shap_group_shares.csv          group shares per target, label, feature set
    results/tables/table_shap_group_shares.csv
    results/figures/fig_shap_top_{IDX}.png, .pdf
    thesis_tables/table_shap_group_shares_settings.txt, fig_shap_top_{IDX}_settings.txt
"""

import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from baseline_models import load_inputs, load_yaml, split_index
from features import resolve_feature_set

INDICES = ["GSPC", "N225", "FTSE"]
INDEX_NAMES = {"GSPC": "S&P 500", "N225": "Nikkei 225", "FTSE": "FTSE 100"}
LABEL_NAMES = {"baseline": "Rolling z-score baseline", "hmm": "Regime-dependent threshold",
               "regime_std": "Regime-standardised"}
SET_NAMES = {"cross_market": "Cross-Market (lag 1)", "cross_market_lag0": "Cross-Market (lag 0 + 1)",
             "ablation_foreign": "Domestic + foreign"}
GROUPS = ["domestic", "foreign", "foreign_lag0", "macro", "technical"]


def mean_abs_shap(X, y, masks, fixed, params, seeds, imb, check):
    tr, _, te = masks
    Xs = StandardScaler().fit(X[tr]).transform(X)
    per_seed = []
    for seed in seeds:
        sm = SMOTE(sampling_strategy=imb["sampling_strategy"], k_neighbors=imb["k_neighbors"], random_state=seed)
        Xr, yr = sm.fit_resample(Xs[tr], y[tr])
        model = XGBClassifier(random_state=seed, **fixed, **params).fit(Xr, yr)
        sv = shap.TreeExplainer(model).shap_values(Xs[te])
        if check:
            exact = model.get_booster().predict(xgb.DMatrix(Xs[te]), pred_contribs=True)[:, :-1]
            assert np.allclose(sv, exact, atol=1e-4), "shap.TreeExplainer disagrees with XGBoost pred_contribs"
            print("SHAP check passed: TreeExplainer matches XGBoost's exact contributions.")
            check = False
        per_seed.append(np.abs(sv).mean(axis=0))
    return np.mean(per_seed, axis=0)


def figure(index, label, fs, imp, groups, fcfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    top = imp.sort_values(ascending=False).head(fcfg["top_n"])[::-1]
    hatches = {"domestic": "", "foreign": "///", "foreign_lag0": "xxx", "macro": "...", "technical": "\\\\\\"}
    plt.rcParams.update({"font.size": 8, "axes.grid": False})
    fig, ax = plt.subplots(figsize=(fcfg["width_in"], fcfg["height_in"]))
    for i, (feat, val) in enumerate(top.items()):
        ax.barh(i, val, color="white", edgecolor="black", hatch=hatches[groups[feat]], linewidth=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Mean |SHAP value| on test days (log-odds)")
    ax.xaxis.set_major_locator(MaxNLocator(fcfg["max_ticks"] - 1))
    from matplotlib.patches import Patch
    used = [g for g in GROUPS if g in set(groups[top.index])]
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=hatches[g], label=g.replace("_", " "))
                       for g in used], frameon=False, loc="lower right")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(f"results/figures/fig_shap_top_{index}.{ext}", dpi=fcfg["dpi"])
    plt.close(fig)
    caption = (f"Figure: the {fcfg['top_n']} features with the largest mean absolute SHAP value for XGBoost on "
               f"the {INDEX_NAMES[index]} test window (2019-2024), {LABEL_NAMES[label]} labels, "
               f"{SET_NAMES[fs]} feature set, averaged over 5 seeds. Hatching shows the feature group.")
    with open(f"thesis_tables/fig_shap_top_{index}_settings.txt", "w") as f:
        f.write("\n".join([
            f"Dataset: {INDEX_NAMES[index]} test window 2019-2024", f"Feature set: {fs}", f"Label strategy: {label}",
            "Model: XGBoost (2016), hyperparameters from results/baseline_models_" + index + ".csv",
            "Seed: 42-46 (mean |SHAP| averaged over seeds)", "Config file: configs/shap.yaml",
            "Checkpoint: not saved (refit in code/shap_analysis.py)", "Test window: 2019-01-01 to 2024-12-31",
            "Notes: " + caption,
        ]) + "\n")


def main():
    cfg = load_yaml("configs/shap.yaml")
    base = load_yaml("configs/baseline_models.yaml")
    feat_cfg = load_yaml("configs/features.yaml")
    seeds = [base["seeds"]["base"] + i for i in range(base["seeds"]["n"])]
    fixed = base["models"]["xgboost"]["fixed"]
    shares, check = [], True

    for index in INDICES:
        features, groups, labels, _ = load_inputs(index, feat_cfg)
        all_labels = base["label_sets"] + base["sensitivity"]["label_sets"]
        features = features.loc[labels.loc[features.index, all_labels].notna().all(axis=1)]
        masks = split_index(features.index, base["split"])
        res = pd.read_csv(f"results/baseline_models_{index}.csv")
        rows, seen = [], {}
        for fs in cfg["feature_set_names"]:
            cols = resolve_feature_set(feat_cfg["feature_sets"][fs], groups)
            if any(c == cols for c in seen.values()):
                continue                                   # identical columns (N225 lag 0)
            seen[fs] = cols
            X = features[cols].to_numpy(float)
            for label in cfg["label_sets"]:
                r = res.query("model == 'xgboost' and feature_set == @fs and label == @label").iloc[0]
                print(f"SHAP | {index} | {fs} | {label} ...", flush=True)
                y = labels.loc[features.index, label].astype(int).to_numpy()
                imp = pd.Series(mean_abs_shap(X, y, masks, fixed, json.loads(r["best_params"]), seeds,
                                              base["imbalance"], check), index=cols)
                check = False
                rows += [{"label": label, "feature_set": fs, "feature": c, "group": groups[c],
                          "mean_abs_shap": float(v)} for c, v in imp.items()]
                by_group = imp.groupby(groups[cols].values).sum() / imp.sum()
                shares.append({"index": index, "label": label, "feature_set": fs,
                               "val_pr_auc": r["val_pr_auc_mean"], "test_pr_auc": r["pr_auc_mean"],
                               **{f"share_{g}": float(by_group.get(g, 0.0)) for g in GROUPS},
                               "top5": "; ".join(imp.sort_values(ascending=False).head(5).index)})
                fc = cfg["figure"]
                if index == fc["index"] and label == fc["label"] and fs == "cross_market_lag0":
                    figure(index, label, fs, imp, groups, fc)
        out = pd.DataFrame(rows)
        out["rank"] = out.groupby(["label", "feature_set"])["mean_abs_shap"].rank(ascending=False).astype(int)
        out.to_csv(f"results/shap_global_{index}.csv", index=False)

    sh = pd.DataFrame(shares)
    sh["best_on_validation"] = sh.groupby(["index", "label"])["val_pr_auc"].transform("max") == sh["val_pr_auc"]
    sh.to_csv("results/shap_group_shares.csv", index=False)
    table = pd.DataFrame({
        "index": sh["index"].map(INDEX_NAMES), "label": sh["label"].map(LABEL_NAMES),
        "feature_set": sh["feature_set"].map(SET_NAMES), "best_on_validation": sh["best_on_validation"],
        "test_pr_auc": sh["test_pr_auc"].round(3),
        **{f"share_{g}": sh[f"share_{g}"].round(3) for g in GROUPS}, "top5_features": sh["top5"],
    })
    os.makedirs("results/tables", exist_ok=True)
    table.to_csv("results/tables/table_shap_group_shares.csv", index=False)
    with open("thesis_tables/table_shap_group_shares_settings.txt", "w") as f:
        f.write("\n".join([
            "Dataset: S&P 500, Nikkei 225, FTSE 100 test windows 2019-2024 (same rows as baseline_models.py)",
            f"Feature set: {', '.join(cfg['feature_set_names'])} (N225 lag 0 = lag 1, not repeated)",
            f"Label strategy: {', '.join(cfg['label_sets'])}",
            "Model: XGBoost (2016) with validation-chosen hyperparameters; path-dependent Tree SHAP (Lundberg et al., 2020; SHAP values: Lundberg & Lee, 2017)",
            "Seed: 42-46 (mean |SHAP| averaged over seeds)", "Config file: configs/shap.yaml, configs/baseline_models.yaml",
            "Checkpoint: not saved (refit in code/shap_analysis.py)", "Test window: 2019-01-01 to 2024-12-31",
            "Notes: share_g = group's share of total mean |SHAP| on test days (log-odds scale, standardised "
            "inputs). best_on_validation marks the feature set with the highest validation PR-AUC per index and "
            "label; test results are not used for that choice. SHAP explains the fitted model, so attributions "
            "for weak models (low test PR-AUC) should not be read as evidence about markets.",
        ]) + "\n")
    with pd.option_context("display.width", 220, "display.max_colwidth", 80):
        print(table.drop(columns=["top5_features"]).to_string(index=False))
    print("\nSaved results/shap_group_shares.csv, results/tables/table_shap_group_shares.csv and SHAP figure")


if __name__ == "__main__":
    main()
