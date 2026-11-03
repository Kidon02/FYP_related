"""
Isolation Forest benchmark (unsupervised), as specified in the brief.

For one target index, fits sklearn's IsolationForest on training-period features only
(no labels), scores every day by -score_samples (higher = more anomalous), and
evaluates those scores against each label set with the same protocol as
code/baseline_models.py: grid point chosen by mean validation PR-AUC over seeds,
F1 threshold chosen per seed on validation, test window 2019-2024 scored once, and
the same block-bootstrap 95% CIs (identical resamples, same bootstrap seed).

No scaling and no SMOTE: the forest is unsupervised and its splits are invariant to
per-feature scaling.

Usage:
    python code/isolation_forest.py --index GSPC

Writes:
    results/isolation_forest_{IDX}.csv
    thesis_tables/isolation_forest_{IDX}_settings.txt
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from baseline_models import (block_bootstrap_samples, bootstrap_ci, check_inputs, f1_threshold,
                             grid_points, load_inputs, load_yaml, split_index)
from features import resolve_feature_set


def run_iforest(X, ys, masks, model_cfg, seeds):
    """Fit once per (grid point, seed); evaluate the same scores against every label set."""
    tr, va, te = masks
    scores = {}
    for params in grid_points(model_cfg["grid"]):
        key = json.dumps(params, sort_keys=True)
        for seed in seeds:
            forest = IsolationForest(random_state=seed, **model_cfg["fixed"], **params).fit(X[tr])
            scores[(key, seed)] = -forest.score_samples(X)
    out = {}
    for label, y in ys.items():
        val = {k: average_precision_score(y[va], s[va]) for k, s in scores.items()}
        keys = sorted({k for k, _ in scores})
        means = {k: np.mean([val[(k, s)] for s in seeds]) for k in keys}
        best = max(keys, key=lambda k: means[k])
        pr, f1s, mccs, thrs, s_tests = [], [], [], [], []
        for seed in seeds:
            s = scores[(best, seed)]
            thr = f1_threshold(y[va], s[va])
            pred = (s[te] >= thr).astype(int)
            pr.append(average_precision_score(y[te], s[te]))
            f1s.append(f1_score(y[te], pred, zero_division=0))
            mccs.append(matthews_corrcoef(y[te], pred))
            thrs.append(thr)
            s_tests.append(s[te])
        out[label] = ({
            "best_params": best,
            "val_pr_auc_mean": float(means[best]),
            "val_pr_auc_std": float(np.std([val[(best, s)] for s in seeds], ddof=1)),
            "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr, ddof=1)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
            "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs, ddof=1)),
            "test_n": int(te.sum()), "test_positives": int(y[te].sum()),
            "pr_auc_prevalence_baseline": float(y[te].mean()),
        }, s_tests, thrs)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--config", default="configs/isolation_forest.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline_models.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--labelling-config", default="configs/labelling.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = load_yaml(args.baseline_config)
    feat_cfg = load_yaml(args.features_config)
    lab_cfg = load_yaml(args.labelling_config)
    seeds = [base["seeds"]["base"] + i for i in range(base["seeds"]["n"])]

    features, groups, labels, panel = load_inputs(args.index, feat_cfg)
    check_inputs(args.index, features, groups, labels, panel, lab_cfg)
    # Same rows and the same bootstrap resamples as baseline_models.py.
    all_labels = base["label_sets"] + base["sensitivity"]["label_sets"]
    defined = labels.loc[features.index, all_labels].notna().all(axis=1)
    features = features.loc[defined]
    tr, va, te = split_index(features.index, base["split"])
    b = base["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    y_all = {lab: labels.loc[features.index, lab].astype(int).to_numpy() for lab in all_labels}
    samples = {lab: block_bootstrap_samples(y_all[lab][te].astype(bool), b["block_length"], b["cluster_gap"],
                                            b["n_resamples"], rng) for lab in all_labels}
    print(f"Rows: train {tr.sum()}, validation {va.sum()}, test {te.sum()}")

    rows, seen = [], {}
    for fs in cfg["feature_set_names"]:
        cols = resolve_feature_set(feat_cfg["feature_sets"][fs], groups)
        twin = next((o for o, c in seen.items() if c == cols), None)
        seen[fs] = cols
        if twin:
            print(f"Skipping {fs}: same columns as {twin}", flush=True)
            for lab in cfg["label_sets"]:
                rows.append({"model": "isolation_forest", "feature_set": fs, "label": lab,
                             "n_features": len(cols), "note": f"same columns as {twin}"})
            continue
        print(f"Running isolation_forest | {fs} ({len(cols)} features) ...", flush=True)
        X = features[cols].to_numpy(float)
        ys = {lab: y_all[lab] for lab in cfg["label_sets"]}
        for lab, (summary, s_tests, thrs) in run_iforest(X, ys, (tr, va, te), cfg["model"], seeds).items():
            ci = bootstrap_ci(ys[lab][te], s_tests, thrs, samples[lab], b["confidence"])
            rows.append({"model": "isolation_forest", "feature_set": fs, "label": lab,
                         "n_features": len(cols), **summary, **ci, "note": ""})

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    os.makedirs("thesis_tables", exist_ok=True)
    out.to_csv(f"results/isolation_forest_{args.index}.csv", index=False)
    with open(f"thesis_tables/isolation_forest_{args.index}_settings.txt", "w") as f:
        f.write("\n".join([
            f"Dataset: {args.index} ({os.path.basename(feat_cfg['panel_path'])}); same rows as baseline_models.py",
            f"Feature set: {', '.join(cfg['feature_set_names'])} (configs/features.yaml)",
            f"Label strategy: {', '.join(cfg['label_sets'])} (labels used only for selection and scoring)",
            "Model: isolation_forest (sklearn IsolationForest, unsupervised, fit on 2000-2015 features only)",
            f"Seed: {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds; mean and std reported)",
            "Config file: configs/isolation_forest.yaml, configs/baseline_models.yaml (split, seeds, bootstrap)",
            "Checkpoint: not saved (forests are refit inside code/isolation_forest.py)",
            "Test window: 2019-01-01 to 2024-12-31",
            f"Notes: anomaly score = -score_samples; grid on validation PR-AUC; 95% CIs from the same "
            f"{b['n_resamples']} block-bootstrap resamples as baseline_models.py (bootstrap seed {b['seed']}).",
        ]) + "\n")
    show = out.dropna(subset=["pr_auc_mean"])[["feature_set", "label", "pr_auc_mean", "pr_auc_ci_low",
                                                "pr_auc_ci_high", "f1_mean", "mcc_mean", "pr_auc_prevalence_baseline"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results/isolation_forest_{args.index}.csv")


if __name__ == "__main__":
    main()
