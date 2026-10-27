"""
Baseline models: XGBoost and L2 logistic regression.

For one target index, runs both models on every feature set in
configs/baseline_models.yaml (defined in configs/features.yaml) and every label set
(baseline, causal regime-dependent threshold, causal regime-standardised), with 5
seeds each, plus the full-sample-HMM sensitivity labels on the main comparison.

Usage:
    python code/baseline_models.py --index GSPC

Protocol per combination:
  1. Fit a StandardScaler on train only.
  2. Apply SMOTE on train only (after scaling), with the seed.
  3. For each grid point and seed, fit and score validation PR-AUC.
  4. Pick the grid point with the highest mean validation PR-AUC.
  5. For each seed, pick the F1-maximising threshold on validation.
  6. Apply that threshold, unchanged, to test. Report mean and std over seeds, and a
     block-bootstrap 95% CI of the seed-mean test metric (anomaly clusters kept whole).

Every label set is evaluated on the same rows: feature rows where all label sets in
the run are defined. The PR-AUC of a random classifier equals the test prevalence,
reported as pr_auc_prevalence_baseline.

Writes:
    results/baseline_models_{IDX}.csv
    results/test_predictions_{IDX}.csv        seed-mean test probabilities (for DM tests)
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

from features import resolve_feature_set
from labelling import baseline_label, regime_conditioned_label, regime_standardised_label

LABEL_TOKENS = {"baseline", "hmm", "regime"}


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


# ---------- test metrics and the block bootstrap ----------

def average_precision(y, p):
    """PR-AUC, computed as sklearn's average_precision_score does (ties grouped)."""
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]   # last position of each distinct score
    tp = np.cumsum(y)[last]
    if tp[-1] == 0:
        return np.nan
    return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * tp / (last + 1)))


def f1_mcc(y, pred):
    """F1 and MCC from boolean labels and predictions (0 when undefined, as sklearn)."""
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = len(y) - tp - fp - fn
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return f1, ((tp * tn - fp * fn) / denom if denom else 0.0)


def cluster_spans(y, gap):
    """[first, last] positions of anomaly clusters: anomaly days at most `gap` positions apart."""
    spans = []
    for p in np.flatnonzero(y):
        if spans and p - spans[-1][1] <= gap:
            spans[-1][1] = p
        else:
            spans.append([p, p])
    return spans


def block_bootstrap_samples(y, block_length, gap, n_resamples, rng):
    """
    Index arrays for a moving-block bootstrap that keeps anomaly clusters whole: a
    block may not start inside a cluster, and a block that would end inside one is
    extended to the cluster's last day. Blocks are drawn until a resample has at
    least len(y) days.
    """
    n = len(y)
    cluster_end = np.full(n, -1)
    can_start = np.ones(n, bool)
    for a, b in cluster_spans(y, gap):
        cluster_end[a:b + 1] = b
        can_start[a + 1:b + 1] = False
    starts = np.flatnonzero(can_start)
    samples = []
    for _ in range(n_resamples):
        parts, size = [], 0
        while size < n:
            for s in rng.choice(starts, size=n // block_length + 1):
                e = min(s + block_length, n) - 1
                if cluster_end[e] >= 0:
                    e = cluster_end[e]
                parts.append(np.arange(s, e + 1))
                size += e - s + 1
                if size >= n:
                    break
        samples.append(np.concatenate(parts))
    return samples


def bootstrap_ci(y, p_seeds, thresholds, samples, confidence):
    """Percentile CIs of the seed-mean PR-AUC, F1 and MCC over the bootstrap resamples."""
    y = y.astype(bool)
    preds = [p >= t for p, t in zip(p_seeds, thresholds)]
    stats = np.empty((len(samples), 3))
    for i, idx in enumerate(samples):
        ys = y[idx]
        fm = [f1_mcc(ys, d[idx]) for d in preds]
        stats[i] = [np.mean([average_precision(ys, p[idx]) for p in p_seeds]),
                    np.mean([f for f, _ in fm]), np.mean([m for _, m in fm])]
    tail = (1 - confidence) / 2
    lo, hi = np.nanquantile(stats, [tail, 1 - tail], axis=0)
    return {f"{m}_ci_low": float(lo[j]) for j, m in enumerate(["pr_auc", "f1", "mcc"])} | \
           {f"{m}_ci_high": float(hi[j]) for j, m in enumerate(["pr_auc", "f1", "mcc"])}


# ---------- inputs ----------

def load_inputs(index, feat_cfg):
    features = pd.read_csv(f"data/processed/features_{index}.csv", index_col=0, parse_dates=True)
    groups = pd.read_csv(f"data/processed/feature_groups_{index}.csv", index_col=0)["group"]
    labels = pd.read_csv(f"results/labels_{index}.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(feat_cfg["panel_path"], index_col=0, parse_dates=True)
    return features, groups, labels, panel


def check_inputs(index, features, groups, labels, panel, lab_cfg):
    assert list(features.columns) == list(groups.index), "feature file and group map disagree"
    for col in features.columns:
        assert not (set(col.lower().split("_")) & LABEL_TOKENS), f"label-derived feature: {col}"
    assert features.index.isin(labels.index).all(), "feature dates missing from labels"
    assert features.notna().all().all(), "features contain NaN"

    # The saved labels must be reproducible from the panel and the saved regimes
    # with labelling.py's own functions.
    r, w = panel[f"{index}_logret"], lab_cfg["zscore_window"]
    thr, rs = tuple(lab_cfg["regime_threshold"]["thresholds"]), lab_cfg["regime_standardised"]
    recomputed = {
        "baseline": baseline_label(r, w, lab_cfg["baseline"]["threshold"]),
        "hmm": regime_conditioned_label(r, labels["regime_filtered"], w, thr),
        "regime_std": regime_standardised_label(r, labels["regime_predicted"], rs["window_obs"],
                                                rs["min_obs"], rs["threshold"]),
        "hmm_fullsample": regime_conditioned_label(r, labels["regime_fullsample"], w, thr),
    }
    for name, lab in recomputed.items():
        a, b = lab.loc[features.index], labels.loc[features.index, name]
        assert ((a == b) | (a.isna() & b.isna())).all(), f"{name} labels do not match labelling.py"
    print("Input checks passed: no label-derived features, all four label sets reproduce.")


def split_index(index, split):
    def within(bounds):
        lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        return (index >= lo) & (index <= hi)
    return within(split["train"]), within(split["validation"]), within(split["test"])


# ---------- one combination ----------

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
    pr, f1s, mccs, thrs, p_tests = [], [], [], [], []
    for seed in seeds:
        model, p_val = fitted[(best_key, seed)]
        thr = f1_threshold(yva, p_val)
        p_te = model.predict_proba(Xte)[:, 1]
        pred = (p_te >= thr).astype(int)
        pr.append(average_precision_score(yte, p_te))
        f1s.append(f1_score(yte, pred, zero_division=0))
        mccs.append(matthews_corrcoef(yte, pred))
        thrs.append(thr)
        p_tests.append(p_te)
        # The bootstrap's fast metrics must agree with sklearn's.
        f1_fast, mcc_fast = f1_mcc(yte.astype(bool), pred.astype(bool))
        assert np.isclose(average_precision(yte, p_te), pr[-1]) and np.isclose(f1_fast, f1s[-1]) \
            and np.isclose(mcc_fast, mccs[-1]), "fast test metrics disagree with sklearn"

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
        "pr_auc_prevalence_baseline": float(yte.mean()),
    }
    return summary, grid_df, p_tests, thrs


def settings_text(index, out, cfg, rows, n_seeds, seed_base, panel_path):
    b = cfg["bootstrap"]
    prev = out.groupby("label")["pr_auc_prevalence_baseline"].first()
    return "\n".join([
        f"Dataset: {index} ({os.path.basename(panel_path)}, 2000-01-01 to 2024-12-31); "
        f"{rows} rows where every label set is defined",
        f"Feature set: {', '.join(cfg['feature_set_names'])} (configs/features.yaml)",
        f"Label strategy: {', '.join(cfg['label_sets'])}; sensitivity {', '.join(cfg['sensitivity']['label_sets'])} "
        f"on {', '.join(cfg['sensitivity']['feature_set_names'])}",
        "Model: xgboost and logistic_regression (all rows reported)",
        f"Seed: {seed_base}-{seed_base + n_seeds - 1} ({n_seeds} seeds; mean and std reported)",
        "Config file: configs/baseline_models.yaml, configs/features.yaml, configs/labelling.yaml",
        "Checkpoint: not saved (models are refit inside code/baseline_models.py)",
        "Test window: 2019-01-01 to 2024-12-31",
        f"Notes: 95% CIs = percentile intervals of the seed-mean metric over {b['n_resamples']} moving-block "
        f"bootstrap resamples (block {b['block_length']} trading days, anomaly clusters with gaps of at most "
        f"{b['cluster_gap']} days kept whole, seed {b['seed']}). Test prevalence (PR-AUC baseline): "
        + ", ".join(f"{k} {100 * v:.2f}%" for k, v in prev.items()) + ". Grid search on validation only.",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--labelling-config", default="configs/labelling.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    feat_cfg = load_yaml(args.features_config)
    lab_cfg = load_yaml(args.labelling_config)
    seeds = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["n"])]

    features, groups, labels, panel = load_inputs(args.index, feat_cfg)
    check_inputs(args.index, features, groups, labels, panel, lab_cfg)
    all_labels = cfg["label_sets"] + cfg["sensitivity"]["label_sets"]
    defined = labels.loc[features.index, all_labels].notna().all(axis=1)
    print(f"Rows: {len(features)} feature rows, {int(defined.sum())} with every label set defined "
          f"(dropped {int((~defined).sum())})")
    features = features.loc[defined]
    tr, va, te = split_index(features.index, cfg["split"])
    print(f"Rows: train {tr.sum()}, validation {va.sum()}, test {te.sum()}")

    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    y_all = {lab: labels.loc[features.index, lab].astype(int).to_numpy() for lab in all_labels}
    samples = {lab: block_bootstrap_samples(y_all[lab][te].astype(bool), b["block_length"], b["cluster_gap"],
                                            b["n_resamples"], rng) for lab in all_labels}

    combos = [(fs, lab) for fs in cfg["feature_set_names"] for lab in cfg["label_sets"]]
    combos += [(fs, lab) for fs in cfg["sensitivity"]["feature_set_names"] for lab in cfg["sensitivity"]["label_sets"]]
    columns = {fs: resolve_feature_set(feat_cfg["feature_sets"][fs], groups) for fs, _ in combos}
    results, grid_all = [], []
    predictions = pd.DataFrame({f"y__{lab}": y_all[lab][te] for lab in all_labels}, index=features.index[te])
    for (fs, label), model_name in itertools.product(combos, cfg["models"]):
        cols = columns[fs]
        twin = next((other for other in cfg["feature_set_names"]
                     if other != fs and columns.get(other) == cols
                     and cfg["feature_set_names"].index(other) < cfg["feature_set_names"].index(fs)), None)
        row = {"model": model_name, "feature_set": fs, "label": label, "n_features": len(cols)}
        if twin:
            print(f"Skipping {model_name} | {fs} | {label}: same columns as {twin}", flush=True)
            results.append({**row, "note": f"same columns as {twin}: no foreign session closes before the target's"})
            continue
        X = features[cols].to_numpy(float)
        y = y_all[label]
        print(f"Running {model_name} | {fs} ({len(cols)} features) | {label} ...", flush=True)

        summary, grid_df, p_tests, thrs = run_combination(
            X, y, (tr, va, te), model_name, cfg["models"][model_name], seeds, cfg
        )
        ci = bootstrap_ci(y[te], p_tests, thrs, samples[label], b["confidence"])
        results.append({**row, **summary, **ci, "note": ""})
        predictions[f"p__{label}__{fs}__{model_name}"] = np.mean(p_tests, axis=0)
        grid_df.insert(0, "model", model_name)
        grid_df.insert(1, "feature_set", fs)
        grid_df.insert(2, "label", label)
        grid_all.append(grid_df)

    out = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("thesis_tables", exist_ok=True)

    out.to_csv(f"results/baseline_models_{args.index}.csv", index=False)
    predictions.rename_axis("date").to_csv(f"results/test_predictions_{args.index}.csv")
    pd.concat(grid_all).to_csv(f"logs/baseline_models_{args.index}_grid.csv", index=False)
    with open(f"thesis_tables/baseline_models_{args.index}_settings.txt", "w") as f:
        f.write(settings_text(args.index, out.dropna(subset=["pr_auc_mean"]), cfg, int(defined.sum()),
                              len(seeds), seeds[0], feat_cfg["panel_path"]))

    show = out.dropna(subset=["pr_auc_mean"])[[
        "model", "feature_set", "label", "pr_auc_mean", "pr_auc_ci_low", "pr_auc_ci_high",
        "f1_mean", "mcc_mean", "pr_auc_prevalence_baseline"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results/baseline_models_{args.index}.csv and results/test_predictions_{args.index}.csv")


if __name__ == "__main__":
    main()
