"""
Diebold-Mariano tests: Domestic-Only vs Cross-Market forecasts on 2019-2024.

Forecast: the seed-mean predicted probability, prior-corrected for SMOTE (the models
train on balanced classes, which inflates raw probabilities): the odds are multiplied
by (training odds / resampled odds), using the 2000-2015 training prevalence. The
correction is monotone, so it leaves rankings and PR-AUC unchanged.

Loss: squared error against the 0/1 label (the daily Brier score). For a pair [A, B],
d_t = loss_A,t - loss_B,t, so a positive mean favours B. The long-run variance of d is
a Newey-West (Bartlett) estimate with floor(4 (n/100)^(2/9)) lags. The statistic is
multiplied by the Harvey-Leybourne-Newbold (1997) small-sample factor
sqrt((n + 1 - 2h + h(h-1)/n) / n), h = 1, and compared with Student's t with n - 1
degrees of freedom (two-sided p-value). Settings: configs/dm_tests.yaml.

Usage:
    python code/dm_tests.py

Reads results/test_predictions_{IDX}.csv, results/labels_{IDX}.csv and
data/processed/features_{IDX}.csv (for the training rows); writes results/dm_tests.csv.
The same tests on raw probabilities are kept as *_raw columns.
"""

import argparse

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def newey_west_variance(d, lags):
    """Bartlett-kernel long-run variance of d (the mean is removed first)."""
    e = d - d.mean()
    n = len(e)
    lrv = e @ e / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return lrv


def dm_test(loss_a, loss_b, h=1):
    """HLN-corrected Diebold-Mariano statistic and two-sided p-value for H0: E[loss_a - loss_b] = 0."""
    d = loss_a - loss_b
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dm = d.mean() / np.sqrt(newey_west_variance(d, lags) / n)
    stat = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "nw_lags": lags, "mean_loss_diff": float(d.mean()), "dm_uncorrected": float(dm),
            "dm_stat": float(stat), "p_value": float(2 * stats.t.sf(abs(stat), df=n - 1))}


def prior_correct(p, train_prevalence, resampled_prevalence):
    """Rescale probabilities from the resampled class balance to the training prevalence."""
    w = (train_prevalence / (1 - train_prevalence)) / (resampled_prevalence / (1 - resampled_prevalence))
    return w * p / (w * p + 1 - p)


def resampled_prevalence(sampling_strategy):
    """Positive share after SMOTE: 'auto' balances the classes; a float r is minority/majority."""
    if sampling_strategy == "auto":
        return 0.5
    return float(sampling_strategy) / (1 + float(sampling_strategy))


def train_prevalences(index, model_cfg):
    """Positive share of each label set on the training rows the models used."""
    rows = pd.read_csv(f"data/processed/features_{index}.csv", index_col=0, parse_dates=True, usecols=[0]).index
    labels = pd.read_csv(f"results/labels_{index}.csv", index_col=0, parse_dates=True)
    all_labels = model_cfg["label_sets"] + model_cfg["sensitivity"]["label_sets"]
    rows = rows[labels.loc[rows, all_labels].notna().all(axis=1).to_numpy()]
    lo, hi = (pd.Timestamp(d) for d in model_cfg["split"]["train"])
    train = rows[(rows >= lo) & (rows <= hi)]
    return {lab: float(labels.loc[train, lab].mean()) for lab in all_labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dm_tests.yaml")
    parser.add_argument("--models-config", default="configs/baseline_models.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    assert cfg["loss"] == "squared_error" and cfg["small_sample_correction"] == "hln"
    assert cfg["probability"] == "prior_corrected"
    pi_s = resampled_prevalence(model_cfg["imbalance"]["sampling_strategy"])

    rows = []
    for index in cfg["indices"]:
        pred = pd.read_csv(f"results/test_predictions_{index}.csv", index_col=0, parse_dates=True)
        pi_train = train_prevalences(index, model_cfg)
        for label in cfg["label_sets"]:
            y = pred[f"y__{label}"].to_numpy(float)
            for model in cfg["models"]:
                for a, b in cfg["pairs"]:
                    ca, cb = f"p__{label}__{a}__{model}", f"p__{label}__{b}__{model}"
                    row = {"index": index, "label": label, "model": model, "a": a, "b": b,
                           "train_prevalence": pi_train[label], "test_prevalence": float(y.mean())}
                    if ca not in pred or cb not in pred:   # e.g. N225 has no distinct lag-0 set
                        rows.append({**row, "note": "not run: feature set identical to another"})
                        continue
                    pa, pb = pred[ca].to_numpy(), pred[cb].to_numpy()
                    raw = dm_test((pa - y) ** 2, (pb - y) ** 2, h=cfg["horizon"])
                    ca_, cb_ = (prior_correct(p, pi_train[label], pi_s) for p in (pa, pb))
                    la, lb = (ca_ - y) ** 2, (cb_ - y) ** 2
                    rows.append({**row, "mean_p_a": float(ca_.mean()), "mean_p_b": float(cb_.mean()),
                                 "mean_loss_a": float(la.mean()), "mean_loss_b": float(lb.mean()),
                                 **dm_test(la, lb, h=cfg["horizon"]),
                                 "mean_loss_diff_raw": raw["mean_loss_diff"], "dm_stat_raw": raw["dm_stat"],
                                 "p_value_raw": raw["p_value"], "note": ""})
    out = pd.DataFrame(rows)
    out["favours"] = np.where(out["mean_loss_diff"] > 0, out["b"], out["a"])
    out.loc[out["mean_loss_diff"].isna(), "favours"] = ""
    out.to_csv("results/dm_tests.csv", index=False)
    show = out.dropna(subset=["dm_stat"])[["index", "label", "model", "a", "b", "mean_loss_a", "mean_loss_b",
                                           "dm_stat", "p_value"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nSaved results/dm_tests.csv")


if __name__ == "__main__":
    main()
