"""
Assemble the tables for the supervisor from the per-index result files.

Usage (after labelling.py, features.py, baseline_models.py and dm_tests.py):
    python code/report_tables.py

Writes results/tables/ and one settings file per table in thesis_tables/:
    table_labelling_evaluation.csv   three schemes per index: prevalence, run length, concordance
    table_hmm_sensitivity.csv        causal vs full-sample HMM: label statistics and model PR-AUC
    table_model_results.csv          every model run: mean (std) over seeds, block-bootstrap 95% CI
    table_session_close_times.csv    exchange closing times in UTC, with DST
    table_lag0_ordering.csv          which foreign sessions close before each target's
    table_dm_tests.csv               Diebold-Mariano tests
"""

import os

import pandas as pd
import yaml

INDICES = ["GSPC", "N225", "FTSE"]
INDEX_NAMES = {"GSPC": "S&P 500", "N225": "Nikkei 225", "FTSE": "FTSE 100"}
LABEL_NAMES = {"baseline": "Rolling z-score baseline", "hmm": "Regime-dependent threshold",
               "regime_std": "Regime-standardised", "hmm_fullsample": "Regime-dependent threshold (full-sample HMM)"}
SET_NAMES = {"domestic_only": "Domestic-Only", "cross_market": "Cross-Market (lag 1)",
             "cross_market_lag0": "Cross-Market (lag 0 + 1)", "ablation_foreign": "Domestic + foreign",
             "ablation_macro": "Domestic + macro", "rv_only": "Realised volatility only"}
# Method name and year of publication, as the supervisor requires in every table.
MODEL_NAMES = {"xgboost": "XGBoost, 2016", "logistic_regression": "LR, 1958", "isolation_forest": "iForest, 2008"}
COMMON = [
    "Seed: 42-46 (5 seeds; mean and std over seeds), HMM random_state 42",
    "Checkpoint: not saved (models are refit inside code/baseline_models.py)",
]


def fmt(mean, std=None, lo=None, hi=None):
    out = f"{mean:.3f}"
    if std is not None:
        out += f" ({std:.3f})"
    if lo is not None:
        out += f" [{lo:.3f}, {hi:.3f}]"
    return out


def write(table, name, settings):
    table.to_csv(f"results/tables/{name}.csv", index=False)
    with open(f"thesis_tables/{name}_settings.txt", "w") as f:
        f.write("\n".join(settings) + "\n")
    print(f"\n=== {name}\n" + table.to_string(index=False))


def main():
    os.makedirs("results/tables", exist_ok=True)
    with open("configs/labelling.yaml") as f:
        lab_cfg = yaml.safe_load(f)
    with open("configs/baseline_models.yaml") as f:
        b = yaml.safe_load(f)["bootstrap"]
    events = "; ".join(f"{e['name']} {e['window'][0]} to {e['window'][1]}" for e in lab_cfg["crisis_events"])
    models = pd.concat([pd.read_csv(f"results/baseline_models_{i}.csv").assign(index=i) for i in INDICES])
    forests = [f"results/isolation_forest_{i}.csv" for i in INDICES if os.path.exists(f"results/isolation_forest_{i}.csv")]
    all_runs = pd.concat([models] + [pd.read_csv(f).assign(index=f.split("_")[-1][:-4]) for f in forests])

    # Item 4: three labelling schemes per index.
    ev = pd.concat([pd.read_csv(f"results/labelling_evaluation_{i}.csv").assign(index=INDEX_NAMES[i])
                    for i in INDICES])
    write(ev[["index", "scheme", "days", "from", "to", "prevalence_pct", "mean_run_length_days",
              "concordance_pct", "events_missed"]], "table_labelling_evaluation", [
        "Dataset: S&P 500, Nikkei 225, FTSE 100 (panel_full_2026-11-07.csv); days where all three schemes are defined",
        "Feature set: n/a (labels only)",
        "Label strategy: baseline (|z|>2.5), hmm (2.5/3.0 by filtered regime), regime_std (one threshold 2.5)",
        "Model: 2-state Gaussian HMM on rv21, fit 2000-2015, filtered/predicted probabilities",
        "Seed: HMM random_state 42 (labels are deterministic)",
        "Config file: configs/labelling.yaml",
        "Checkpoint: results/labels_{IDX}.csv, results/hmm_params_{IDX}.json",
        "Test window: n/a (whole labelled sample)",
        f"Notes: concordance = share of the six events with a flagged day in its window. Windows (student's choice): {events}",
    ])

    # Item 2: causal vs full-sample HMM, labels and model PR-AUC on the same rows.
    rows = []
    for i in INDICES:
        sens = pd.read_csv(f"results/hmm_sensitivity_{i}.csv")
        for _, s in sens.iterrows():
            r = {"index": INDEX_NAMES[i], "hmm_fit": s["hmm_fit"], "high_vol_days_pct": s["high_vol_days_pct"],
                 "prevalence_pct": s["prevalence_pct"], "mean_run_length_days": s["mean_run_length_days"],
                 "concordance_pct": s["concordance_pct"], "regime_agreement_pct": s["regime_agreement_pct"]}
            for fs in ["domestic_only", "cross_market"]:
                for model in ["xgboost", "logistic_regression"]:
                    m = models[(models["index"] == i) & (models["label"] == s["label"])
                               & (models["feature_set"] == fs) & (models["model"] == model)].iloc[0]
                    r[f"pr_auc_{model}_{fs}"] = fmt(m["pr_auc_mean"], m["pr_auc_std"])
            rows.append(r)
    write(pd.DataFrame(rows), "table_hmm_sensitivity", [
        "Dataset: S&P 500, Nikkei 225, FTSE 100 (panel_full_2026-11-07.csv)",
        "Feature set: label statistics: none; PR-AUC: domestic_only and cross_market",
        "Label strategy: hmm (causal: fit 2000-2015, filtered) vs hmm_fullsample (fit 2000-2024, Viterbi path)",
        "Model: 2-state Gaussian HMM; xgboost and logistic_regression for PR-AUC",
        COMMON[0], "Config file: configs/labelling.yaml, configs/baseline_models.yaml", COMMON[1],
        "Test window: label statistics on all labelled days; PR-AUC on 2019-01-01 to 2024-12-31",
        "Notes: PR-AUC mean (std) over seeds. The full-sample fit uses future observations and is a sensitivity "
        "check only. regime_agreement_pct = days where both HMMs give the same regime.",
    ])

    # Item 6: every model run.
    ran = all_runs.dropna(subset=["pr_auc_mean"])
    res = pd.DataFrame({
        "index": ran["index"].map(INDEX_NAMES), "label": ran["label"].map(LABEL_NAMES),
        "feature_set": ran["feature_set"].map(SET_NAMES), "model": ran["model"].map(MODEL_NAMES), "n_features": ran["n_features"],
        "prevalence_baseline": ran["pr_auc_prevalence_baseline"].round(4),
        "pr_auc": [fmt(*v) for v in ran[["pr_auc_mean", "pr_auc_std", "pr_auc_ci_low", "pr_auc_ci_high"]].to_numpy()],
        "f1": [fmt(*v) for v in ran[["f1_mean", "f1_std", "f1_ci_low", "f1_ci_high"]].to_numpy()],
        "mcc": [fmt(*v) for v in ran[["mcc_mean", "mcc_std", "mcc_ci_low", "mcc_ci_high"]].to_numpy()],
    })
    skipped = all_runs[all_runs["pr_auc_mean"].isna()]
    write(res, "table_model_results", [
        "Dataset: S&P 500, Nikkei 225, FTSE 100 (panel_full_2026-11-07.csv); rows where every label set is defined",
        "Feature set: domestic_only, cross_market (lag 1), cross_market_lag0, ablation_foreign, ablation_macro, rv_only",
        "Label strategy: baseline, hmm, regime_std; hmm_fullsample as sensitivity",
        "Model: XGBoost (2016), L2 logistic regression (LR, 1958), Isolation Forest (iForest, 2008; unsupervised, "
        "domestic_only/cross_market/cross_market_lag0 only); grid on validation PR-AUC; F1 threshold on validation",
        COMMON[0], "Config file: configs/baseline_models.yaml, configs/features.yaml, configs/labelling.yaml",
        COMMON[1], "Test window: 2019-01-01 to 2024-12-31",
        f"Notes: mean (std) over seeds [block-bootstrap 95% CI of the seed mean: {b['n_resamples']} resamples, "
        f"block {b['block_length']} trading days, anomaly clusters (gaps <= {b['cluster_gap']} days) kept whole]. "
        "prevalence_baseline = PR-AUC of a random classifier. Macro group has no credit spread (TED excluded; "
        "HY OAS unavailable). Not run (identical columns): "
        + ("; ".join(f"{r['index']} {r['feature_set']} {r['label']} {r['model']}" for _, r in skipped.iterrows())
           or "none"),
    ])

    # Item 5: session timing.
    close = pd.read_csv("results/session_close_times.csv")
    write(close, "table_session_close_times", [
        "Dataset: exchange closing times on the panel's 6288 trading days (2000-01-03 to 2024-12-30)",
        "Feature set: decides foreign_lag0 (configs/features.yaml)", "Label strategy: n/a", "Model: n/a",
        "Seed: n/a", "Config file: configs/features.yaml (sessions)", "Checkpoint: n/a", "Test window: n/a",
        "Notes: UTC times from the IANA time zone database via zoneinfo; day counts show how often each UTC "
        "time applies. Generated by code/features.py.",
    ])
    order = pd.concat([pd.read_csv(f"results/lag0_ordering_{i}.csv") for i in INDICES])
    write(order, "table_lag0_ordering", [
        "Dataset: panel trading days 2000-01-03 to 2024-12-30",
        "Feature set: foreign_lag0 per target (configs/features.yaml)", "Label strategy: n/a", "Model: n/a",
        "Seed: n/a", "Config file: configs/features.yaml (sessions)", "Checkpoint: n/a", "Test window: n/a",
        "Notes: gap = target close minus foreign close, hours; lag 0 only if the foreign session closes first "
        "on every date. Generated by code/features.py.",
    ])

    # Item 7: DM tests.
    dm = pd.read_csv("results/dm_tests.csv").dropna(subset=["dm_stat"])
    dm_out = pd.DataFrame({
        "index": dm["index"].map(INDEX_NAMES), "label": dm["label"].map(LABEL_NAMES), "model": dm["model"].map(MODEL_NAMES),
        "comparison": dm["a"].map(SET_NAMES) + " vs " + dm["b"].map(SET_NAMES), "n": dm["n"],
        "mean_brier_a": dm["mean_loss_a"].round(5), "mean_brier_b": dm["mean_loss_b"].round(5),
        "dm_stat": dm["dm_stat"].round(3), "p_value": dm["p_value"].round(4),
        "p_holm": dm["p_holm"].round(4), "lower_loss": dm["favours"].map(SET_NAMES),
    })
    write(dm_out, "table_dm_tests", [
        "Dataset: S&P 500, Nikkei 225, FTSE 100 test predictions (results/test_predictions_{IDX}.csv)",
        "Feature set: domestic_only vs cross_market (lag 1), domestic_only vs cross_market_lag0, "
        "cross_market vs cross_market_lag0",
        "Label strategy: baseline, hmm, regime_std",
        "Model: xgboost and logistic_regression (seed-mean probabilities)",
        COMMON[0], "Config file: configs/dm_tests.yaml", COMMON[1], "Test window: 2019-01-01 to 2024-12-31",
        "Notes: loss = squared error (daily Brier score) of the seed-mean probability after undoing SMOTE's class "
        "balance with the 2000-2015 training prevalence (monotone; PR-AUC unchanged); d = loss_a - loss_b. "
        "Newey-West long-run variance (Bartlett, lags floor(4(n/100)^(2/9)) = 7), Harvey-Leybourne-Newbold factor "
        "for h=1, Student t(n-1), two-sided p-values; p_holm = Holm adjustment across all " + str(len(dm)) + " tests in the "
        "table (one family), p_value unadjusted. After the correction, mean "
        "probabilities are 0.4-1.3x the test prevalence for logistic regression but 0.05-0.8x for XGBoost, so "
        "XGBoost Brier differences partly reflect calibration. Raw-probability tests: results/dm_tests.csv (*_raw).",
    ])


if __name__ == "__main__":
    main()
