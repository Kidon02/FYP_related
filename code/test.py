"""
Main evaluation entry point.

Usage:
    python code/test.py --config configs/xgboost_crossmarket.yaml \
                         --checkpoint results/checkpoints/xgboost_crossmarket_gspc.pkl

Runs the held-out 2019-2024 test window, reports PR-AUC/F1/MCC, and — if
explainability.run_shap is true in the config — generates global SHAP rankings
and local SHAP/LIME plots for the configured historical events.
"""

import argparse
import pickle
import yaml

from data_utils import load_panel, build_feature_set, build_labels, walk_forward_split
from metrics import pr_auc, f1, mcc
from explain import run_global_shap, run_local_shap, run_lime


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(args.checkpoint, "rb") as f:
        model = pickle.load(f)

    panel = load_panel(cfg["data"]["panel_path"])
    X, _ = build_feature_set(
        panel,
        target_index=cfg["data"]["target_index"],
        feature_set=cfg["data"]["feature_set"],
    )
    y = build_labels(panel, strategy=cfg["data"]["label_strategy"])
    _, _, _, _, X_test, y_test = walk_forward_split(X, y, cfg["split"])

    preds = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    print(f"PR-AUC: {pr_auc(y_test, preds):.3f}")
    print(f"F1:     {f1(y_test, preds):.3f}")
    print(f"MCC:    {mcc(y_test, preds):.3f}")

    if cfg.get("explainability", {}).get("run_shap"):
        run_global_shap(model, X_test, out_dir=f"results/shap/{cfg['experiment_name']}")
        for event_date in cfg["explainability"].get("local_shap_events", []):
            run_local_shap(model, X_test, event_date, out_dir=f"results/shap/{cfg['experiment_name']}")
            run_lime(model, X_test, event_date, out_dir=f"results/shap/{cfg['experiment_name']}")


if __name__ == "__main__":
    main()
