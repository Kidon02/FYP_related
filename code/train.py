"""
Main training entry point.

Usage:
    python code/train.py --config configs/xgboost_crossmarket.yaml

Loads the processed panel, builds the requested feature set + label strategy,
trains the requested model across n_seeds runs, and writes a checkpoint plus
a training log. Keep this file simple — it's the single place a supervisor or
teammate looks to understand "how was this trained".
"""

import argparse
import pickle
import numpy as np
import yaml

from data_utils import load_panel, build_feature_set, build_labels, walk_forward_split
from model_factory import build_model
from imblearn.over_sampling import SMOTE


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Running experiment: {cfg['experiment_name']}")

    # 1. Load data and build feature/label matrices
    panel = load_panel(cfg["data"]["panel_path"])
    X, y = build_feature_set(
        panel,
        target_index=cfg["data"]["target_index"],
        feature_set=cfg["data"]["feature_set"],
    )
    y = build_labels(panel, strategy=cfg["data"]["label_strategy"])

    # 2. Walk-forward split
    X_train, y_train, X_val, y_val, _, _ = walk_forward_split(X, y, cfg["split"])

    # 3. Handle class imbalance
    if cfg["imbalance"]["method"] == "smote":
        sm = SMOTE(sampling_strategy=cfg["imbalance"]["sampling_strategy"])
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # 4. Train across multiple seeds, keep mean/std for the paper
    seeds = [cfg["evaluation"]["seed_base"] + i for i in range(cfg["evaluation"]["n_seeds"])]
    val_scores = []
    trained_models = []

    for seed in seeds:
        model = build_model(cfg["model"]["type"], cfg["model"]["params"], seed=seed)
        model.fit(X_train, y_train)
        trained_models.append(model)
        # TODO: compute validation PR-AUC/F1/MCC here and append to val_scores

    print(f"Validation scores across {len(seeds)} seeds: "
          f"mean={np.mean(val_scores) if val_scores else 'TODO'}, "
          f"std={np.std(val_scores) if val_scores else 'TODO'}")

    # 5. Save the best (or first, for now) checkpoint
    with open(cfg["output"]["checkpoint_path"], "wb") as f:
        pickle.dump(trained_models[0], f)

    print(f"Checkpoint saved to {cfg['output']['checkpoint_path']}")


if __name__ == "__main__":
    main()
