"""
Out-of-range table for the macro features, before and after the stationary transform.

For each macro variable, the feature under macro.transform "levels" (before) and
"stationary" (after): training (2000-2015) and test (2019-2024) minimum and maximum,
and the share of test days outside the training range. USD/JPY was already a log change
before; its raw level is shown as an extra "before" row for reference.

Macro features do not depend on the target index, so they are computed once on the
panel's (S&P 500) trading calendar, over the days where each feature is defined.

Usage:
    python code/macro_range.py
Writes results/tables/table_macro_range.csv and thesis_tables/table_macro_range_settings.txt.
"""

import copy
import os

import pandas as pd

from features import build_raw_features, load_config

TRAIN = ("2000-01-01", "2015-12-31")
TEST = ("2019-01-01", "2024-12-31")


def macro_features(panel, cfg, transform):
    c = copy.deepcopy(cfg)
    c["macro"]["transform"] = transform
    lag0 = {idx: False for idx in c["universe"]}
    feats, groups = build_raw_features(panel, "GSPC", c, lag0)
    return feats[groups.index[groups == "macro"]]


def row(variable, stage, name, s):
    s = s.dropna()
    tr, te = s.loc[TRAIN[0]:TRAIN[1]], s.loc[TEST[0]:TEST[1]]
    lo, hi = tr.min(), tr.max()
    return {
        "variable": variable, "stage": stage, "feature": name,
        "train_min": lo, "train_max": hi, "test_min": te.min(), "test_max": te.max(),
        "test_days": len(te),
        "test_share_below_train_min": (te < lo).mean(),
        "test_share_above_train_max": (te > hi).mean(),
        "test_share_outside_train_range": ((te < lo) | (te > hi)).mean(),
    }


def main():
    cfg = load_config()
    panel = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True)
    panel = panel.drop(columns=cfg["excluded_columns"], errors="ignore")
    before = macro_features(panel, cfg, "levels")
    after = macro_features(panel, cfg, "stationary")

    def var_of(name):
        for v in ["usdjpy", "dgs10", "baa10y", "epu_us", "epu_jp"]:
            if name.startswith(f"mac_{v}_"):
                return v.upper()

    rows = [row("USDJPY", "before (raw level, lag 1)", "USDJPY_lag1", panel["USDJPY"].shift(1))]
    for stage, df in [("before (levels)", before), ("after (stationary)", after)]:
        for name in df.columns:
            rows.append(row(var_of(name), stage, name, df[name]))
    order = {v: i for i, v in enumerate(["USDJPY", "DGS10", "BAA10Y", "EPU_US", "EPU_JP"])}
    table = pd.DataFrame(rows)
    stage_order = {"before (raw level, lag 1)": 0, "before (levels)": 1, "after (stationary)": 2}
    table = table.sort_values(["variable", "stage"],
                              key=lambda s: s.map(order) if s.name == "variable" else s.map(stage_order),
                              kind="stable").reset_index(drop=True)
    table = table.round(4)

    os.makedirs("results/tables", exist_ok=True)
    table.to_csv("results/tables/table_macro_range.csv", index=False)
    m = cfg["macro"]
    settings = [
        f"Dataset: {os.path.basename(cfg['panel_path'])}, S&P 500 trading calendar; days where each feature is defined",
        "Feature set: macro group (configs/features.yaml), under macro.transform levels (before) and stationary (after)",
        "Label strategy: none (input ranges only)",
        "Model: none",
        "Seed: none (deterministic)",
        "Config file: configs/features.yaml",
        "Checkpoint: none (code/macro_range.py)",
        f"Test window: {TEST[0]} to {TEST[1]}; training window {TRAIN[0]} to {TRAIN[1]}",
        f"Notes: stationary = DGS10 and {m['credit_spread']} daily change, USDJPY log change, EPU log of previous-month "
        f"value standardised by trailing {m['epu_standardise_window']}-day mean and std (data up to t-1). "
        "Share outside = share of test days below the training minimum or above the training maximum.",
    ]
    with open("thesis_tables/table_macro_range_settings.txt", "w") as f:
        f.write("\n".join(settings) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
