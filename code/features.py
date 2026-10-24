"""
Feature engineering for anomaly detection.

Usage:
    python code/features.py --index GSPC

Reads data/processed/panel_full.csv and configs/features.yaml, and writes:
    data/processed/features_{IDX}.csv         wide: one column per feature
    data/processed/feature_groups_{IDX}.csv   feature -> group

Every feature for day t uses only data dated t-1 or earlier. Labels are never
read here. Rows where the rolling z-score is undefined, or any feature is
undefined, are dropped.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = "configs/features.yaml"
LABEL_TOKENS = {"baseline", "hmm", "regime"}


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- technical indicators (all computed on close, unshifted) ----------

def rsi(close, window):
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def bollinger_width(close, window, k, normalise_by_middle):
    """Upper minus lower band (2k standard deviations), optionally over the middle band."""
    middle = close.rolling(window).mean()
    width = 2 * k * close.rolling(window).std()
    return width / middle if normalise_by_middle else width


def macd_signal(close, fast, slow, signal, normalise_by_close):
    """Signal line of MACD(fast, slow), optionally divided by close."""
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return sig / close if normalise_by_close else sig


def lagged_epu(s, publication_lag_months, day_lag):
    """
    Monthly EPU value for month M is published in month M+lag. Map each
    trading day to the latest value published before it, then lag by day_lag.
    """
    month = s.index.to_period("M")
    monthly = s.groupby(month).first()
    published = pd.Series(monthly.values, index=monthly.index + publication_lag_months)
    daily = pd.Series(published.reindex(month).to_numpy(), index=s.index)
    return daily.shift(day_lag)


# ---------- feature construction ----------

def build_raw_features(panel, target, cfg):
    """
    Return (features, groups) with features unfiltered (warm-up NaNs kept).
    Every series is shifted by at least one day before use.
    """
    L = cfg["lags"]
    feats, groups = {}, {}

    def add(name, group, series):
        feats[name] = series
        groups[name] = group

    # Domestic
    r = panel[f"{target}_logret"]
    for k in L["target_logret"]:
        add(f"dom_logret_lag{k}", "domestic", r.shift(k))
    rv = panel[f"{target}_rv21"]
    for k in L["target_rv21"]:
        add(f"dom_rv21_lag{k}", "domestic", rv.shift(k))

    vol = panel[f"{target}_volume"].astype(float)
    if cfg["volume"]["zero_as_missing"]:
        vol = vol.where(vol > 0)
    logvol = np.log(vol.ffill())
    lv = L["volume"]
    w = cfg["volume"]["normalise_trailing_window"]
    if w:
        # Trailing mean covers logvol up to day t-lv, so it is known at t.
        trailing_mean = logvol.shift(lv).rolling(w).mean()
        add(f"dom_logvol_dev_lag{lv}", "domestic", logvol.shift(lv) - trailing_mean)
    else:
        add(f"dom_logvol_lag{lv}", "domestic", logvol.shift(lv))

    vix = panel["VIX"]
    lvx = L["vix"]
    add(f"dom_vix_lag{lvx}", "domestic", vix.shift(lvx))
    add(f"dom_vix_chg_lag{lvx}", "domestic", vix.shift(lvx) - vix.shift(lvx + 1))

    # Foreign (every universe index except the target)
    for idx in cfg["universe"]:
        if idx == target:
            continue
        for k in L["foreign_logret"]:
            add(f"for_{idx}_logret_lag{k}", "foreign", panel[f"{idx}_logret"].shift(k))
        for k in L["foreign_rv21"]:
            add(f"for_{idx}_rv21_lag{k}", "foreign", panel[f"{idx}_rv21"].shift(k))

    # Macro
    lm = L["usdjpy"]
    usd = panel["USDJPY"]
    add(f"mac_usdjpy_logchg_lag{lm}", "macro", np.log(usd.shift(lm) / usd.shift(lm + 1)))
    add(f"mac_dgs10_lag{L['dgs10']}", "macro", panel["DGS10"].shift(L["dgs10"]))
    for col in ["EPU_US", "EPU_JP"]:
        add(
            f"mac_{col.lower()}_lag{L['epu_day']}",
            "macro",
            lagged_epu(panel[col], cfg["epu"]["publication_lag_months"], L["epu_day"]),
        )

    # Technical (on target close, then lagged)
    t = cfg["technical"]
    lt = L["technical"]
    close = panel[f"{target}_close"]
    add(f"tec_rsi{t['rsi_window']}_lag{lt}", "technical", rsi(close, t["rsi_window"]).shift(lt))
    add(
        f"tec_bbwidth{t['bollinger_window']}_lag{lt}",
        "technical",
        bollinger_width(close, t["bollinger_window"], t["bollinger_k"],
                        t["bollinger_normalise_by_middle"]).shift(lt),
    )
    add(
        f"tec_macdsig_lag{lt}",
        "technical",
        macd_signal(close, t["macd_fast"], t["macd_slow"], t["macd_signal"],
                    t["macd_normalise_by_close"]).shift(lt),
    )

    return pd.DataFrame(feats), pd.Series(groups)


def assert_no_label_or_excluded(features, cfg):
    """Fail if a feature name refers to a label, or an excluded input was used."""
    for name in features.columns:
        tokens = set(name.lower().split("_"))
        assert not (tokens & LABEL_TOKENS), f"label-derived feature name: {name}"
        for ex in cfg["excluded_columns"]:
            assert ex.lower() not in name.lower(), f"excluded input {ex} in feature {name}"


def check_no_lookahead(panel, target, cfg, n_checks=3, seed=0):
    """
    Overwrite every panel row dated T or later with noise, rebuild, and check
    that features dated T or earlier are unchanged. Also check that features
    after T do change, so the test is not vacuous.
    """
    base, _ = build_raw_features(panel, target, cfg)
    rng = np.random.default_rng(seed)
    n = len(panel)
    positions = [int(n * q) for q in np.linspace(0.3, 0.85, n_checks)]
    for pos in positions:
        T = panel.index[pos]
        pert = panel.copy()
        for col in pert.columns:
            if pd.api.types.is_numeric_dtype(pert[col]):
                pert[col] = pert[col].astype(float)
                mu, sd = panel[col].mean(), panel[col].std()
                noise = np.abs(rng.normal(mu, sd, size=(pert.index >= T).sum()))  # keeps volume positive
                pert.loc[pert.index >= T, col] = noise
        new, _ = build_raw_features(pert, target, cfg)
        before = base.loc[:T]
        after = new.loc[:T]
        same = np.isclose(before.to_numpy(float), after.to_numpy(float), equal_nan=True)
        assert same.all(), f"look-ahead: features at or before {T.date()} changed"
        changed_after = not np.isclose(base.loc[T:].iloc[1:].to_numpy(float),
                                       new.loc[T:].iloc[1:].to_numpy(float),
                                       equal_nan=True).all()
        assert changed_after, f"look-ahead test vacuous at {T.date()}"
    print(f"No-lookahead check passed at {len(positions)} dates.")


def build_feature_table(panel, target, cfg):
    raw, groups = build_raw_features(panel, target, cfg)
    assert_no_label_or_excluded(raw, cfg)

    z = (panel[f"{target}_logret"] - panel[f"{target}_logret"].rolling(cfg["zscore_window"]).mean()) \
        / panel[f"{target}_logret"].rolling(cfg["zscore_window"]).std()
    z_defined = z.notna()
    feature_defined = raw.notna().all(axis=1)
    keep = z_defined & feature_defined

    print(f"Rows total: {len(raw)}")
    print(f"Dropped, z-score undefined: {int((~z_defined).sum())}")
    print(f"Dropped, z defined but a feature undefined: {int((z_defined & ~feature_defined).sum())}")
    out = raw.loc[keep]

    constant = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"Dropping constant features: {constant}")
        out = out.drop(columns=constant)
        groups = groups.drop(constant)
    else:
        print("No constant features.")

    assert out.notna().all().all()
    return out, groups.loc[out.columns]


def print_range_report(features, cfg):
    train = features.loc[cfg["_train_start"]:cfg["_train_end"]]
    test = features.loc[cfg["_test_start"]:cfg["_test_end"]]
    print("\nRange check, train (2000-2015) vs test (2019-2024):")
    rows = []
    for c in cfg["range_report"]:
        if c not in features.columns:
            print(f"  {c}: not in feature set")
            continue
        rows.append({
            "feature": c,
            "train_min": train[c].min(), "train_max": train[c].max(),
            "test_min": test[c].min(), "test_max": test[c].max(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True)
    panel = panel.drop(columns=cfg["excluded_columns"], errors="ignore")
    cfg["_train_start"], cfg["_train_end"] = "2000-01-01", "2015-12-31"
    cfg["_test_start"], cfg["_test_end"] = "2019-01-01", "2024-12-31"

    check_no_lookahead(panel, args.index, cfg)
    features, groups = build_feature_table(panel, args.index, cfg)

    for name, members in cfg["feature_sets"].items():
        n = int(groups.isin(members).sum())
        print(f"Feature set {name}: {n} features")
    print(groups.value_counts().to_string())
    print(f"Final rows: {len(features)} ({features.index.min().date()} to {features.index.max().date()})")

    print_range_report(features, cfg)

    os.makedirs("data/processed", exist_ok=True)
    features.index.name = "date"
    features.to_csv(f"data/processed/features_{args.index}.csv")
    groups.rename("group").rename_axis("feature").to_csv(f"data/processed/feature_groups_{args.index}.csv")
    print(f"\nSaved data/processed/features_{args.index}.csv and feature_groups_{args.index}.csv")
