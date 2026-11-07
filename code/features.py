"""
Feature engineering for anomaly detection.

Usage:
    python code/features.py --index GSPC

Reads data/processed/panel_full.csv and configs/features.yaml, and writes:
    data/processed/features_{IDX}.csv         wide: one column per feature
    data/processed/feature_groups_{IDX}.csv   feature -> group
    results/session_close_times.csv           exchange closing times in UTC, with DST
    results/lag0_ordering_{IDX}.csv           which foreign sessions close before the target's

Every feature for day t uses only information available before the target's close
on day t: data dated t-1 or earlier, plus the same-day (lag-0) log-return and
realised volatility of a foreign index whose session closes before the target's on
every date. Labels are never read here. Rows where the rolling z-score is undefined,
or any feature is undefined, are dropped.
"""

import argparse
import os
from datetime import datetime
from fnmatch import fnmatchcase
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = "configs/features.yaml"
LABEL_TOKENS = {"baseline", "hmm", "regime"}


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_feature_set(spec, groups):
    """Columns of a feature set, in file order. Entries are group names or fnmatch patterns."""
    return [c for c in groups.index if any(groups[c] == e or fnmatchcase(c, e) for e in spec)]


# ---------- session timing (decides which foreign data is known at the target's close) ----------

def session_close_utc(dates, session):
    """UTC closing time of an exchange's session on each local trading date."""
    tz = ZoneInfo(session["timezone"])
    changes = sorted((pd.Timestamp(d), t) for d, t in (session.get("close_from") or {}).items())
    closes = []
    for d in dates:
        close = session["close"]
        for start, t in changes:
            if d >= start:
                close = t
        hour, minute = map(int, close.split(":"))
        closes.append(pd.Timestamp(datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz)).tz_convert("UTC"))
    return pd.Series(closes, index=dates)


def lag0_ordering(target, cfg, dates):
    """
    For each foreign index, whether its session closes before the target's on the
    same date. The answer must be the same on every date, through every DST change,
    otherwise this fails. Returns ({index: bool}, table).
    """
    target_close = session_close_utc(dates, cfg["sessions"][target])
    allowed, rows = {}, []
    for idx in cfg["universe"]:
        if idx == target:
            continue
        gap = (target_close - session_close_utc(dates, cfg["sessions"][idx])).dt.total_seconds() / 3600
        first = int((gap > 0).sum())
        assert first in (0, len(dates)), f"{idx} closes before {target} on only {first} of {len(dates)} dates"
        allowed[idx] = first == len(dates)
        rows.append({
            "target": target, "foreign": idx, "days": len(dates), "foreign_closes_first_days": first,
            "gap_hours_min": round(float(gap.min()), 2), "gap_hours_max": round(float(gap.max()), 2),
            "latest_foreign_session_at_target_close": "same date (lag 0)" if allowed[idx] else "previous date (lag 1)",
        })
    return allowed, pd.DataFrame(rows)


def session_close_table(cfg, dates):
    """Each exchange's closing time in local time and the UTC times it maps to over the panel's dates."""
    rows = []
    for idx, s in cfg["sessions"].items():
        counts = session_close_utc(dates, s).dt.strftime("%H:%M").value_counts().sort_index()
        changes = "; ".join(f"{t} from {d}" for d, t in (s.get("close_from") or {}).items()) or "none"
        rows.append({
            "index": idx, "exchange": s["exchange"], "local_close": s["close"], "local_close_changes": changes,
            "time_zone": s["timezone"],
            "utc_close": "; ".join(f"{t} UTC ({n} days)" for t, n in counts.items()),
            "daylight_saving": s["dst"],
        })
    return pd.DataFrame(rows)


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

def build_raw_features(panel, target, cfg, lag0):
    """
    Return (features, groups) with features unfiltered (warm-up NaNs kept).
    Every series is shifted by at least one day before use, except the lag-0
    series of foreign indices whose session closes before the target's (lag0[idx]).
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
        if lag0[idx]:  # this session closes before the target's on the same date
            for col in cfg["lag0_features"]:
                add(f"for_{idx}_{col}_lag0", "foreign_lag0", panel[f"{idx}_{col}"])

    # Macro. macro.transform "stationary" (default since Nov 2026, supervisor request):
    # daily changes of DGS10 and BAA10Y, and EPU as log of the published value
    # standardised by a trailing window. "levels" keeps the earlier definitions.
    mcfg = cfg.get("macro", {"transform": "levels", "credit_spread": None})
    lm = L["usdjpy"]
    usd = panel["USDJPY"]
    add(f"mac_usdjpy_logchg_lag{lm}", "macro", np.log(usd.shift(lm) / usd.shift(lm + 1)))
    rate_cols = [("DGS10", L["dgs10"])]
    if mcfg.get("credit_spread"):
        rate_cols.append((mcfg["credit_spread"], L["credit_spread"]))
    for col, k in rate_cols:
        if mcfg["transform"] == "stationary":
            add(f"mac_{col.lower()}_chg_lag{k}", "macro", panel[col].shift(k) - panel[col].shift(k + 1))
        else:
            add(f"mac_{col.lower()}_lag{k}", "macro", panel[col].shift(k))
    for col in ["EPU_US", "EPU_JP"]:
        epu = lagged_epu(panel[col], cfg["epu"]["publication_lag_months"], L["epu_day"])
        if mcfg["transform"] == "stationary":
            # epu at t is already known at t-1 (month M-1 value, lagged one day), so the
            # trailing window ending at t uses information up to t-1 only.
            w = mcfg["epu_standardise_window"]
            x = np.log(epu)
            z = (x - x.rolling(w).mean()) / x.rolling(w).std()
            add(f"mac_{col.lower()}_logz_lag{L['epu_day']}", "macro", z)
        else:
            add(f"mac_{col.lower()}_lag{L['epu_day']}", "macro", epu)

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


def _with_noise(panel, rows, cols, rng):
    """Copy of the panel with the given rows of the given columns replaced by positive noise."""
    pert = panel.copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(pert[col]):
            pert[col] = pert[col].astype(float)
            mu, sd = panel[col].mean(), panel[col].std()
            pert.loc[rows, col] = np.abs(rng.normal(mu, sd, size=int(rows.sum())))  # keeps volume positive
    return pert


def check_no_lookahead(panel, target, cfg, lag0, n_checks=3, seed=0):
    """
    At several dates T:
      (a) overwrite every row after T with noise: features dated T or earlier must not
          change, and later ones must, so the test is not vacuous;
      (b) overwrite day T of the target's own columns and of every non-index column
          (VIX, macro, EPU): features at T must not change;
      (c) overwrite day T of each foreign index: features at T must change if that index
          is lag-0 for the target, and must not change if it is not.
    """
    base, _ = build_raw_features(panel, target, cfg, lag0)
    rng = np.random.default_rng(seed)
    own = {idx: [c for c in panel.columns if c.startswith(f"{idx}_")] for idx in cfg["universe"]}
    shared = [c for c in panel.columns if not any(c in cols for cols in own.values())]

    def same(a, b):
        return np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True).all()

    n = len(panel)
    positions = [int(n * q) for q in np.linspace(0.3, 0.85, n_checks)]
    for pos in positions:
        T = panel.index[pos]
        new, _ = build_raw_features(_with_noise(panel, panel.index > T, panel.columns, rng), target, cfg, lag0)
        assert same(base.loc[:T], new.loc[:T]), f"look-ahead: features at or before {T.date()} changed"
        assert not same(base.loc[T:].iloc[1:], new.loc[T:].iloc[1:]), f"look-ahead test vacuous at {T.date()}"

        day = panel.index == T
        new, _ = build_raw_features(_with_noise(panel, day, own[target] + shared, rng), target, cfg, lag0)
        assert same(base.loc[[T]], new.loc[[T]]), f"same-day target or macro data used at {T.date()}"

        for idx in cfg["universe"]:
            if idx == target:
                continue
            new, _ = build_raw_features(_with_noise(panel, day, own[idx], rng), target, cfg, lag0)
            used = not same(base.loc[[T]], new.loc[[T]])
            assert used == lag0[idx], f"same-day {idx} data used={used} at {T.date()}, lag-0 allowed={lag0[idx]}"
    lag0_from = [idx for idx, ok in lag0.items() if ok] or ["none"]
    print(f"No-lookahead check passed at {len(positions)} dates "
          f"(same-day foreign data used only from: {', '.join(lag0_from)}).")


def build_feature_table(panel, target, cfg, lag0):
    raw, groups = build_raw_features(panel, target, cfg, lag0)
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

    lag0, ordering = lag0_ordering(args.index, cfg, panel.index)
    print("Session ordering on the panel's dates (gap = target close minus foreign close):")
    print(ordering.to_string(index=False))
    check_no_lookahead(panel, args.index, cfg, lag0)
    features, groups = build_feature_table(panel, args.index, cfg, lag0)

    for name, spec in cfg["feature_sets"].items():
        print(f"Feature set {name}: {len(resolve_feature_set(spec, groups))} features")
    print(groups.value_counts().to_string())
    print(f"Final rows: {len(features)} ({features.index.min().date()} to {features.index.max().date()})")

    print_range_report(features, cfg)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    features.index.name = "date"
    features.to_csv(f"data/processed/features_{args.index}.csv")
    groups.rename("group").rename_axis("feature").to_csv(f"data/processed/feature_groups_{args.index}.csv")
    session_close_table(cfg, panel.index).to_csv("results/session_close_times.csv", index=False)
    ordering.to_csv(f"results/lag0_ordering_{args.index}.csv", index=False)
    print(f"\nSaved data/processed/features_{args.index}.csv, feature_groups_{args.index}.csv, "
          f"results/session_close_times.csv and results/lag0_ordering_{args.index}.csv")
