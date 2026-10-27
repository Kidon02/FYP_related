"""
Anomaly labelling: three schemes, all causal (configs/labelling.yaml).

  1. Baseline: |rolling 252-day z-score of the log-return| > 2.5.
  2. Regime-dependent threshold: the same z-score, with the threshold set by the
     HMM regime on day t (filtered probabilities).
  3. Regime-standardised: each return standardised by the mean and std of the last
     252 returns before t in the same regime, where the regime for day t is the one
     predicted from information up to t-1; one threshold for all regimes.

The HMM (code/regimes.py) is fit on 2000-2015 only and applied forward with the
forward recursion. The original full-sample HMM (Viterbi path over 2000-2024, which
uses future data) is kept only as a sensitivity analysis: label hmm_fullsample.

Usage:
    python code/labelling.py --index GSPC

Writes:
    results/labels_{IDX}.csv                 per-day labels and regime probabilities
    results/labelling_evaluation_{IDX}.csv   the three schemes, on the days all three define
    results/hmm_sensitivity_{IDX}.csv        causal vs full-sample HMM labels
    results/hmm_params_{IDX}.json            frozen HMM parameters and fit diagnostics
Days where a label is undefined are left empty, never counted as normal.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from regimes import check_filter_causality, fit_hmm, forward_filter, most_probable

CONFIG_PATH = "configs/labelling.yaml"
SCHEMES = {
    "baseline": "Baseline (rolling z-score)",
    "hmm": "Regime-dependent threshold (causal HMM)",
    "regime_std": "Regime-standardised (causal HMM)",
}


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rolling_zscore(returns, window=252):
    """Rolling z-score using only data up to and including day t (no lookahead)."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (returns - mean) / std


def flag(z, threshold):
    """1 where |z| exceeds the threshold (a number or a per-day Series), 0 where not, NaN where undefined."""
    thr = threshold if isinstance(threshold, pd.Series) else pd.Series(float(threshold), index=z.index)
    return (z.abs() > thr).astype(float).where(z.notna() & thr.notna())


def baseline_label(returns, window=252, threshold=2.5):
    """Flag a day anomalous if |rolling z-score| exceeds `threshold`."""
    return flag(rolling_zscore(returns, window=window), threshold)


def regime_conditioned_label(returns, regime, window=252, thresholds=(2.5, 3.0)):
    """
    thresholds = (low_vol_threshold, high_vol_threshold). The high-vol
    threshold should be LARGER than the low-vol one: a 3% move during a
    crisis is less anomalous than the same move in calm markets, so it
    should take a bigger z-score to get flagged when volatility is already high.
    """
    z = rolling_zscore(returns, window=window)
    return flag(z, regime.map({0: thresholds[0], 1: thresholds[1]}))


def regime_standardised_zscore(returns, regime, window_obs=252, min_obs=60):
    """
    z_t = (r_t - mean) / std over the last `window_obs` returns before t that fall in
    the same regime as day t; undefined with fewer than `min_obs` of them. `regime`
    must be known before day t, so pass the one-step predicted regime.
    """
    z = pd.Series(np.nan, index=returns.index)
    for k in regime.dropna().unique():
        r = returns[(regime == k) & returns.notna()]
        past = r.shift(1)                   # earlier same-regime returns only, never r_t
        mean = past.rolling(window_obs, min_periods=min_obs).mean()
        std = past.rolling(window_obs, min_periods=min_obs).std()
        z.loc[r.index] = ((r - mean) / std).to_numpy()
    return z


def regime_standardised_label(returns, regime, window_obs=252, min_obs=60, threshold=2.5):
    """Flag a day anomalous if |regime-standardised z-score| exceeds one `threshold`."""
    return flag(regime_standardised_zscore(returns, regime, window_obs, min_obs), threshold)


def fullsample_regimes(vol_series, n_states=2, random_state=42):
    """
    The original regime labels: an HMM fit on 2000-2024 and decoded with the Viterbi
    path over the whole sample, which uses future observations. Kept only as the
    sensitivity analysis; 0 = low-volatility, 1 = high-volatility.
    """
    from hmmlearn.hmm import GaussianHMM

    clean = vol_series.dropna()
    X = clean.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=200, random_state=random_state
    )
    model.fit(X)
    states = model.predict(X)

    order = np.argsort(model.means_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    states = np.array([remap[s] for s in states])

    regime = pd.Series(states, index=clean.index, name="regime")
    return regime.reindex(vol_series.index)  # NaN during the vol series' own warm-up window


def mean_run_length(label):
    """Average length of consecutive-day runs where label == 1 (undefined days end a run)."""
    label = label.fillna(0).astype(int)
    if label.sum() == 0:
        return 0.0
    diffs = label.diff().fillna(label.iloc[0])
    run_id = (diffs == 1).cumsum()
    run_lengths = label[label == 1].groupby(run_id[label == 1]).size()
    return float(run_lengths.mean())


def concordance(label, events):
    """Fraction of events with at least one flagged day inside the event window, and per-event hits."""
    hits = [bool(label.loc[pd.Timestamp(e["window"][0]):pd.Timestamp(e["window"][1])].fillna(0).sum() > 0)
            for e in events]
    return float(np.mean(hits)), hits


def build_labels(panel, target, cfg):
    r = panel[f"{target}_logret"]
    vol = panel[f"{target}_{cfg['hmm']['observation']}"]
    h, rt, rs = cfg["hmm"], cfg["regime_threshold"], cfg["regime_standardised"]
    w = cfg["zscore_window"]

    params, info = fit_hmm(vol, h["fit_window"], h["n_states"], h["covariance_type"],
                           h["n_iter"], h["random_state"])
    check_filter_causality(vol, params, n_checks=h["causality_checks"])
    filt, pred, _ = forward_filter(vol, params)
    regime_filt, regime_pred = most_probable(filt), most_probable(pred)
    regime_full = fullsample_regimes(vol, h["n_states"], h["random_state"])

    labels = pd.DataFrame({
        "baseline": baseline_label(r, w, cfg["baseline"]["threshold"]),
        "hmm": regime_conditioned_label(r, regime_filt, w, tuple(rt["thresholds"])),
        "regime_std": regime_standardised_label(r, regime_pred, rs["window_obs"], rs["min_obs"],
                                                rs["threshold"]),
        "hmm_fullsample": regime_conditioned_label(r, regime_full, w, tuple(rt["thresholds"])),
        "p_high_filtered": filt.iloc[:, -1],
        "regime_filtered": regime_filt,
        "p_high_predicted": pred.iloc[:, -1],
        "regime_predicted": regime_pred,
        "regime_fullsample": regime_full,
    })
    return labels, params, info


def label_summary(lab, events):
    conc, hits = concordance(lab, events)
    return {
        "anomaly_days": int(lab.sum()),
        "prevalence_pct": round(100 * lab.mean(), 2),
        "mean_run_length_days": round(mean_run_length(lab), 2),
        "concordance_pct": round(100 * conc, 1),
        "events_missed": "; ".join(e["name"] for e, hit in zip(events, hits) if not hit) or "none",
    }


def evaluation_table(labels, events):
    """The three schemes, each on the same days: those where all three are defined."""
    common = labels[list(SCHEMES)].notna().all(axis=1)
    days = labels.index[common]
    return pd.DataFrame([
        {"scheme": name, "label": col, "days": len(days), "from": days.min().date(), "to": days.max().date(),
         **label_summary(labels.loc[common, col], events)}
        for col, name in SCHEMES.items()
    ])


def sensitivity_table(labels, events):
    """Regime-dependent threshold labels from the causal HMM vs the full-sample HMM, on the same days."""
    common = labels[["hmm", "hmm_fullsample"]].notna().all(axis=1)
    sub = labels.loc[common]
    regime_agree = 100 * (sub["regime_filtered"] == sub["regime_fullsample"]).mean()
    label_agree = 100 * (sub["hmm"] == sub["hmm_fullsample"]).mean()
    rows = []
    for col, regime_col, name in [
        ("hmm", "regime_filtered", "Causal HMM (fit 2000-2015, filtered)"),
        ("hmm_fullsample", "regime_fullsample", "Full-sample HMM (fit 2000-2024, Viterbi)"),
    ]:
        rows.append({"hmm_fit": name, "label": col, "days": int(common.sum()),
                     "high_vol_days_pct": round(100 * sub[regime_col].mean(), 1),
                     **label_summary(sub[col], events),
                     "regime_agreement_pct": round(regime_agree, 1),
                     "label_agreement_pct": round(label_agree, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="GSPC", choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True)
    labels, params, info = build_labels(panel, args.index, cfg)
    events = cfg["crisis_events"]

    print(f"HMM fit {info['fit_start']} to {info['fit_end']} ({info['fit_days']} days), "
          f"converged={info['converged']} after {info['iterations']} iterations")
    print(f"State means (rv21): {np.round(params['means'], 5)}, "
          f"P(stay): {np.round(np.diag(params['transmat']), 4)}")
    table = evaluation_table(labels, events)
    sens = sensitivity_table(labels, events)
    print("\n" + table.drop(columns=["label"]).to_string(index=False))
    print("\nSensitivity, causal vs full-sample HMM:\n" + sens.drop(columns=["label"]).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    labels.to_csv(f"results/labels_{args.index}.csv")
    table.to_csv(f"results/labelling_evaluation_{args.index}.csv", index=False)
    sens.to_csv(f"results/hmm_sensitivity_{args.index}.csv", index=False)
    with open(f"results/hmm_params_{args.index}.json", "w") as f:
        json.dump({**{k: np.asarray(v).tolist() for k, v in params.items()}, **info}, f, indent=2)
    print(f"\nSaved results/labels_{args.index}.csv, labelling_evaluation_{args.index}.csv, "
          f"hmm_sensitivity_{args.index}.csv and hmm_params_{args.index}.json")
