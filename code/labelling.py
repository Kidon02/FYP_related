"""
Month 2-3 pipeline: anomaly labelling.

Implements both labelling strategies from the supervisor's brief:
  - Baseline: rolling 252-day z-score threshold on daily log-returns.
  - Regime-conditioned: 2-state Gaussian HMM fit on realised volatility,
    with separate z-score thresholds calibrated per regime.

Also builds the labelling evaluation table required by the supervisor's
project-plan feedback: prevalence, mean run length, and concordance with
named crisis events, comparing baseline vs HMM labels.

Usage:
    python code/labelling.py --panel data/processed/panel_full.csv --index GSPC
"""

import argparse

import numpy as np
import pandas as pd

# Six crisis events spanning the dataset's stated regimes (dot-com bust,
# GFC, European debt crisis, COVID, 2022 rate-hike cycle) plus Aug 2015,
# which the brief already names for local SHAP analysis. Placeholder until
# confirmed against the supervisor's actual feedback email — swap these for
# his exact list before this goes in the report.
CRISIS_EVENTS = {
    "Dot-com bust (2002 trough)": "2002-10-09",
    "GFC / Lehman collapse": "2008-09-15",
    "European debt crisis (US downgrade)": "2011-08-08",
    "China devaluation selloff": "2015-08-24",
    "COVID-19 crash": "2020-03-16",
    "2022 rate-hike / UK gilt crisis": "2022-09-23",
}


def rolling_zscore(returns, window=252):
    """Rolling z-score using only trailing data (no lookahead)."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (returns - mean) / std


def baseline_label(returns, window=252, threshold=2.5):
    """Flag a day anomalous if |rolling z-score| exceeds `threshold`."""
    z = rolling_zscore(returns, window=window)
    return (z.abs() > threshold).astype(int)


def fit_hmm_regimes(vol_series, n_states=2, random_state=42):
    """
    Fit a Gaussian HMM on realised volatility and return a per-day regime
    label, ordered so that 0 = low-volatility and 1 = high-volatility
    (hmmlearn does not guarantee state ordering, so this remaps it).
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


def regime_conditioned_label(returns, regime, window=252, thresholds=(2.5, 3.0)):
    """
    thresholds = (low_vol_threshold, high_vol_threshold). The high-vol
    threshold should be LARGER than the low-vol one: a 3% move during a
    crisis is less anomalous than the same move in calm markets, so it
    should take a bigger z-score to get flagged when volatility is already high.
    """
    z = rolling_zscore(returns, window=window)
    label = pd.Series(0, index=returns.index)
    label[(regime == 0) & (z.abs() > thresholds[0])] = 1
    label[(regime == 1) & (z.abs() > thresholds[1])] = 1
    return label


def mean_run_length(label):
    """Average length of consecutive-day runs where label == 1."""
    label = label.fillna(0).astype(int)
    if label.sum() == 0:
        return 0.0
    diffs = label.diff().fillna(label.iloc[0])
    run_id = (diffs == 1).cumsum()
    run_lengths = label[label == 1].groupby(run_id[label == 1]).size()
    return float(run_lengths.mean())


def concordance(label, event_dates, window_days=5):
    """
    Fraction of named events for which at least one day within
    +/- window_days is flagged anomalous.
    """
    idx = label.index
    hits = 0
    for event in event_dates:
        event = pd.Timestamp(event)
        mask = (idx >= event - pd.Timedelta(days=window_days)) & (idx <= event + pd.Timedelta(days=window_days))
        if label[mask].sum() > 0:
            hits += 1
    return hits / len(event_dates)


def build_evaluation_table(
    returns, vol_series, window=252, baseline_threshold=2.5,
    regime_thresholds=(2.5, 3.0), crisis_events=None,
):
    crisis_events = crisis_events or CRISIS_EVENTS
    baseline = baseline_label(returns, window=window, threshold=baseline_threshold)
    regime = fit_hmm_regimes(vol_series)
    hmm_label = regime_conditioned_label(returns, regime, window=window, thresholds=regime_thresholds)

    rows = []
    for name, label in [("Baseline (rolling z-score)", baseline), ("Regime-conditioned (HMM)", hmm_label)]:
        rows.append({
            "method": name,
            "prevalence_pct": round(100 * label.mean(), 2),
            "mean_run_length_days": round(mean_run_length(label), 2),
            "concordance_pct": round(100 * concordance(label, list(crisis_events.values())), 1),
        })
    return pd.DataFrame(rows), baseline, hmm_label, regime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel_full.csv")
    parser.add_argument("--index", default="GSPC", choices=["GSPC", "N225", "FTSE"])
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--baseline-threshold", type=float, default=2.5)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    returns = panel[f"{args.index}_logret"]
    vol = panel[f"{args.index}_rv21"]

    table, baseline, hmm_label, regime = build_evaluation_table(
        returns, vol, window=args.window, baseline_threshold=args.baseline_threshold
    )
    print(table.to_string(index=False))

    import os
    os.makedirs("results", exist_ok=True)
    out_path = f"results/labelling_evaluation_{args.index}.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved evaluation table to {out_path}")

    labels_out = f"results/labels_{args.index}.csv"
    pd.DataFrame({"baseline": baseline, "hmm": hmm_label, "regime": regime}).to_csv(labels_out)
    print(f"Saved per-day labels and regimes to {labels_out}")
