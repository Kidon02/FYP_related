"""
Event-by-scheme matrix and matched single-threshold control (labels only; no models).

1. results/tables/table_event_matrix.csv: for each index, labelling scheme and crisis
   window in configs/labelling.yaml, on the days where all three schemes are defined
   (as in table_labelling_evaluation): flagged days inside the window, detected or
   missed, and the distance in trading days from the window to the nearest flagged day
   (0 if a day inside is flagged).
2. results/tables/table_matched_control.csv: for each index and for the hmm and
   regime_std schemes, on rows where the 252-day rolling z-score, regime_filtered and
   the scheme are all defined: the single |z| threshold that flags the same number of
   days as the scheme, the share of its flags in the high-volatility regime
   (regime_filtered = 1), the scheme's share, and the regime's share of days.

Usage:
    python code/event_tables.py
"""

import os

import numpy as np
import pandas as pd
import yaml

from labelling import rolling_zscore

INDICES = ["GSPC", "N225", "FTSE"]
SCHEMES = ["baseline", "hmm", "regime_std"]
MATCHED = ["hmm", "regime_std"]


def load_cfg():
    with open("configs/labelling.yaml") as f:
        return yaml.safe_load(f)


def event_rows(idx, labels, events):
    common = labels[SCHEMES].notna().all(axis=1)
    lab = labels.loc[common, SCHEMES]
    pos = pd.Series(np.arange(len(lab)), index=lab.index)
    rows = []
    for ev in events:
        start, end = pd.Timestamp(ev["window"][0]), pd.Timestamp(ev["window"][1])
        inside = (lab.index >= start) & (lab.index <= end)
        if not inside.any():
            for s in SCHEMES:
                rows.append({"index": idx, "scheme": s, "event": ev["name"], "window_start": start.date(),
                             "window_end": end.date(), "days_in_window": 0, "flagged_days_in_window": None,
                             "detected": "not evaluable (no labelled days)", "distance_trading_days": None,
                             "nearest_flag": None})
            continue
        p0, p1 = pos[inside].iloc[0], pos[inside].iloc[-1]
        for s in SCHEMES:
            flags = lab[s] == 1
            n_in = int(flags[inside].sum())
            fpos = pos[flags]
            if n_in:
                dist, nearest = 0, "inside"
            elif len(fpos):
                d = np.where(fpos < p0, p0 - fpos, fpos - p1)
                j = int(np.argmin(d))
                dist = int(d[j])
                nearest = f"{fpos.index[j].date()} ({'before' if fpos.iloc[j] < p0 else 'after'})"
            else:
                dist, nearest = None, "no flags"
            rows.append({"index": idx, "scheme": s, "event": ev["name"], "window_start": start.date(),
                         "window_end": end.date(), "days_in_window": int(inside.sum()),
                         "flagged_days_in_window": n_in, "detected": "detected" if n_in else "missed",
                         "distance_trading_days": dist, "nearest_flag": nearest})
    return rows


def matched_rows(idx, labels, panel, window):
    z = rolling_zscore(panel[f"{idx}_logret"], window=window).reindex(labels.index)
    rows = []
    for s in MATCHED:
        ok = z.notna() & labels["regime_filtered"].notna() & labels[s].notna()
        az, reg, lab = z[ok].abs(), labels.loc[ok, "regime_filtered"], labels.loc[ok, s]
        k = int(lab.sum())
        thr = float(az.sort_values(ascending=False).iloc[k - 1])  # flags: |z| >= thr
        ctrl = az >= thr
        rows.append({
            "index": idx, "scheme": s, "rows": int(ok.sum()), "scheme_flags": k,
            "matched_threshold_abs_z": round(thr, 4), "control_flags": int(ctrl.sum()),
            "control_share_flags_high_vol": round(float((reg[ctrl] == 1).mean()), 4),
            "scheme_share_flags_high_vol": round(float((reg[lab == 1] == 1).mean()), 4),
            "high_vol_share_of_days": round(float((reg == 1).mean()), 4),
        })
    return rows


def write_settings(name, lines):
    with open(f"thesis_tables/{name}_settings.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cfg = load_cfg()
    panel = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True)
    ev_rows, m_rows = [], []
    for idx in INDICES:
        labels = pd.read_csv(f"results/labels_{idx}.csv", index_col=0, parse_dates=True)
        ev_rows += event_rows(idx, labels, cfg["crisis_events"])
        m_rows += matched_rows(idx, labels, panel, cfg["zscore_window"])
    ev, m = pd.DataFrame(ev_rows), pd.DataFrame(m_rows)
    os.makedirs("results/tables", exist_ok=True)
    ev.to_csv("results/tables/table_event_matrix.csv", index=False)
    m.to_csv("results/tables/table_matched_control.csv", index=False)
    panel_name = os.path.basename(cfg["panel_path"])
    common = [
        "Model: none (labels only)",
        "Seed: none (deterministic; HMM random_state 42 in configs/labelling.yaml)",
        "Config file: configs/labelling.yaml",
        "Checkpoint: none (code/event_tables.py, from results/labels_{IDX}.csv)",
        "Test window: full labelled sample 2000-2024 (not only 2019-2024)",
    ]
    write_settings("table_event_matrix", [
        f"Dataset: S&P 500, Nikkei 225, FTSE 100 ({panel_name}); days where all three schemes are defined",
        "Feature set: none",
        "Label strategy: baseline, hmm, regime_std",
        *common,
        "Notes: crisis windows from configs/labelling.yaml (inclusive, still to be confirmed). Detected = at least one "
        "flagged day inside the window. Distance = trading days (on the common days) from the nearest window edge to "
        "the nearest flagged day; 0 if a day inside is flagged.",
    ])
    write_settings("table_matched_control", [
        f"Dataset: S&P 500, Nikkei 225, FTSE 100 ({panel_name}); rows where the 252-day z-score, regime_filtered and "
        "the scheme are defined",
        "Feature set: none",
        "Label strategy: hmm and regime_std, each against a single |z| threshold flagging the same number of days",
        *common,
        "Notes: matched threshold = k-th largest |z| (k = scheme flags; ties can add flags, see control_flags). "
        "High-vol regime = regime_filtered = 1 (causal HMM fit on 2000-2015, filtered probability on day t).",
    ])
    pd.set_option("display.width", 250)
    print(ev.to_string(index=False))
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
