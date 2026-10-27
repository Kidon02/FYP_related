"""
Interim report outputs: the data panel summary table and the regime figure.

Usage:
    python code/figures.py

Writes:
    results/data_panel_summary.csv               series, source, first/last date, frequency, missing %
    results/figures/fig_regimes_{IDX}.png, .pdf  regime shading and flagged anomalies
    thesis_tables/table_data_panel_settings.txt
    thesis_tables/fig_regimes_{IDX}_settings.txt (includes the caption)
Figure rules (supervisor): no colour, no grids, at most 6 ticks per axis, caption not title.
Settings: configs/figures.yaml.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from data_pipeline import load_epu

# name, source, identifier, raw file, column; all daily
DAILY_SERIES = [
    ("S&P 500 close", "Yahoo Finance (yfinance)", "^GSPC", "data/raw/yfinance_GSPC.csv", "GSPC_close"),
    ("S&P 500 volume", "Yahoo Finance (yfinance)", "^GSPC", "data/raw/yfinance_GSPC.csv", "GSPC_volume"),
    ("Nikkei 225 close", "Yahoo Finance (yfinance)", "^N225", "data/raw/yfinance_N225.csv", "N225_close"),
    ("Nikkei 225 volume", "Yahoo Finance (yfinance)", "^N225", "data/raw/yfinance_N225.csv", "N225_volume"),
    ("FTSE 100 close", "Yahoo Finance (yfinance)", "^FTSE", "data/raw/yfinance_FTSE.csv", "FTSE_close"),
    ("FTSE 100 volume", "Yahoo Finance (yfinance)", "^FTSE", "data/raw/yfinance_FTSE.csv", "FTSE_volume"),
    ("VIX", "FRED", "VIXCLS", "data/raw/fred_VIX.csv", "VIX"),
    ("USD/JPY", "FRED", "DEXJPUS", "data/raw/fred_USDJPY.csv", "USDJPY"),
    ("10-year Treasury yield", "FRED", "DGS10", "data/raw/fred_DGS10.csv", "DGS10"),
    ("TED spread (excluded from features)", "FRED", "TEDRATE", "data/raw/fred_TED.csv", "TED"),
]
EPU_FILES = ("data/raw/US_Policy_Uncertainty_Data.xlsx", "data/raw/Japan_Policy_Uncertainty_Data.xlsx")


def data_panel_summary(cfg):
    """
    One row per raw series. Missing % for daily series: share of panel trading days
    (the S&P 500 calendar) with no value on that day before forward-filling, zero
    volume counted as missing. For the monthly EPU series: share of months without a value.
    """
    lo, hi = (pd.Timestamp(d) for d in cfg["window"])
    days = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True).index
    rows = []
    for name, source, ident, path, col in DAILY_SERIES:
        s = pd.read_csv(path, index_col=0, parse_dates=True)[col].loc[lo:hi]
        if col.endswith("_volume"):
            s = s.where(s > 0)
        s = s.dropna()
        rows.append({"series": name, "source": source, "identifier": ident, "frequency": "daily",
                     "first_date": s.index.min().date(), "last_date": s.index.max().date(),
                     "missing_pct": round(100 * (~days.isin(s.index)).mean(), 1)})
    months = pd.date_range(lo, hi, freq="MS")
    for name, s in zip(["EPU United States", "EPU Japan"], load_epu(*EPU_FILES)):
        s = s.loc[lo:hi].dropna()
        rows.append({"series": name, "source": "policyuncertainty.com (Baker, Bloom and Davis)",
                     "identifier": "monthly news-based index", "frequency": "monthly",
                     "first_date": s.index.min().date(), "last_date": s.index.max().date(),
                     "missing_pct": round(100 * (~months.isin(s.index)).mean(), 1)})
    return pd.DataFrame(rows), len(days), days.min().date(), days.max().date()


def _runs(mask):
    """(first, last) dates of each run of consecutive True values."""
    m = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return [(mask.index[a], mask.index[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


def _month_ticks(ax, start, end, max_ticks):
    """Month ticks on fixed calendar months, as few steps as keep the count at or below max_ticks."""
    import matplotlib.dates as mdates

    for step in (1, 2, 3, 4, 6, 12):
        months = list(range(1, 13, step))
        ticks = [d for d in pd.date_range(start, end, freq="MS") if d.month in months]
        if len(ticks) <= max_ticks:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            return
    raise ValueError("window too long for the tick limit")


def regime_figure(cfg, out_base):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    fc = cfg["regime_figure"]
    idx = fc["index"]
    panel = pd.read_csv(cfg["panel_path"], index_col=0, parse_dates=True)
    labels = pd.read_csv(f"results/labels_{idx}.csv", index_col=0, parse_dates=True)
    markers = [("baseline", "o", "Baseline z-score"),
               ("hmm", "^", "Regime-dependent threshold"),
               ("regime_std", "x", "Regime-standardised")]

    fig, axes = plt.subplots(len(fc["windows"]), 1, figsize=(fc["width_in"], fc["height_in"]))
    for ax, (a, b) in zip(np.atleast_1d(axes), fc["windows"]):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        r = 100 * panel.loc[a:b, f"{idx}_logret"]
        lab = labels.loc[a:b]
        half = pd.Timedelta(hours=12)
        for s, e in _runs(lab["regime_filtered"] == 1):
            ax.axvspan(s - half, e + half, color="0.85", lw=0, zorder=0)
        ax.plot(r.index, r.to_numpy(), color="black", lw=0.6, zorder=2)
        for col, marker, _ in markers:
            d = lab.index[lab[col] == 1]
            filled = marker == "x"
            ax.scatter(d, r.loc[d], marker=marker, s=26 if filled else 30, linewidths=0.8, zorder=3,
                       facecolors="black" if filled else "none", edgecolors="black")
        ax.set_xlim(a, b)
        _month_ticks(ax, a, b, fc["max_ticks"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=fc["max_ticks"] - 1))
        ax.set_ylabel("Daily log-return (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        fig.canvas.draw()
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            shown = [t for t in axis.get_majorticklocs() if lim[0] <= t <= lim[1]]
            assert len(shown) <= fc["max_ticks"], "too many ticks"

    handles = [Patch(facecolor="0.85", edgecolor="none", label="High-volatility regime (causal HMM)")]
    handles += [Line2D([], [], ls="none", marker=m, color="black", markerfacecolor="black" if m == "x" else "none",
                       label=name) for _, m, name in markers]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=fc["dpi"])
    plt.close(fig)
    flagged = {col: int((labels.loc[pd.Timestamp(a):pd.Timestamp(b), col] == 1).sum())
               for a, b in fc["windows"] for col, _, _ in markers}
    return flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figures.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("thesis_tables", exist_ok=True)

    table, n_days, d0, d1 = data_panel_summary(cfg)
    table.to_csv("results/data_panel_summary.csv", index=False)
    print(table.to_string(index=False))
    with open("thesis_tables/table_data_panel_settings.txt", "w") as f:
        f.write("\n".join([
            f"Dataset: raw series behind panel_full.csv; panel calendar = {n_days} S&P 500 trading days, {d0} to {d1}",
            "Feature set: n/a (input data)",
            "Label strategy: n/a",
            "Model: n/a",
            "Seed: n/a",
            "Config file: configs/figures.yaml (generated by code/figures.py from data/raw)",
            "Checkpoint: n/a",
            "Test window: n/a (whole 2000-2024 range)",
            "Notes: missing % = share of panel trading days with no value that day before forward-filling "
            "(zero volume counts as missing); EPU: share of months. TEDRATE was discontinued on 2022-01-21. "
            "BAMLH0A0HYM2 (HY OAS) is not in the panel: FRED serves it only from 2023-10-27.",
        ]) + "\n")

    idx = cfg["regime_figure"]["index"]
    out_base = f"results/figures/fig_regimes_{idx}"
    regime_figure(cfg, out_base)
    windows = " and ".join(f"{a} to {b}" for a, b in cfg["regime_figure"]["windows"])
    caption = (f"S&P 500 daily log-returns around the 2008 financial crisis (top) and the COVID-19 crash (bottom). "
               "Grey shading marks days the causal HMM (fit on 2000-2015 realised volatility, filtered "
               "probabilities) places in the high-volatility regime. Markers show days flagged as anomalous by the "
               "baseline rolling z-score (circles), the regime-dependent threshold (triangles) and the "
               "regime-standardised scheme (crosses).")
    with open(f"thesis_tables/fig_regimes_{idx}_settings.txt", "w") as f:
        f.write("\n".join([
            f"Dataset: {idx} (panel_full.csv), {windows}",
            "Feature set: n/a (labels only)",
            "Label strategy: baseline, hmm (regime-dependent threshold), regime_std; shading = regime_filtered",
            f"Model: 2-state Gaussian HMM on rv21, fit 2000-2015 (results/hmm_params_{idx}.json)",
            "Seed: 42 (HMM random_state)",
            "Config file: configs/labelling.yaml, configs/figures.yaml",
            f"Checkpoint: labels in results/labels_{idx}.csv; figure {out_base}.png and .pdf",
            "Test window: n/a",
            f"Notes: caption: {caption}",
        ]) + "\n")
    print(f"\nSaved results/data_panel_summary.csv and {out_base}.png/.pdf\nCaption: {caption}")


if __name__ == "__main__":
    main()
