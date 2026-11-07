"""
Interim report outputs: the data panel summary table and the regime figure.

Usage:
    python code/figures.py

Writes:
    results/data_panel_summary.csv               series, source, first/last date, frequency, missing %
    results/figures/fig_regimes_{IDX}.png, .pdf  regime shading and flagged anomalies
    results/figures/fig_prauc_{IDX}.png, .pdf    test PR-AUC by method and feature set, 95% CIs
    thesis_tables/table_data_panel_settings.txt
    thesis_tables/fig_regimes_{IDX}_settings.txt (includes the caption)
    thesis_tables/fig_prauc_{IDX}_settings.txt
Figure rules (supervisor): no colour, no grids, at most 6 ticks per axis, caption not title.
Settings: configs/figures.yaml.
"""

import argparse
import os
import textwrap

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


def prauc_figure(cfg, out_base):
    """
    Grouped bars of seed-mean test PR-AUC (one label set, one index): feature sets on the x axis, one
    hatched bar per method, block-bootstrap 95% CIs as error bars, dashed line at the test prevalence.
    Feature sets a method was not run on are left empty and labelled "not run".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from report_tables import MODEL_NAMES, SET_NAMES

    fc = cfg["prauc_figure"]
    idx, label = fc["index"], fc["label"]
    res = pd.concat([pd.read_csv(f"results/baseline_models_{idx}.csv"),
                     pd.read_csv(f"results/isolation_forest_{idx}.csv")])
    res = res[res["label"] == label].set_index(["model", "feature_set"])
    prevalence = res["pr_auc_prevalence_baseline"].unique()
    assert len(prevalence) == 1, "test prevalence differs between rows"
    prevalence = float(prevalence[0])

    sets, models = fc["feature_sets"], fc["models"]
    width = 0.8 / len(models)
    x = np.arange(len(sets))
    fig, ax = plt.subplots(figsize=(fc["width_in"], fc["height_in"]))
    used, missing = [], []
    for j, (model, hatch) in enumerate(zip(models, fc["hatches"])):
        pos = x + (j - (len(models) - 1) / 2) * width
        for p, fs in zip(pos, sets):
            if (model, fs) not in res.index:
                missing.append((model, fs))
                ax.text(p, prevalence + 0.01, "not run", rotation=90, ha="center", va="bottom", fontsize=6)
                continue
            r = res.loc[(model, fs)]
            m = r["pr_auc_mean"]
            ax.bar(p, m, width, facecolor="white", edgecolor="black", hatch=hatch, lw=0.7, zorder=2)
            ax.errorbar(p, m, yerr=[[m - r["pr_auc_ci_low"]], [r["pr_auc_ci_high"] - m]], fmt="none",
                        ecolor="black", elinewidth=0.7, capsize=2, zorder=3)
            used.append({"model": model, "feature_set": fs, "pr_auc_mean": m, "pr_auc_std": r["pr_auc_std"],
                         "ci_low": r["pr_auc_ci_low"], "ci_high": r["pr_auc_ci_high"]})
    ax.axhline(prevalence, color="black", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(SET_NAMES[s], 13) for s in sets], fontsize=7)
    ax.set_ylabel("Test PR-AUC (2019-2024)")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=fc["max_ticks"] - 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    handles = [Patch(facecolor="white", edgecolor="black", hatch=h, label=MODEL_NAMES[m])
               for m, h in zip(models, fc["hatches"])]
    handles.append(Line2D([], [], color="black", ls="--", lw=0.8, label=f"Prevalence ({prevalence:.3f})"))
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=7, ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.canvas.draw()
    for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
        shown = [t for t in axis.get_majorticklocs() if lim[0] <= t <= lim[1]]
        assert len(shown) <= fc["max_ticks"], "too many ticks"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=fc["dpi"])
    plt.close(fig)
    return pd.DataFrame(used), missing, prevalence


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

    fc = cfg["prauc_figure"]
    from report_tables import INDEX_NAMES, LABEL_NAMES, MODEL_NAMES, SET_NAMES

    idx = fc["index"]
    out_base = f"results/figures/fig_prauc_{idx}"
    used, missing, prevalence = prauc_figure(cfg, out_base)
    print(used.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    not_run = "; ".join(f"{MODEL_NAMES[m]} on {SET_NAMES[fs]}" for m, fs in missing) or "none"
    caption = (f"Test PR-AUC (2019-2024) for the {INDEX_NAMES[idx]} with regime-dependent threshold labels "
               "(causal HMM), by feature set and method. Bars show the mean over five seeds; error bars show "
               "block-bootstrap 95% intervals (2000 resamples, 21-day blocks, anomaly clusters kept whole). The "
               f"dashed line is the test prevalence ({prevalence:.3f}), the PR-AUC of a random classifier. "
               "Isolation Forest was run only on the Domestic-Only and Cross-Market feature sets.")
    with open(f"thesis_tables/fig_prauc_{idx}_settings.txt", "w") as f:
        f.write("\n".join([
            f"Dataset: {idx} ({os.path.basename(cfg['panel_path'])}), test rows 2019-2024",
            "Feature set: " + ", ".join(fc["feature_sets"]) + " (configs/features.yaml)",
            f"Label strategy: {fc['label']} ({LABEL_NAMES[fc['label']]}, causal HMM)",
            "Model: " + ", ".join(fc["models"]),
            "Seed: 42-46 (5 seeds; bar = seed mean)",
            "Config file: configs/figures.yaml (prauc_figure), configs/baseline_models.yaml, "
            "configs/isolation_forest.yaml",
            f"Checkpoint: results/baseline_models_{idx}.csv, results/isolation_forest_{idx}.csv; "
            f"figure {out_base}.png and .pdf",
            "Test window: 2019-01-01 to 2024-12-31",
            f"Notes: not run: {not_run}. Caption: {caption}",
        ]) + "\n")
    print(f"Saved {out_base}.png/.pdf\nCaption: {caption}")


if __name__ == "__main__":
    main()
