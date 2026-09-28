"""
Month 1 pipeline: retrieve equity indices, macro indicators, and EPU data,
then align, compute log-returns and rolling realised volatility.

Implements the "Semester One, Month 1" section of the supervisor's brief:
    - Daily OHLCV for S&P 500 (^GSPC), Nikkei 225 (^N225), FTSE 100 (^FTSE)
      via yfinance, Jan 2000 - Dec 2024, cross-checked against the Kaggle
      Global Stock Indices dataset.
    - FRED series: VIX (VIXCLS), USD/JPY (DEXJPUS), 10-year yield (DGS10),
      TED spread (TEDRATE).
    - US and Japan EPU indices, forward-filled monthly -> daily.
    - Alignment to a common trading calendar (last-observed-carried-forward
      for local holidays), daily log-returns, and rolling 21-day realised
      volatility.

Usage:
    python code/data_pipeline.py
    python code/data_pipeline.py --epu-us data/raw/US_Policy_Uncertainty_Data.xlsx \\
        --epu-japan data/raw/Japan_Policy_Uncertainty_Data.xlsx \\
        --kaggle data/raw/kaggle_global_indices.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

TICKERS = {"GSPC": "^GSPC", "N225": "^N225", "FTSE": "^FTSE"}
FRED_SERIES = {"VIX": "VIXCLS", "USDJPY": "DEXJPUS", "DGS10": "DGS10", "TED": "TEDRATE"}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def _flatten_columns(df):
    """Guard against yfinance returning MultiIndex columns in some versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_yfinance(start, end):
    """Retrieve daily OHLCV for the three target indices via yfinance."""
    import yfinance as yf

    frames = {}
    for name, ticker in TICKERS.items():
        print(f"Fetching {name} ({ticker}) from yfinance...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = _flatten_columns(df)
        df = df[["Close", "Volume"]].rename(
            columns={"Close": f"{name}_close", "Volume": f"{name}_volume"}
        )
        df.to_csv(f"{RAW_DIR}/yfinance_{name}.csv")
        frames[name] = df
    return frames


def fetch_fred(start, end):
    """Retrieve macro/volatility series from FRED. Requires FRED_API_KEY."""
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("Set FRED_API_KEY before fetching FRED series.")
    fred = Fred(api_key=api_key)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) from FRED...")
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = name
        s.to_csv(f"{RAW_DIR}/fred_{name}.csv")
        frames[name] = s
    return frames


def load_epu(us_xlsx_path, japan_xlsx_path):
    """
    Load the official EPU spreadsheets directly (no manual reformatting
    needed) and forward-fill from monthly to daily frequency.

    Direct downloads (no API, no account needed):
        US:    https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx
        Japan: https://www.policyuncertainty.com/media/Japan_Policy_Uncertainty_Data.xlsx

    Both files have Year/Month columns plus one or more index columns; this
    picks the first non-Year/Month numeric column as the index value. If
    that picks the wrong column for your downloaded version of the file,
    check df.columns after reading and pass a different one explicitly.
    """
    def _parse(path):
        df = pd.read_excel(path)
        df = df.dropna(subset=["Year", "Month"])
        df["Year"] = df["Year"].astype(int)
        df["Month"] = df["Month"].astype(int)
        value_col = next(
            c for c in df.columns
            if c not in ("Year", "Month") and pd.api.types.is_numeric_dtype(df[c])
        )
        print(f"  {path}: using column '{value_col}' as the EPU value")
        date = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
        return pd.Series(df[value_col].values, index=date)

    us = _parse(us_xlsx_path).rename("EPU_US")
    jp = _parse(japan_xlsx_path).rename("EPU_JP")
    return us, jp


def cross_check_kaggle(yfinance_close, kaggle_path, ticker, kaggle_col="close", tolerance=0.01):
    """
    Compare yfinance closing prices against the Kaggle Global Stock Indices
    dataset on overlapping dates, per the brief's consistency-check step.

    The Kaggle file is long-format: one row per (ticker, date) with all
    indices stacked together, so it must be filtered to a single `ticker`
    (e.g. "^GSPC") before comparing — otherwise dates collide across indices.
    Returns the per-day relative difference series.
    """
    kaggle = pd.read_csv(kaggle_path, parse_dates=["date"])
    kaggle = kaggle[kaggle["ticker"] == ticker].set_index("date")
    merged = yfinance_close.to_frame("yfinance_close").join(kaggle[[kaggle_col]], how="inner")
    diff = (merged["yfinance_close"] - merged[kaggle_col]).abs() / merged["yfinance_close"]
    n_bad = int((diff > tolerance).sum())
    print(f"Cross-check ({ticker}): {n_bad} of {len(merged)} overlapping days differ by more than {tolerance:.0%}")
    return diff


def align_calendar(frames, base="GSPC"):
    """
    Align all price frames to a common trading-day calendar, using `base`'s
    trading days as the reference index and forward-filling (last-observed-
    carried-forward) the others onto days that are local holidays elsewhere —
    exactly as specified in the brief.
    """
    base_index = frames[base].index
    return {name: df.reindex(base_index).ffill() for name, df in frames.items()}


def compute_log_returns(price_series):
    """Daily log-return: ln(P_t / P_{t-1})."""
    return np.log(price_series / price_series.shift(1))


def compute_realized_vol(returns, window=21):
    """Rolling realised volatility: rolling std of log-returns over `window` days."""
    return returns.rolling(window).std()


def build_panel(start, end, epu_us_path=None, epu_japan_path=None, kaggle_path=None, base="GSPC"):
    ensure_dirs()

    price_frames = fetch_yfinance(start, end)
    aligned = align_calendar(price_frames, base=base)

    panel = pd.DataFrame(index=aligned[base].index)
    for name, df in aligned.items():
        panel[f"{name}_close"] = df[f"{name}_close"]
        panel[f"{name}_volume"] = df[f"{name}_volume"]
        panel[f"{name}_logret"] = compute_log_returns(df[f"{name}_close"])
        panel[f"{name}_rv21"] = compute_realized_vol(panel[f"{name}_logret"], window=21)

    try:
        fred_frames = fetch_fred(start, end)
        for name, s in fred_frames.items():
            panel[name] = s.reindex(panel.index).ffill()
    except EnvironmentError as e:
        print(f"Skipping FRED retrieval: {e}")

    if epu_us_path and epu_japan_path:
        epu_us, epu_jp = load_epu(epu_us_path, epu_japan_path)
        # method="ffill" here (not a separate .ffill() after reindex) is required:
        # EPU is monthly and panel.index is daily, so a plain reindex would only
        # keep the rare day where a 1st-of-month lands on a trading day, dropping
        # nearly all data before ffill ever runs.
        panel["EPU_US"] = epu_us.reindex(panel.index, method="ffill")
        panel["EPU_JP"] = epu_jp.reindex(panel.index, method="ffill")
    else:
        print("Skipping EPU: pass --epu-us and --epu-japan once downloaded.")

    if kaggle_path:
        # Cross-check all three indices, not just the target — the Kaggle
        # file has every ticker stacked in one CSV, so this is essentially free.
        for name in aligned:
            cross_check_kaggle(aligned[name][f"{name}_close"], kaggle_path, ticker=TICKERS[name])
    else:
        print("Skipping Kaggle cross-check: pass --kaggle once downloaded.")

    out_path = f"{PROCESSED_DIR}/panel_full.csv"
    panel.to_csv(out_path)
    print(f"Saved panel to {out_path} ({len(panel)} rows, {len(panel.columns)} columns)")
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--epu-us", default=None)
    parser.add_argument("--epu-japan", default=None)
    parser.add_argument("--kaggle", default=None)
    parser.add_argument("--base", default="GSPC", choices=list(TICKERS.keys()))
    args = parser.parse_args()
    build_panel(args.start, args.end, args.epu_us, args.epu_japan, args.kaggle, args.base)
