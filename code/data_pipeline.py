"""
Month 1 pipeline: retrieve equity indices, macro indicators, and EPU data,
then align, compute log-returns and rolling realised volatility.

Implements the "Semester One, Month 1" section of the supervisor's brief:
    - Daily OHLCV for S&P 500 (^GSPC), Nikkei 225 (^N225), FTSE 100 (^FTSE)
      via yfinance, Jan 2000 - Dec 2024, cross-checked against the Kaggle
      Global Stock Indices dataset.
    - FRED series: VIX (VIXCLS), USD/JPY (DEXJPUS), 10-year yield (DGS10),
      TED spread (TEDRATE), and BAA10Y (Moody's Baa yield minus 10-year Treasury), which
      replaces TED as the credit spread (TEDRATE was discontinued on 2022-01-21).
    - US and Japan EPU indices, forward-filled monthly -> daily.
    - Alignment to a common trading calendar (last-observed-carried-forward
      for local holidays), daily log-returns, and rolling 21-day realised
      volatility.

Raw yfinance_*.csv / fred_*.csv files in data/raw/ are loaded from cache when
they exist. Pass --refresh to re-download them (this overwrites the raw files).

Usage:
    python code/data_pipeline.py
    python code/data_pipeline.py --refresh
    python code/data_pipeline.py --fetch-series BAA10Y   # downloads only that FRED series
    python code/data_pipeline.py --epu-us data/raw/US_Policy_Uncertainty_Data.xlsx \\
        --epu-japan data/raw/Japan_Policy_Uncertainty_Data.xlsx \\
        --kaggle data/raw/kaggle_global_indices.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

TICKERS = {"GSPC": "^GSPC", "N225": "^N225", "FTSE": "^FTSE"}
FRED_SERIES = {"VIX": "VIXCLS", "USDJPY": "DEXJPUS", "DGS10": "DGS10", "TED": "TEDRATE",
               "BAA10Y": "BAA10Y"}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guard against yfinance returning MultiIndex columns in some versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _check_cache_range(path, index, start, end, require_end_coverage):
    """
    Stop if a cached raw file does not match the requested date range.
    The first or last day may not be a trading day, so the ends get a
    10-day tolerance. FRED series can be discontinued before `end`
    (TEDRATE stops in 2022), so end coverage is only required for yfinance.
    """
    tol = pd.Timedelta(days=10)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    problem = None
    if index.min() > start_ts + tol:
        problem = f"starts {index.min().date()}, after requested start {start}"
    elif index.max() > end_ts:
        problem = f"runs to {index.max().date()}, after requested end {end}"
    elif require_end_coverage and index.max() < end_ts - tol:
        problem = f"ends {index.max().date()}, before requested end {end}"
    if problem:
        raise RuntimeError(f"Cached file {path} {problem}. Re-run with --refresh to re-download.")


def fetch_yfinance(start, end, refresh=False, sources=None):
    """
    Retrieve daily OHLCV for the three target indices via yfinance.

    An existing data/raw/yfinance_{name}.csv is loaded instead of downloaded
    unless `refresh` is set. Raw files are only written when downloading.
    `sources` (dict name -> "cache" | "download") records where each came from.
    """
    frames = {}
    for name, ticker in TICKERS.items():
        path = f"{RAW_DIR}/yfinance_{name}.csv"
        if os.path.exists(path) and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            _check_cache_range(path, df.index, start, end, require_end_coverage=True)
            print(f"[cache] {path}")
            source = "cache"
        else:
            if not refresh:
                raise FileNotFoundError(f"Missing raw file {path}. Downloads only happen with --refresh.")
            import yfinance as yf

            print(f"[download] {name} ({ticker}) from yfinance...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df is None or df.empty:
                raise RuntimeError(f"yfinance returned no data for {ticker}.")
            df = _flatten_columns(df)
            df = df.loc[:, ["Close", "Volume"]]
            df.columns = [f"{name}_close", f"{name}_volume"]
            df.to_csv(path)
            source = "download"
        if sources is not None:
            sources[path] = source
        frames[name] = df
    return frames


def fetch_fred(start, end, refresh=False, sources=None):
    """
    Retrieve macro/volatility series from FRED.

    An existing data/raw/fred_{name}.csv is loaded instead of downloaded
    unless `refresh` is set. FRED_API_KEY is only needed for files that
    have to be downloaded.
    """
    fred = None
    frames = {}
    for name, series_id in FRED_SERIES.items():
        path = f"{RAW_DIR}/fred_{name}.csv"
        if os.path.exists(path) and not refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = name
            _check_cache_range(path, s.index, start, end, require_end_coverage=False)
            print(f"[cache] {path}")
            source = "cache"
        else:
            if not refresh:
                raise FileNotFoundError(f"Missing raw file {path}. Downloads only happen with --refresh.")
            if fred is None:
                from fredapi import Fred

                api_key = os.environ.get("FRED_API_KEY")
                if not api_key:
                    raise EnvironmentError("Set FRED_API_KEY before fetching FRED series.")
                fred = Fred(api_key=api_key)
            print(f"[download] {name} ({series_id}) from FRED...")
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            s.to_csv(path)
            source = "download"
        if sources is not None:
            sources[path] = source
        frames[name] = s
    return frames


def fetch_single_fred_series(name, start, end):
    """
    Download one FRED series to data/raw/fred_{name}.csv and touch nothing else.
    Refuses to overwrite an existing raw file.
    """
    if name not in FRED_SERIES:
        raise ValueError(f"Unknown series {name}; known: {', '.join(FRED_SERIES)}")
    ensure_dirs()
    path = f"{RAW_DIR}/fred_{name}.csv"
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; delete it manually to re-download.")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("Set FRED_API_KEY before fetching FRED series.")
    from fredapi import Fred

    print(f"[download] {name} ({FRED_SERIES[name]}) from FRED...")
    s = Fred(api_key=api_key).get_series(FRED_SERIES[name], observation_start=start, observation_end=end)
    s.name = name
    s.to_csv(path)
    print(f"Saved {path}: {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}, "
          f"{s.isna().sum()} missing values")
    return s


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
        date = pd.to_datetime(
            pd.DataFrame({"year": df["Year"], "month": df["Month"], "day": 1})
        )
        s = pd.Series(df[value_col].values, index=date)
        # The US file appends a historical block (1900-1984, newest first) after the
        # 1985+ data. reindex(method="ffill") needs a sorted index; on an unsorted one
        # it silently returns the wrong months, which corrupted EPU_US in the
        # 2026-09-16 panel.
        assert not s.index.duplicated().any(), f"{path}: duplicate months"
        return s.sort_index()

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


def build_panel(start, end, epu_us_path=None, epu_japan_path=None, kaggle_path=None, base="GSPC",
                refresh=False, out_path=f"{PROCESSED_DIR}/panel_full.csv"):
    ensure_dirs()

    sources = {}
    price_frames = fetch_yfinance(start, end, refresh=refresh, sources=sources)
    aligned = align_calendar(price_frames, base=base)

    panel = pd.DataFrame(index=aligned[base].index)
    for name, df in aligned.items():
        panel[f"{name}_close"] = df[f"{name}_close"]
        panel[f"{name}_volume"] = df[f"{name}_volume"]
        panel[f"{name}_logret"] = compute_log_returns(df[f"{name}_close"])
        panel[f"{name}_rv21"] = compute_realized_vol(panel[f"{name}_logret"], window=21)

    try:
        fred_frames = fetch_fred(start, end, refresh=refresh, sources=sources)
        for name, s in fred_frames.items():
            panel[name] = s.reindex(panel.index).ffill()
    except EnvironmentError as e:
        print(f"Skipping FRED retrieval: {e}")

    from_cache = [p for p, src in sources.items() if src == "cache"]
    downloaded = [p for p, src in sources.items() if src == "download"]
    print(f"Raw files from cache ({len(from_cache)}): {', '.join(from_cache) or 'none'}")
    print(f"Raw files downloaded ({len(downloaded)}): {', '.join(downloaded) or 'none'}")

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
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download yfinance/FRED data and overwrite data/raw files. "
             "Without it, existing raw files are loaded from cache.",
    )
    parser.add_argument("--out", default=f"{PROCESSED_DIR}/panel_full.csv",
                        help="Output path for the panel (default: data/processed/panel_full.csv).")
    parser.add_argument("--fetch-series", default=None,
                        help="Download only this FRED series (e.g. BAA10Y) to data/raw and exit.")
    args = parser.parse_args()
    if args.fetch_series:
        fetch_single_fred_series(args.fetch_series, args.start, args.end)
        raise SystemExit(0)
    build_panel(args.start, args.end, args.epu_us, args.epu_japan, args.kaggle, args.base,
                refresh=args.refresh, out_path=args.out)
