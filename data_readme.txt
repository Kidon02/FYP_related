DATA README
===========

This project does not redistribute raw data. Below are the sources and the
expected local paths the pipeline scripts assume. Run code/data_pipeline.py
to regenerate everything from source.

1. Equity index OHLCV (primary source)
   Source: Yahoo Finance, via the `yfinance` Python library
   Tickers: ^GSPC (S&P 500), ^N225 (Nikkei 225), ^FTSE (FTSE 100)
   Range: 2000-01-01 to 2024-12-31 (the panel ends 2024-12-30: yfinance's end date
   is exclusive)
   Expected path: data/raw/yfinance_{GSPC,N225,FTSE}.csv
   Retrieved by: code/data_pipeline.py (loaded from the data/raw cache; --refresh
   re-downloads)

2. Global Stock Indices Historical Data (cross-check source)
   Source: Kaggle, https://www.kaggle.com/datasets/guillemservera/global-stock-indices-historical-data
   Expected path: data/raw/kaggle_indices_data.csv
   Used to sanity-check the yfinance series for the same tickers/dates.

3. Macro & volatility indicators
   Source: FRED, via the `fredapi` Python library (requires FRED_API_KEY env var)
   Series used in features: VIXCLS (VIX), DEXJPUS (USDJPY), DGS10 (DGS10)
   Retrieved but excluded: TEDRATE (TED). FRED discontinued it on 2022-01-21, so the
   forward-fill would leave it stale at 0.09 for the last 738 rows.
   Replacement for TED (approved by the supervisor, Nov 2026): BAA10Y, Moody's Baa
   corporate yield minus the 10-year Treasury yield (FRED series BAA10Y).
   Why not HY OAS (BAMLH0A0HYM2, the first choice): since April 2026 FRED serves only a
   rolling 3-year window of it (first observation 2023-10-27 on 2026-10-27; 309
   observations in 2000-2024). Missing years are never backfilled, interpolated or
   forward-filled. Evidence: logs/fred_coverage_check_2026-10-27.txt
   (code/check_fred_coverage.py, which any new series should pass first).
   BAA10Y fetched 2026-11-07 with `python code/data_pipeline.py --fetch-series BAA10Y`
   (downloads only this series; no other raw file touched). Raw file: 6522 rows,
   2000-01-03 to 2024-12-31, 6248 valid observations (274 NaN = US bond-market
   holidays, 4.2%). On the 6288 panel (S&P 500) trading days, 49 are missing (0.78%),
   all single days (Columbus Day, Veterans Day, etc.); longest gap 1 trading day.
   These single days are forward-filled from the previous day, like the other FRED series.
   Expected path: data/raw/fred_{VIX,USDJPY,DGS10,TED,BAA10Y}.csv
   Retrieved by: code/data_pipeline.py (loaded from the data/raw cache; --refresh
   re-downloads)

4. Economic Policy Uncertainty (EPU) Index
   Source: https://www.policyuncertainty.com (US and Japan monthly series)
   Expected path: data/raw/US_Policy_Uncertainty_Data.xlsx,
   data/raw/Japan_Policy_Uncertainty_Data.xlsx
   Columns used: News_Based_Policy_Uncert_Index (US); News-based Economic Policy
   Uncertainty Index (Japan). Manually downloaded (no public API) and forward-filled
   to daily frequency in code/data_pipeline.py. The US file appends a historical
   1900-1984 block, newest first, after the 1985+ rows; the pipeline sorts by date
   before forward-filling (fixed 2026-10-27; the 2026-09-16 panel has EPU_US
   misaligned on 6094 of 6288 days).

5. 34-Year Daily Stock Market Data (initial prototyping only)
   Source: Kaggle, https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data
   Expected path: data/raw/kaggle_34yr_data.csv
   Used only for early feature-engineering prototyping before switching to the
   full multi-index panel above — not part of the final reported results.

6. Massive Stock News Analysis DB (optional / stretch extension)
   Source: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
   Expected path: data/raw/news_sentiment.csv
   Only used if pursuing the sentiment-channel extension — not part of the
   core dissertation.

Processed output
----------------
code/data_pipeline.py writes the cleaned, aligned panel (Deliverable 1). The panel
in use is:
   data/processed/panel_full_2026-10-27.csv   (EPU_US fixed; set by --out)
data/processed/panel_full.csv is the 2026-09-16 original, kept unchanged; it differs
only in EPU_US. Columns are documented in code/data_pipeline.py; features and labels
in code/features.py and code/labelling.py.
