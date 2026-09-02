DATA README
===========

This project does not redistribute raw data. Below are the sources and the
expected local paths the pipeline scripts assume. Run code/data_pipeline.py
to regenerate everything from source.

1. Equity index OHLCV (primary source)
   Source: Yahoo Finance, via the `yfinance` Python library
   Tickers: ^GSPC (S&P 500), ^N225 (Nikkei 225), ^FTSE (FTSE 100)
   Range: 2000-01-01 to present
   Expected path: data/raw/yfinance_{ticker}.csv
   Retrieved by: code/data_pipeline.py --source yfinance

2. Global Stock Indices Historical Data (cross-check source)
   Source: Kaggle, https://www.kaggle.com/datasets/guillemservera/global-stock-indices-historical-data
   Expected path: data/raw/kaggle_global_indices.csv
   Used to sanity-check the yfinance series for the same tickers/dates.

3. Macro & volatility indicators
   Source: FRED, via the `fredapi` Python library (requires FRED_API_KEY env var)
   Series: VIXCLS, DEXJPUS, DGS10, TEDRATE
   Expected path: data/raw/fred_{series}.csv
   Retrieved by: code/data_pipeline.py --source fred

4. Economic Policy Uncertainty (EPU) Index
   Source: https://www.policyuncertainty.com/us_monthly.html (US and Japan monthly series)
   Expected path: data/raw/epu_us.csv, data/raw/epu_japan.csv
   Manually downloaded (no public API) — forward-filled to daily frequency in
   code/data_pipeline.py.

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
code/data_pipeline.py writes the cleaned, aligned, feature-engineered panel to:
   data/processed/panel_full.csv
This is Deliverable 1. Columns and label definitions are documented in the
notebook header of code/feature_engineering.ipynb (create alongside the panel).
