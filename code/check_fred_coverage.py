"""
Check that FRED series cover the project date range before anything is built on them.

Usage:
    python code/check_fred_coverage.py BAMLH0A0HYM2 BAA10Y

For each series, prints FRED's metadata range, the first and last observation
actually returned for 2000-01-01 to 2024-12-31, and the opening of FRED's notes.
A series passes only if both ends fall within 10 days of the project range (the
same tolerance as data_pipeline._check_cache_range). Nothing is written to
data/raw. Exits with status 1 if any series fails. Needs FRED_API_KEY.
"""

import argparse
import os
import sys

import pandas as pd

START, END = "2000-01-01", "2024-12-31"
TOLERANCE = pd.Timedelta(days=10)


def check(fred, series_id):
    info = fred.get_series_info(series_id)
    s = fred.get_series(series_id, observation_start=START, observation_end=END).dropna()
    first = s.index.min() if len(s) else None
    last = s.index.max() if len(s) else None
    ok = (first is not None and first <= pd.Timestamp(START) + TOLERANCE
          and last >= pd.Timestamp(END) - TOLERANCE)
    notes = " ".join((info.get("notes") or "").split())
    print(f"{series_id}: {info['title']}")
    print(f"  FRED metadata range: {info['observation_start']} to {info['observation_end']}")
    print(f"  Returned for {START} to {END}: first {first.date() if first is not None else None}, "
          f"last {last.date() if last is not None else None}, {len(s)} observations")
    print(f"  Notes: {notes[:160]}")
    print(f"  {'PASS' if ok else 'FAIL'}: {'covers' if ok else 'does not cover'} {START} to {END}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("series", nargs="+")
    args = parser.parse_args()

    from fredapi import Fred

    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    results = [check(fred, sid) for sid in args.series]
    sys.exit(0 if all(results) else 1)
