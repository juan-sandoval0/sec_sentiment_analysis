"""
fetch_test_years.py

Fetches 10-K Item 1A + financials for TEST_YEARS (2020-2023) only,
then appends to the existing raw CSVs (which already have 2013-2019).

Run:
    python src/fetch_test_years.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Reuse helpers from data_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import (
    EDGAR_IDENTITY,
    REQUEST_DELAY,
    select_companies,
    fetch_item1a,
    compute_volatility,
    fetch_financial_ratios,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "Data", "raw")

TEST_YEARS = list(range(2020, 2024))  # 2020–2023


def run():
    # Load existing data to know what's already fetched
    existing_filings = pd.read_csv(os.path.join(RAW_DIR, "filings.csv"))
    existing_targets = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))

    # Use the same 50 companies as the original run
    companies = select_companies()
    print(f"Using {len(companies)} companies. Fetching years {TEST_YEARS}.\n")

    # Build pairs — skip any (ticker, year) already in the CSV
    existing_keys = set(zip(existing_filings["ticker"], existing_filings["year"]))
    pairs = [
        (row["Symbol"], yr, row.get("CIK"))
        for _, row in companies.iterrows()
        for yr in TEST_YEARS
        if (row["Symbol"], yr) not in existing_keys
    ]
    print(f"  {len(pairs)} (ticker, year) pairs to fetch.\n")

    filings_rows    = []
    targets_rows    = []
    financials_rows = []
    skipped         = 0

    for ticker, year, cik in tqdm(pairs, desc="Fetching test filings"):
        text, filing_date = fetch_item1a(ticker, year, cik)
        if text is None:
            skipped += 1
            continue

        vol = compute_volatility(ticker, filing_date)
        if vol is None:
            skipped += 1
            continue

        filings_rows.append({
            "ticker": ticker, "year": year,
            "filing_date": filing_date, "item1a_text": text,
        })
        targets_rows.append({
            "ticker": ticker, "year": year,
            "filing_date": filing_date, "volatility": vol,
        })

        ratios = fetch_financial_ratios(ticker, year)
        financials_rows.append({"ticker": ticker, "year": year, **ratios})

    if not filings_rows:
        print("No new filings fetched. Check EDGAR connectivity.")
        return

    # Use the EXISTING train-set threshold (no leakage)
    threshold = existing_targets["vol_threshold"].iloc[0]

    new_targets = pd.DataFrame(targets_rows)
    new_targets["high_volatility"] = (new_targets["volatility"] > threshold).astype(int)
    new_targets["vol_threshold"]   = threshold

    # Append to existing CSVs
    new_filings    = pd.DataFrame(filings_rows)
    new_financials = pd.DataFrame(financials_rows)

    (pd.concat([existing_filings, new_filings], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "filings.csv"), index=False))

    (pd.concat([existing_targets, new_targets], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "targets.csv"), index=False))

    existing_fin = pd.read_csv(os.path.join(RAW_DIR, "financials.csv"))
    (pd.concat([existing_fin, new_financials], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "financials.csv"), index=False))

    print(f"\nDone. Appended {len(new_filings)} test observations ({skipped} skipped).")
    print(f"  High-volatility rate (test): {new_targets['high_volatility'].mean():.1%}")
    print(f"  Threshold (from train):      {threshold:.4f}")

    # Verify final counts
    final = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    print(f"\nFinal dataset: {len(final)} total rows")
    print(final.groupby("year").size().to_string())


if __name__ == "__main__":
    run()
