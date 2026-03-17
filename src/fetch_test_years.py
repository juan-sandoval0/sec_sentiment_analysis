"""
fetch 2020-2023 filings and append to existing raw CSVs
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import (
    select_companies,
    fetch_item1a,
    compute_volatility,
    fetch_financial_ratios,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "Data", "raw")

TEST_YEARS = list(range(2020, 2024))


def run():
    existing_filings = pd.read_csv(os.path.join(RAW_DIR, "filings.csv"))
    existing_targets = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))

    companies = select_companies()
    print(f"{len(companies)} companies, years {TEST_YEARS}\n")

    existing_keys = set(zip(existing_filings["ticker"], existing_filings["year"]))
    pairs = [
        (row["Symbol"], yr, row.get("CIK"))
        for _, row in companies.iterrows()
        for yr in TEST_YEARS
        if (row["Symbol"], yr) not in existing_keys
    ]
    print(f"{len(pairs)} pairs to fetch\n")

    filings_rows    = []
    targets_rows    = []
    financials_rows = []
    skipped         = 0

    for ticker, year, cik in tqdm(pairs, desc="Fetching"):
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
        print("Nothing fetched. Check EDGAR connectivity.")
        return

    # use existing train-set threshold to avoid label leakage
    threshold = existing_targets["vol_threshold"].iloc[0]

    new_targets = pd.DataFrame(targets_rows)
    new_targets["high_volatility"] = (new_targets["volatility"] > threshold).astype(int)
    new_targets["vol_threshold"]   = threshold

    new_filings    = pd.DataFrame(filings_rows)
    new_financials = pd.DataFrame(financials_rows)

    (pd.concat([existing_filings, new_filings], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "filings.csv"), index=False))

    (pd.concat([existing_targets, new_targets], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "targets.csv"), index=False))

    existing_fin = pd.read_csv(os.path.join(RAW_DIR, "financials.csv"))
    (pd.concat([existing_fin, new_financials], ignore_index=True)
       .to_csv(os.path.join(RAW_DIR, "financials.csv"), index=False))

    print(f"\nAppended {len(new_filings)} rows ({skipped} skipped).")
    print(f"  high-vol rate: {new_targets['high_volatility'].mean():.1%}")
    print(f"  threshold: {threshold:.4f}")

    final = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    print(f"\n{len(final)} total rows")
    print(final.groupby("year").size().to_string())


if __name__ == "__main__":
    run()
