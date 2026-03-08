"""
data_pipeline.py

Fetches 10-K Item 1A text + financial data for dev subset (100 companies, 2010-2019).
Outputs:
    Data/raw/filings.csv     — ticker, year, filing_date, item1a_text
    Data/raw/targets.csv     — ticker, year, filing_date, volatility, high_volatility
    Data/raw/financials.csv  — ticker, year, debt_equity, roa, current_ratio, log_mktcap

Usage:
    python src/data_pipeline.py

Set EDGAR_IDENTITY env var or edit the constant below before running.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

# SEC EDGAR requires a User-Agent with your name and email.
# Update this or set the EDGAR_IDENTITY environment variable.
EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY", "Juan Sandoval juansd@stanford.edu")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
TICKERS_PATH = os.path.join(DATA_DIR, "sp500_tickers.csv")

TRAIN_YEARS = list(range(2013, 2018))   # 2013–2017 (dev subset)
VAL_YEARS   = list(range(2018, 2020))   # 2018–2019
ALL_YEARS   = TRAIN_YEARS + VAL_YEARS

N_COMPANIES   = 50
SEED          = 42
REQUEST_DELAY = 0.6   # seconds between EDGAR requests (be a good citizen)

os.makedirs(RAW_DIR, exist_ok=True)


# ── Company selection ──────────────────────────────────────────────────────────

def select_companies(n: int = N_COMPANIES, seed: int = SEED) -> pd.DataFrame:
    """Stratified sample of n companies across GICS sectors."""
    df = pd.read_csv(TICKERS_PATH).dropna(subset=["Symbol", "GICS Sector"])

    rng     = np.random.default_rng(seed)
    sectors = df["GICS Sector"].unique()
    n_per_sector = max(1, n // len(sectors))

    selected = []
    for sector in sectors:
        pool = df[df["GICS Sector"] == sector]
        k    = min(n_per_sector, len(pool))
        idx  = rng.choice(len(pool), size=k, replace=False)
        selected.append(pool.iloc[idx])

    result = pd.concat(selected, ignore_index=True)

    # Top-up if rounding left us short
    if len(result) < n:
        remaining = df[~df["Symbol"].isin(result["Symbol"])]
        extra_idx = rng.choice(len(remaining), size=n - len(result), replace=False)
        result = pd.concat([result, remaining.iloc[extra_idx]], ignore_index=True)

    return result.head(n)


# ── EDGAR: fetch Item 1A ───────────────────────────────────────────────────────

def fetch_item1a(ticker: str, year: int, cik=None):
    """
    Fetch Item 1A (Risk Factors) from the 10-K covering fiscal year `year`.
    Most companies file their FY-Y 10-K in early Y+1, so we search filings
    with filing_date in [Jul Y, Jun Y+1].

    Returns (text: str, filing_date: pd.Timestamp) or (None, None) on failure.
    """
    try:
        from edgar import Company, set_identity
        set_identity(EDGAR_IDENTITY)

        # Prefer CIK for reliability; fall back to ticker symbol
        company = Company(int(cik)) if pd.notna(cik) else Company(ticker)
        filings = company.get_filings(form="10-K")

        # Date window: Jul Y  →  Jun Y+1  captures FY-Y regardless of fiscal year end
        window_start = pd.Timestamp(f"{year}-07-01")
        window_end   = pd.Timestamp(f"{year + 1}-06-30")

        target_filing = None
        for filing in filings:
            fd = pd.Timestamp(str(filing.filing_date))
            if window_start <= fd <= window_end:
                target_filing = filing
                filing_date   = fd
                break

        if target_filing is None:
            return None, None

        tenk = target_filing.obj()
        if tenk is None:
            return None, None

        # edgartools v5.x uses .risk_factors for Item 1A text
        item1a_text = None
        for attr in ("risk_factors", "item_1a", "item1a", "Item1A"):
            val = getattr(tenk, attr, None)
            if val is not None:
                item1a_text = str(val).strip()
                break

        if not item1a_text or len(item1a_text) < 200:
            return None, None

        time.sleep(REQUEST_DELAY)
        return item1a_text, filing_date

    except Exception:
        return None, None


# ── Yahoo Finance: realized volatility ────────────────────────────────────────

def compute_volatility(ticker: str, filing_date) -> float | None:
    """
    Annualized realized volatility = std(log_daily_returns) * sqrt(252)
    computed over the 252 trading days immediately following `filing_date`.
    """
    try:
        start = pd.Timestamp(filing_date) + pd.Timedelta(days=1)
        end   = start + pd.Timedelta(days=420)   # fetch extra buffer

        hist = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if len(hist) < 50:
            return None

        prices      = hist["Close"].values[:252]
        log_returns = np.diff(np.log(prices + 1e-10))
        return float(np.std(log_returns, ddof=1) * np.sqrt(252))

    except Exception:
        return None


# ── Yahoo Finance: financial ratios ───────────────────────────────────────────

def fetch_financial_ratios(ticker: str, year: int) -> dict:
    """
    Returns debt_equity, roa, current_ratio, log_mktcap for fiscal year `year`
    sourced from yfinance annual balance sheet / income statement.
    """
    out = {"debt_equity": np.nan, "roa": np.nan,
           "current_ratio": np.nan, "log_mktcap": np.nan}
    try:
        yf_t = yf.Ticker(ticker)
        bs   = yf_t.balance_sheet   # rows = line items, cols = fiscal year-end dates
        inc  = yf_t.financials

        if bs is None or bs.empty:
            return out

        # Find the column whose date is closest to Dec 31 of `year` (max ±18 months)
        target = pd.Timestamp(f"{year}-12-31")

        def nearest_col(df):
            cols  = pd.to_datetime(df.columns)
            diffs = [(abs((c - target).days), orig) for c, orig in zip(cols, df.columns)]
            diffs.sort(key=lambda x: x[0])
            gap, col = diffs[0]
            return col if gap <= 548 else None

        bs_col  = nearest_col(bs)
        inc_col = nearest_col(inc) if inc is not None and not inc.empty else None

        if bs_col is None:
            return out

        bs_data = bs[bs_col]

        # Debt / Equity
        debt   = bs_data.get("Total Debt",
                 bs_data.get("Long Term Debt", np.nan))
        equity = bs_data.get("Stockholders Equity",
                 bs_data.get("Common Stock Equity", np.nan))
        if pd.notna(debt) and pd.notna(equity) and equity != 0:
            out["debt_equity"] = float(debt / equity)

        # Current Ratio
        ca = bs_data.get("Current Assets", np.nan)
        cl = bs_data.get("Current Liabilities", np.nan)
        if pd.notna(ca) and pd.notna(cl) and cl != 0:
            out["current_ratio"] = float(ca / cl)

        # ROA
        if inc_col is not None:
            net_income   = inc[inc_col].get("Net Income", np.nan)
            total_assets = bs_data.get("Total Assets", np.nan)
            if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
                out["roa"] = float(net_income / total_assets)

        # Log Market Cap  (shares × price at start of year)
        try:
            hist = yf.Ticker(ticker).history(
                start=f"{year}-01-01", end=f"{year}-02-15", auto_adjust=True
            )
            if not hist.empty:
                price  = hist["Close"].iloc[0]
                shares = yf_t.info.get("sharesOutstanding")
                if shares:
                    out["log_mktcap"] = float(np.log(shares * price))
        except Exception:
            pass

        return out

    except Exception:
        return out


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline():
    companies = select_companies()
    print(f"Selected {len(companies)} companies across "
          f"{companies['GICS Sector'].nunique()} sectors.\n")

    pairs = [
        (row["Symbol"], yr, row.get("CIK"))
        for _, row in companies.iterrows()
        for yr in ALL_YEARS
    ]

    filings_rows    = []
    targets_rows    = []
    financials_rows = []
    skipped         = 0

    for ticker, year, cik in tqdm(pairs, desc="Fetching filings"):

        # 1. 10-K Item 1A text
        text, filing_date = fetch_item1a(ticker, year, cik)
        if text is None:
            skipped += 1
            continue

        # 2. Realized volatility (required — drop row if unavailable)
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

        # 3. Financial ratios (missing values handled downstream)
        ratios = fetch_financial_ratios(ticker, year)
        financials_rows.append({"ticker": ticker, "year": year, **ratios})

    # ── Binary label: threshold computed on TRAIN set only ──
    df_targets = pd.DataFrame(targets_rows)
    if df_targets.empty:
        print("\nERROR: No observations collected. Check EDGAR connectivity and identity.")
        return
    train_mask  = df_targets["year"].isin(TRAIN_YEARS)
    threshold   = df_targets.loc[train_mask, "volatility"].quantile(0.75)
    df_targets["high_volatility"] = (df_targets["volatility"] > threshold).astype(int)
    df_targets["vol_threshold"]   = threshold

    # ── Save ──
    df_filings    = pd.DataFrame(filings_rows)
    df_financials = pd.DataFrame(financials_rows)

    df_filings.to_csv(os.path.join(RAW_DIR, "filings.csv"),     index=False)
    df_targets.to_csv(os.path.join(RAW_DIR, "targets.csv"),     index=False)
    df_financials.to_csv(os.path.join(RAW_DIR, "financials.csv"), index=False)

    n_train = train_mask.sum()
    n_val   = (~train_mask).sum()
    print(f"\nDone. Collected {len(df_filings)} observations  ({skipped} skipped).")
    print(f"  Train 2010-2017 : {n_train}")
    print(f"  Val   2018-2019 : {n_val}")
    print(f"  Volatility 75th pct (train) : {threshold:.4f}")
    print(f"  High-volatility rate        : {df_targets['high_volatility'].mean():.1%}")


if __name__ == "__main__":
    run_pipeline()
