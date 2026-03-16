"""
data_pipeline.py

scrapes Item 1A text from EDGAR 10-Ks and grabs volatility + financial ratios
from yahoo finance. basically builds the raw dataset needed before anything else.
outputs three CSVs into Data/raw/
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# SEC requires a user-agent string or requests get blocked
EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY", "Juan Sandoval juansd@stanford.edu")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
TICKERS_PATH = os.path.join(DATA_DIR, "sp500_tickers.csv")

TRAIN_YEARS = list(range(2013, 2018))   # 2013–2017
VAL_YEARS   = list(range(2018, 2020))   # 2018–2019
TEST_YEARS  = list(range(2020, 2024))   # 2020–2023, keeping this locked until final eval
ALL_YEARS   = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

N_COMPANIES   = 50
SEED          = 42
REQUEST_DELAY = 0.6   # EDGAR rate-limits if requests come too fast

os.makedirs(RAW_DIR, exist_ok=True)


def select_companies(n: int = N_COMPANIES, seed: int = SEED) -> pd.DataFrame:
    """pick n companies spread across sectors so the sample isn't all tech"""
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

    # rounding sometimes leaves me a few short, top up to exactly n
    if len(result) < n:
        remaining = df[~df["Symbol"].isin(result["Symbol"])]
        extra_idx = rng.choice(len(remaining), size=n - len(result), replace=False)
        result = pd.concat([result, remaining.iloc[extra_idx]], ignore_index=True)

    return result.head(n)


def fetch_item1a(ticker: str, year: int, cik=None):
    """
    grabs the risk factors section from the 10-K for fiscal year `year`.
    most companies file their FY-Y 10-K in early Y+1, so i search Jul Y to Jun Y+1.
    returns (text, filing_date) or (None, None) if anything goes wrong
    """
    try:
        from edgar import Company, set_identity
        set_identity(EDGAR_IDENTITY)

        # CIK lookup is more reliable than ticker symbol
        company = Company(int(cik)) if pd.notna(cik) else Company(ticker)
        filings = company.get_filings(form="10-K")

        # Jul Y -> Jun Y+1 catches FY-Y regardless of fiscal year end date
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

        # edgartools changed the attribute name at some point, try a few variations
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


def compute_volatility(ticker: str, filing_date) -> float | None:
    """
    annualized realized vol = std(log daily returns) * sqrt(252)
    measured over the 252 trading days right after the filing date
    """
    try:
        start = pd.Timestamp(filing_date) + pd.Timedelta(days=1)
        end   = start + pd.Timedelta(days=420)   # grab extra buffer, trim to 252 later

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


def fetch_financial_ratios(ticker: str, year: int) -> dict:
    """
    debt/equity, roa, current ratio, log market cap for fiscal year `year`
    pulled from yfinance balance sheet / income statement. lots of NaNs, imputed downstream
    """
    out = {"debt_equity": np.nan, "roa": np.nan,
           "current_ratio": np.nan, "log_mktcap": np.nan}
    try:
        yf_t = yf.Ticker(ticker)
        bs   = yf_t.balance_sheet   # rows = line items, cols = fiscal year-end dates
        inc  = yf_t.financials

        if bs is None or bs.empty:
            return out

        # find the column closest to Dec 31 of target year (within 18 months)
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

        debt   = bs_data.get("Total Debt",
                 bs_data.get("Long Term Debt", np.nan))
        equity = bs_data.get("Stockholders Equity",
                 bs_data.get("Common Stock Equity", np.nan))
        if pd.notna(debt) and pd.notna(equity) and equity != 0:
            out["debt_equity"] = float(debt / equity)

        ca = bs_data.get("Current Assets", np.nan)
        cl = bs_data.get("Current Liabilities", np.nan)
        if pd.notna(ca) and pd.notna(cl) and cl != 0:
            out["current_ratio"] = float(ca / cl)

        if inc_col is not None:
            net_income   = inc[inc_col].get("Net Income", np.nan)
            total_assets = bs_data.get("Total Assets", np.nan)
            if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
                out["roa"] = float(net_income / total_assets)

        # log to squish the scale
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

        # 1. grab the risk factors text
        text, filing_date = fetch_item1a(ticker, year, cik)
        if text is None:
            skipped += 1
            continue

        # 2. need volatility too, skip the whole row if this fails
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

        # 3. financial ratios are nice to have but NaNs are fine, imputed later
        ratios = fetch_financial_ratios(ticker, year)
        financials_rows.append({"ticker": ticker, "year": year, **ratios})

    # compute binary label from TRAIN set only, don't leak val/test info
    df_targets = pd.DataFrame(targets_rows)
    if df_targets.empty:
        print("\nERROR: No observations collected. Check EDGAR connectivity and identity.")
        return
    train_mask  = df_targets["year"].isin(TRAIN_YEARS)
    threshold   = df_targets.loc[train_mask, "volatility"].quantile(0.75)
    df_targets["high_volatility"] = (df_targets["volatility"] > threshold).astype(int)
    df_targets["vol_threshold"]   = threshold

    df_filings    = pd.DataFrame(filings_rows)
    df_financials = pd.DataFrame(financials_rows)

    df_filings.to_csv(os.path.join(RAW_DIR, "filings.csv"),     index=False)
    df_targets.to_csv(os.path.join(RAW_DIR, "targets.csv"),     index=False)
    df_financials.to_csv(os.path.join(RAW_DIR, "financials.csv"), index=False)

    val_mask  = df_targets["year"].isin(VAL_YEARS)
    test_mask = df_targets["year"].isin(TEST_YEARS)
    n_train = train_mask.sum()
    n_val   = val_mask.sum()
    n_test  = test_mask.sum()
    print(f"\nDone. Collected {len(df_filings)} observations  ({skipped} skipped).")
    print(f"  Train 2013-2017 : {n_train}")
    print(f"  Val   2018-2019 : {n_val}")
    print(f"  Test  2020-2023 : {n_test}")
    print(f"  Volatility 75th pct (train) : {threshold:.4f}")
    print(f"  High-volatility rate        : {df_targets['high_volatility'].mean():.1%}")


if __name__ == "__main__":
    run_pipeline()
