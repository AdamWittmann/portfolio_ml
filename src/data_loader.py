import yfinance as yf
import os
import pandas as pd
import datetime as dt
import logging

logging.basicConfig(
    filename='data_loader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from sqlalchemy import create_engine

engine = create_engine("postgresql://adam_ml:Woodruff5614!@localhost:5432/portfolio_ml")

def push_to_db(df: pd.DataFrame, ticker: str):
    """
    Push an in-memory, feature-enriched DataFrame to Postgres.
    No disk I/O; assumes df contains OHLCV + engineered columns.
    """
    if df is None or df.empty:
        print(f"⚠️ Nothing to insert for {ticker}.")
        return

    # Ensure symbol column exists
    if "symbol" not in df.columns:
        df = df.copy()
        df["symbol"] = ticker

    # Rename to match DB schema exactly (note capital V in Volume)
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low":  "low",
        "Close":"close",
        "Volume":"Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Keep only columns the table has (safe subset)
    expected_cols = [
        "symbol","date","open","high","low","close","Volume",
        "sma_20","sma_50","sma_200","daily_return","volatility","drawdown"
    ]
    cols = [c for c in expected_cols if c in df.columns]
    df = df[cols]

    print(f"📤 Pushing {ticker}: {len(df)} rows, {len(cols)} cols -> prices")

    # Atomic bulk insert
    with engine.begin() as conn:
        df.to_sql("prices", conn, if_exists="append", index=False, method="multi")

    print(f"✅ Inserted {len(df)} rows for {ticker}.")


CACHE_DIR = "data_cache"
TICKERS = ["AAPL","MTB","NVDA","TSLA","META","MSFT"]

#Cache Directoy for stock data, inital step before connecting to db
def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
def get_cached_path(ticker):
    return os.path.join(CACHE_DIR, f"{ticker}.csv")

def load_cached_data(ticker):
    path = get_cached_path(ticker)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    return pd.DataFrame()

def fetch_new_data(ticker, start_date):
    df = yf.download(ticker, start=start_date)
    df.reset_index(inplace=True)
    
    # Flatten MultiIndex columns properly
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    return df

def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df

    df["symbol"] = symbol
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()
    df["sma_200"] = df["Close"].rolling(window=200).mean()
    df["daily_return"] = df["Close"].pct_change()
    df["volatility"] = df["daily_return"].rolling(window=20).std()
    df["cummax"] = df["Close"].cummax()
    df["drawdown"] = (df["Close"] - df["cummax"]) / df["cummax"]
    df.drop(columns=["cummax"], inplace=True)
    df.dropna(inplace=True)
    return df

def update_data(ticker):
    print(f"Updating {ticker}...")
    cached_df = load_cached_data(ticker)

    #find last updated date
    if not cached_df.empty:
        last_date = cached_df["Date"].max()
        start_date = last_date + pd.Timedelta(days=1)
    else:
        start_date = dt.datetime(2020,1,1)
    #Dont redownload check if start date is > today dt.date.date()/today()
    if start_date.date() > dt.datetime.today().date():
        print(f"{ticker} is up to date.")
        return cached_df
    #return if up to date

    #try download as newdata
    try:
        new_data = fetch_new_data(ticker, start_date)
        if not new_data.empty:  # Add this check back
            combined = (
                new_data if cached_df.empty
                else pd.concat([cached_df, new_data]).drop_duplicates(subset=["Date"])
            )

            # Compute features
            combined = compute_features(combined, ticker)

            # Save to cache
            combined.to_csv(get_cached_path(ticker), index=False)
            logging.info(f"Updated {ticker} with {len(new_data)} new rows.")

            # ✅ Return for DB push
            return combined
        else:  # Now this else matches the if above
            logging.warning(f"No new data found for {ticker}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to update {ticker}: {e}")
        return pd.DataFrame()
def main():
    ensure_cache_dir()
    for ticker in TICKERS:
        df = update_data(ticker)          # returns engineered DF
        if df is not None and not df.empty:
            push_to_db(df, ticker)        # pass DF + ticker

if __name__ == "__main__":
    main()