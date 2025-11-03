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
        return
    #return if up to date

    #try download as newdata
    try:
        new_data = fetch_new_data(ticker, start_date)
        if not new_data.empty:  # Add this check back
            if cached_df.empty:
               combined = new_data
            else:
                combined = pd.concat([cached_df, new_data]).drop_duplicates(subset=["Date"])
            combined.to_csv(get_cached_path(ticker), index=False)
            logging.info(f"Updated {ticker} with {len(new_data)} new rows.")
        else:  # Now this else matches the if above
            logging.warning(f"No new data found for {ticker}.")
    except Exception as e:
        logging.error(f"Failed to update {ticker}: {e}")
def main():
    ensure_cache_dir()
    for ticker in TICKERS:
        update_data(ticker)

if __name__ == "__main__":
    main()