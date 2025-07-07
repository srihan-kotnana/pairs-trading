import os
import argparse
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "data/"

def fetch_and_save_data(ticker: str, start: str, end: str, interval: str):

    print(f"Fetching {ticker} from {start} to {end} with interval {interval}...")

    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return

        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp'}, inplace=True)
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index=False)
        print(f"Saved {ticker} data to {DATA_DIR}{ticker}.csv")

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")


def main(tickers, start, end, interval):
    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in tickers:
        fetch_and_save_data(ticker, start, end, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data fetcher")
    parser.add_argument(
        "--tickers", nargs="+", required=True,
        help="List of tickers"
    )
    parser.add_argument(
        "--start", type=str, default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval", type=str, default="1d", choices=["1m", "5m", "15m", "1h", "1d"],
        help="Data granularity"
    )

    args = parser.parse_args()
    main(args.tickers, args.start, args.end, args.interval)
