import yfinance as yf
import pandas as pd
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------

TICKER = "SPY"
START_DATE = "2015-01-01"

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spy_prices.csv"


def download_data():
    print("Downloading SPY data from Yahoo Finance...")

    df = yf.download(
        TICKER,
        start=START_DATE,
        progress=False
    )

    df.reset_index(inplace=True)

    print(f"Downloaded {len(df)} rows")

    return df


def save_data(df):
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")


def main():

    df = download_data()

    save_data(df)

    print("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()