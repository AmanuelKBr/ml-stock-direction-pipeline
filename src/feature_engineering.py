import pandas as pd
from pathlib import Path
import ta

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "spy_prices.csv"
OUTPUT_PATH = BASE_DIR / "data" / "spy_features.csv"


# -----------------------------
# Load Data
# -----------------------------

def load_data():

    print("Loading price data...")

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------
# Feature Engineering
# -----------------------------

def create_features(df):

    print("Creating features...")

    # Daily return
    df["return"] = df["Close"].pct_change()

    # Moving averages (normalized)
    df["ma10"] = df["Close"].rolling(10).mean() / df["Close"]
    df["ma50"] = df["Close"].rolling(50).mean() / df["Close"]

    # Volatility
    df["volatility"] = df["return"].rolling(10).std()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi() / 100

    # Lagged returns
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["return_lag3"] = df["return"].shift(3)

    # Momentum (normalized)
    df["momentum"] = (df["Close"] - df["Close"].shift(10)) / df["Close"]

    # Longer volatility
    df["volatility20"] = df["return"].rolling(20).std()

    # MACD normalized
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd() / df["Close"]

    # Target (5-day direction)
    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    # -----------------------------
    # NEW FEATURES
    # -----------------------------

    # Lagged returns
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["return_lag3"] = df["return"].shift(3)

    # Momentum
    df["momentum"] = df["Close"] - df["Close"].shift(10)

    # Rolling volatility (longer window)
    df["volatility20"] = df["return"].rolling(20).std()

    # MACD indicator
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()

    # -----------------------------
    # Target variable
    # -----------------------------

    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    return df


# -----------------------------
# Save Features
# -----------------------------

def save_features(df):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature dataset saved to {OUTPUT_PATH}")


# -----------------------------
# Main Pipeline
# -----------------------------

def main():

    df = load_data()

    df = create_features(df)

    save_features(df)

    print("Feature engineering completed.")


if __name__ == "__main__":
    main()