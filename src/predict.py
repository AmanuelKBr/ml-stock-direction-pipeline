import joblib
import pandas as pd
import yfinance as yf
from pathlib import Path
import ta

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"


# -----------------------------
# Load Model
# -----------------------------

def load_model():

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    return model


# -----------------------------
# Download Latest Market Data
# -----------------------------

def download_latest_data():

    print("Downloading latest SPY data...")

    df = yf.download("SPY", period="3y", interval="1d")

    # Fix for multi-index columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


# -----------------------------
# Feature Engineering
# -----------------------------

def create_features(df):

    df["return"] = df["Close"].pct_change()

    df["ma10"] = df["Close"].rolling(10).mean() / df["Close"]
    df["ma50"] = df["Close"].rolling(50).mean() / df["Close"]

    df["volatility"] = df["return"].rolling(10).std()

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi() / 100

    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["return_lag3"] = df["return"].shift(3)

    df["momentum"] = (df["Close"] - df["Close"].shift(10)) / df["Close"]

    df["volatility20"] = df["return"].rolling(20).std()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd() / df["Close"]

    df.dropna(inplace=True)

    return df


# -----------------------------
# Prepare Latest Features
# -----------------------------

def get_latest_features(df):

    features = [
        "return",
        "ma10",
        "ma50",
        "volatility",
        "rsi",
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "momentum",
        "volatility20",
        "macd"
    ]

    latest_row = df[features].iloc[-1:]

    return latest_row


# -----------------------------
# Prediction
# -----------------------------

def make_prediction(model, X):

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    print("\nPrediction result")

    print("Predicted direction:", "UP" if prediction == 1 else "DOWN")

    print("Probability of UP:", round(probability, 4))


# -----------------------------
# Main Pipeline
# -----------------------------

def main():

    model = load_model()

    df = download_latest_data()

    df = create_features(df)

    latest_features = get_latest_features(df)

    make_prediction(model, latest_features)


if __name__ == "__main__":
    main()