from fastapi import FastAPI
import joblib
import pandas as pd
import yfinance as yf
import ta
from pathlib import Path
from datetime import datetime
import subprocess

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
LOG_PATH = BASE_DIR / "logs" / "prediction_log.csv"

model = joblib.load(MODEL_PATH)


def get_latest_features():

    df = yf.download("SPY", period="3y", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

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
        "macd",
    ]

    latest_price = float(df["Close"].iloc[-1])

    return df[features].iloc[-1:], latest_price


def log_prediction(price, prediction, probability):

    log_entry = pd.DataFrame(
        [[datetime.utcnow(), price, prediction, probability]],
        columns=["timestamp", "price", "prediction", "probability"],
    )

    if LOG_PATH.exists():
        log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)


@app.get("/")
def root():
    return {"message": "Stock Prediction API is running"}


@app.get("/predict")
def predict():

    X, price = get_latest_features()

    prediction_raw = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    prediction = "UP" if prediction_raw == 1 else "DOWN"

    log_prediction(price, prediction, float(probability))

    return {
        "prediction": prediction,
        "probability_up": round(float(probability), 4),
        "current_price": round(price, 2),
    }


@app.post("/retrain")
def retrain():

    try:

        subprocess.run(["python", "src/retrain_pipeline.py"], check=True)

        global model
        model = joblib.load(MODEL_PATH)

        return {"status": "Model retrained successfully"}

    except Exception as e:

        return {"status": "Retraining failed", "error": str(e)}


@app.get("/logs")
def get_logs():

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        return df.to_dict(orient="records")

    return []