import pandas as pd
import yfinance as yf
import ta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"


def download_data():

    print("Downloading fresh market data...")

    df = yf.download("SPY", period="10y", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


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

    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    df.dropna(inplace=True)

    return df


def train_model(df):

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

    X = df[features]
    y = df["target"]

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nModel evaluation")

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))

    return model


def save_model(model):

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel updated and saved.")


def main():

    df = download_data()

    df = create_features(df)

    model = train_model(df)

    save_model(model)


if __name__ == "__main__":
    main()