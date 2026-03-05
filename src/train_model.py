import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ----------------------------------
# Paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spy_features.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"


# ----------------------------------
# Load Dataset
# ----------------------------------

def load_data():

    print("Loading feature dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


# ----------------------------------
# Prepare Features
# ----------------------------------

def prepare_data(df):

    print("Preparing features and target...")

    feature_cols = [
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

    X = df[feature_cols]
    y = df["target"]

    return X, y


# ----------------------------------
# Time Series Train/Test Split
# ----------------------------------

def time_split(X, y):

    print("Splitting dataset (time-series split)...")

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ----------------------------------
# Train Model
# ----------------------------------

def train_model(X_train, y_train):

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ----------------------------------
# Evaluate Model
# ----------------------------------

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating model...")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    # Baseline (always predict majority class)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"Baseline Accuracy: {baseline:.4f}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")


# ----------------------------------
# Save Model
# ----------------------------------

def save_model(model):

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to {MODEL_PATH}")


# ----------------------------------
# Main Pipeline
# ----------------------------------

def main():

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = time_split(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()