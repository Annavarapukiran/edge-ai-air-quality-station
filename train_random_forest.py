"""
train_random_forest.py

Trains an on-chip Random Forest Regressor that maps raw gas sensor voltages
(MQ135, MQ7, FC22, SGP41 VOC/NOx) directly to PM2.5 particulate density.

This model is intentionally lightweight (low tree depth, modest tree count)
so that inference complexity O(D * N) stays cheap enough to run live on a
Raspberry Pi 4 CPU without dedicated graphics hardware.

Usage:
    python3 train_random_forest.py --data clean_dataset.csv --out rf_model.joblib
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = ["MQ135", "MQ7", "FC22", "SGP41_VOC", "SGP41_NOX"]
TARGET_COLUMN = "PM2_5"


def train(data_path: str, model_out: str, n_estimators: int = 100, max_depth: int = 8) -> None:
    df = pd.read_csv(data_path)

    # Use a fully named DataFrame (not a raw NumPy array) for features so
    # scikit-learn does not raise feature-name-mismatch warnings during
    # live inference later on.
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if len(df) < 4:
        print(f"[WARN] Only {len(df)} clean rows available — training on a minimal "
              "reference set. Model will adapt as more clean data accumulates.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print(f"[INFO] Random Forest trained on {len(X_train)} rows.")
    print(f"[INFO] MAE: {mae:.2f} ug/m3 | MSE: {mse:.2f}")

    joblib.dump(model, model_out)
    print(f"[INFO] Model saved to: {model_out}")


def predict(model_path: str, sensor_reading: dict) -> float:
    """Run a single live inference pass. sensor_reading keys must match FEATURE_COLUMNS."""
    model = joblib.load(model_path)
    X = pd.DataFrame([sensor_reading], columns=FEATURE_COLUMNS)
    return float(model.predict(X)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the edge Random Forest PM2.5 model.")
    parser.add_argument("--data", default="clean_dataset.csv", help="Path to cleaned training CSV")
    parser.add_argument("--out", default="rf_model.joblib", help="Path to save trained model")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    train(args.data, args.out, args.n_estimators, args.max_depth)
