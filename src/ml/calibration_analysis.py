from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator, calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/processed/transactions_features.csv")
TARGET = "is_fraud"

CATEGORICAL_FEATURES = [
    "currency",
    "merchant_category",
    "payment_method",
    "country",
    "device_type",
]

NUMERIC_FEATURES = [
    "is_international",
    "amount",
    "customer_transaction_count",
    "customer_avg_amount",
    "amount_vs_customer_avg",
    "customer_unique_countries",
    "customer_unique_devices",
    "is_new_country",
    "is_new_device",
]


def load_data():
    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_end = int(len(df) * 0.64)
    calibration_end = int(len(df) * 0.80)

    train = df.iloc[:train_end].copy()
    calibration = df.iloc[train_end:calibration_end].copy()
    test = df.iloc[calibration_end:].copy()

    return train, calibration, test


def build_model():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    train, calibration, test = load_data()

    X_train = train[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train[TARGET]

    X_calibration = calibration[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_calibration = calibration[TARGET]

    X_test = test[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test[TARGET]

    # ---------------------------------------------------------
    # 1. Train the original model
    # ---------------------------------------------------------

    model = build_model()
    model.fit(X_train, y_train)

    uncalibrated_probabilities = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 2. Fit sigmoid calibration using the calibration set
    # ---------------------------------------------------------
    calibration_model = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
    )

    calibration_model.fit(X_calibration, y_calibration)

    calibrated_probabilities = calibration_model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 3. Evaluate both models
    # ---------------------------------------------------------

    uncalibrated_roc = roc_auc_score(
        y_test,
        uncalibrated_probabilities,
    )

    uncalibrated_pr = average_precision_score(
        y_test,
        uncalibrated_probabilities,
    )

    uncalibrated_brier = brier_score_loss(
        y_test,
        uncalibrated_probabilities,
    )

    calibrated_roc = roc_auc_score(
        y_test,
        calibrated_probabilities,
    )

    calibrated_pr = average_precision_score(
        y_test,
        calibrated_probabilities,
    )

    calibrated_brier = brier_score_loss(
        y_test,
        calibrated_probabilities,
    )

    print("Calibration comparison")
    print("======================")

    print()
    print("Uncalibrated")
    print("------------")
    print(f"ROC-AUC:     {uncalibrated_roc:.4f}")
    print(f"PR-AUC:      {uncalibrated_pr:.4f}")
    print(f"Brier Score: {uncalibrated_brier:.4f}")

    print()
    print("Calibrated")
    print("----------")
    print(f"ROC-AUC:     {calibrated_roc:.4f}")
    print(f"PR-AUC:      {calibrated_pr:.4f}")
    print(f"Brier Score: {calibrated_brier:.4f}")


if __name__ == "__main__":
    main()
        