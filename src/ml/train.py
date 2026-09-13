"""
Baseline fraud detection model.

Uses a temporal train/test split and a preprocessing pipeline
to avoid data leakage during model training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/fraud_model.joblib")

TARGET = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "is_international",
    "customer_transaction_count",
    "customer_avg_amount",
    "amount_vs_customer_avg",
    "customer_unique_countries",
    "customer_unique_devices",
    "is_new_country",
    "is_new_device",
    "transactions_last_1h",
    "transactions_last_24h",
    "amount_last_1h",
    "amount_last_24h",
    "international_digital_wallet",
]

CATEGORICAL_FEATURES = [
    "currency",
    "merchant_category",
    "payment_method",
    "country",
    "device_type",
]


def build_pipeline() -> Pipeline:
    """Build preprocessing and logistic regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
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

def validate_training_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate datasets contain the required ML columns."""

    required_columns = {
        TARGET,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }

    for name, df in [("train", train), ("test", test)]:
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if df.empty:
            raise ValueError(
                f"{name} dataset is empty."
            )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train the fraud detection model."""

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline

def save_model(
    pipeline: Pipeline,
    path: Path,
) -> None:
    """Save the trained model to disk."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        path,
    )

def main() -> None:
    """Train the fraud detection model."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    validate_training_data(
        train,
        test,
    )

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train[features]
    y_train = train[TARGET]

    print("Training Logistic Regression...")
    print()

    pipeline = train_model(
    X_train,
    y_train,
    )

    save_model(
    pipeline,
    MODEL_PATH,
    )

    print("Model training completed.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()