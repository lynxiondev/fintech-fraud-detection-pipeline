"""
Baseline fraud detection model.

Uses a temporal train/test split and a preprocessing pipeline
to avoid data leakage during model training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

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


def main() -> None:
    """Train and evaluate the baseline model."""

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

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    pipeline = build_pipeline()

    print("Training baseline Logistic Regression...")
    print()

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    print("BASELINE RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC:  "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = (
        pipeline
        .named_steps["model"]
        .coef_[0]
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
            }
        )
        .sort_values(
            "coefficient",
            ascending=False,
        )
    )

    print()
    print("TOP FEATURES ASSOCIATED WITH FRAUD")
    print("=" * 60)
    print(
        feature_importance
        .head(15)
        .to_string(index=False)
    )

    print()
    print(
        "TOP FEATURES ASSOCIATED WITH "
        "LEGITIMATE TRANSACTIONS"
    )
    print("=" * 60)
    print(
        feature_importance
        .tail(15)
        .sort_values("coefficient")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
