"""
Operational evaluation for fraud detection.

Evaluates model performance across different probability thresholds
and review capacities using a chronological test set.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
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
    """Load processed data and perform chronological train/test split."""

    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


def build_model():
    """Build the logistic regression model."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def evaluate_thresholds(y_true, probabilities):
    """Evaluate classification performance across probability thresholds."""

    results = []

    for threshold in [i / 100 for i in range(10, 91, 5)]:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "alerts": int(predictions.sum()),
                "alert_rate": predictions.mean(),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "false_positives": int(
                    ((predictions == 1) & (y_true == 0)).sum()
                ),
                "false_negatives": int(
                    ((predictions == 0) & (y_true == 1)).sum()
                ),
            }
        )

    return pd.DataFrame(results)


def evaluate_capacity(y_true, probabilities):
    """Evaluate how many frauds are captured at fixed review capacities."""

    results = []

    for capacity in [0.05, 0.10, 0.15, 0.20, 0.25]:
        n_alerts = int(len(probabilities) * capacity)

        ranked_indices = probabilities.argsort()[::-1]
        selected_indices = ranked_indices[:n_alerts]

        captured_frauds = int(y_true.iloc[selected_indices].sum())

        precision = (
            captured_frauds / n_alerts
            if n_alerts > 0
            else 0
        )

        recall = captured_frauds / y_true.sum()

        baseline_fraud_rate = y_true.mean()
        lift = precision / baseline_fraud_rate 

        results.append(
            {
                "review_capacity": capacity,
                "alerts_reviewed": n_alerts,
                "frauds_captured": captured_frauds,
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )

    return pd.DataFrame(results)


def main():
    train, test = load_data()

    X_train = train[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train[TARGET]

    X_test = test[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test[TARGET]

    model = build_model()

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nModel ranking performance:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    threshold_results = evaluate_thresholds(
        y_test.reset_index(drop=True),
        probabilities,
    )

    print("\nThreshold evaluation:")
    print(threshold_results.to_string(index=False))

    capacity_results = evaluate_capacity(
        y_test.reset_index(drop=True),
        probabilities,
    )

    print("\nCapacity evaluation:")
    print(capacity_results.to_string(index=False))


if __name__ == "__main__":
    main() 