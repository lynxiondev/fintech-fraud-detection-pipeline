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


def build_model(numeric_features):
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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
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


def recall_at_alert_rate(y_true, probabilities, alert_rate=0.10):
    n_alerts = int(len(probabilities) * alert_rate)

    ranked_indices = (
        pd.Series(probabilities)
        .sort_values(ascending=False)
        .head(n_alerts)
        .index
    )

    predictions = pd.Series(0, index=range(len(probabilities)))
    predictions.loc[ranked_indices] = 1

    recall = recall_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)

    frauds_captured = int(
        ((predictions == 1) & (pd.Series(y_true).reset_index(drop=True) == 1)).sum()
    )

    return recall, precision, frauds_captured, n_alerts


def main():
    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    y_train = train[TARGET]
    y_test = test[TARGET]

    experiments = {
        "A_baseline_categorical": [],
        "B_plus_international": [
            "is_international",
        ],
        "C_plus_amount": [
            "is_international",
            "amount",
        ],
        "D_plus_customer_behavior": [
            "is_international",
            "amount",
            "customer_transaction_count",
            "customer_avg_amount",
            "amount_vs_customer_avg",
            "customer_unique_countries",
            "customer_unique_devices",
            "is_new_country",
            "is_new_device",
        ],
        "E_plus_velocity": [
            "is_international",
            "amount",
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
        ],
        "F_plus_interaction": [
            "is_international",
            "amount",
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
        ],
    }

    results = []

    for name, numeric_features in experiments.items():
        model = build_model(numeric_features)

        feature_columns = numeric_features + CATEGORICAL_FEATURES

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)

        recall_10, precision_10, captured_10, alerts_10 = (
            recall_at_alert_rate(
                y_test,
                probabilities,
                alert_rate=0.10,
            )
        )

        results.append(
            {
                "experiment": name,
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "recall_at_10pct": round(recall_10, 4),
                "precision_at_10pct": round(precision_10, 4),
                "frauds_captured": captured_10,
                "alerts_reviewed": alerts_10,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nFull incremental feature ablation:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
