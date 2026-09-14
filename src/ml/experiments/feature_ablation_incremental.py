from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/processed/transactions_features.csv")
TARGET = "is_fraud"


def build_model(numeric_features, categorical_features):
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
            ("categorical", categorical_pipeline, categorical_features),
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
        "A_payment_method": {
            "numeric": [],
            "categorical": [
                "payment_method",
            ],
        },
        "B_plus_international": {
            "numeric": [
                "is_international",
            ],
            "categorical": [
                "payment_method",
            ],
        },
        "C_plus_amount": {
            "numeric": [
                "is_international",
                "amount",
            ],
            "categorical": [
                "payment_method",
            ],
        },
        "D_plus_customer_behavior": {
            "numeric": [
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
            "categorical": [
                "payment_method",
            ],
        },
        "E_plus_velocity": {
            "numeric": [
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
            "categorical": [
                "payment_method",
            ],
        },
        "F_plus_interaction": {
            "numeric": [
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
            "categorical": [
                "payment_method",
            ],
        },
    }

    results = []

    for name, features in experiments.items():
        model = build_model(
            features["numeric"],
            features["categorical"],
        )

        feature_columns = features["numeric"] + features["categorical"]

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)

        results.append(
            {
                "experiment": name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nIncremental feature ablation:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
