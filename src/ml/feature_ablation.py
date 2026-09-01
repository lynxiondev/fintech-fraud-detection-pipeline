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
        "1_digital_wallet": {
            "numeric": [],
            "categorical": ["payment_method"],
        },
        "2_is_international": {
            "numeric": ["is_international"],
            "categorical": [],
        },
        "3_both": {
            "numeric": ["is_international"],
            "categorical": ["payment_method"],
        },
        "4_both_plus_interaction": {
            "numeric": ["is_international", "international_digital_wallet"],
            "categorical": ["payment_method"],
        },
    }

    results = []

    for name, features in experiments.items():
        model = build_model(
            features["numeric"],
            features["categorical"],
        )

        X_train = train[features["numeric"] + features["categorical"]]
        X_test = test[features["numeric"] + features["categorical"]]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        if name == "1_digital_wallet":
            debug = pd.DataFrame(
        {
            "payment_method": test["payment_method"].values,
            "is_fraud": y_test.values,
            "probability": probabilities,
        }
    )

    print("\nDigital wallet score distribution:")
    print(
        debug.groupby("payment_method")["probability"]
        .agg(["count", "mean", "min", "max"])
    )

    print("\nMean probability by fraud:")
    print(
        debug.groupby("is_fraud")["probability"]
        .agg(["count", "mean", "min", "max"])
    )

        

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

    print("\nFeature ablation results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
