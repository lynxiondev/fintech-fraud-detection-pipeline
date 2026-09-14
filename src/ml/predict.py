"""
Generate fraud probabilities using a trained model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


MODEL_PATH = Path("models/fraud_model.joblib")
TEST_PATH = Path("data/processed/test.csv")
PREDICTIONS_PATH = Path("data/processed/predictions.csv")


def load_model(
    path: Path,
) -> Pipeline:
    """Load a trained model from disk."""

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    pipeline = joblib.load(path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "Loaded artifact is not a sklearn Pipeline."
        )

    return pipeline


def predict_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> pd.Series:
    """Generate fraud probabilities for a dataset."""

    probabilities = pipeline.predict_proba(X)[:, 1]

    return pd.Series(
        probabilities,
        index=X.index,
        name="fraud_probability",
    )


def generate_predictions(
    test: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Generate a prediction dataset from test transactions."""

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    probabilities = predict_probabilities(
        pipeline,
        test[features],
    )

    predictions = test[
        [
            "transaction_id",
            "timestamp",
            "is_fraud",
        ]
    ].copy()

    predictions["fraud_probability"] = probabilities

    return predictions


def main() -> None:
    """Generate and save fraud predictions."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    test = pd.read_csv(TEST_PATH)

    pipeline = load_model(
        MODEL_PATH,
    )

    predictions = generate_predictions(
        test,
        pipeline,
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("Prediction generation completed.")
    print()
    print(f"Predictions: {len(predictions):,}")
    print(f"Output: {PREDICTIONS_PATH}")
    print()
    print(
        f"Probability range: "
        f"{predictions['fraud_probability'].min():.4f}"
        f" → "
        f"{predictions['fraud_probability'].max():.4f}"
    )


if __name__ == "__main__":
    main()