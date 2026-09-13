"""
Generate fraud probabilities using a trained model.
"""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

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