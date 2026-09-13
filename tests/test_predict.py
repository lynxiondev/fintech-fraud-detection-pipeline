import joblib
import pandas as pd
import pytest


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.ml.predict import (
    load_model,
    predict_probabilities,
)
from src.ml.train import save_model

class DummyModel:
    def predict_proba(self, X):
        return pd.DataFrame(
            {
                "legitimate": [0.9, 0.2, 0.7],
                "fraud": [0.1, 0.8, 0.3],
            }
        ).to_numpy()


def test_predict_probabilities_returns_fraud_probability():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    probabilities = predict_probabilities(
        DummyModel(),
        X,
    )

    assert probabilities.name == "fraud_probability"
    assert probabilities.tolist() == [0.1, 0.8, 0.3]


def test_predict_probabilities_preserves_index():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        },
        index=[10, 20, 30],
    )

    probabilities = predict_probabilities(
        DummyModel(),
        X,
    )

    assert probabilities.index.tolist() == [10, 20, 30]


def test_load_model(tmp_path):
    pipeline = Pipeline([])

    model_path = tmp_path / "model.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    loaded_model = load_model(
        model_path,
    )

    assert isinstance(
        loaded_model,
        Pipeline,
    )


def test_load_model_missing_file(tmp_path):
    model_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(
            model_path,
        )


def test_load_model_invalid_artifact(tmp_path):
    artifact_path = tmp_path / "invalid.joblib"

    joblib.dump(
        {"not": "a model"},
        artifact_path,
    )

    with pytest.raises(TypeError):
        load_model(
            artifact_path,
        )


def test_model_persistence_round_trip(tmp_path):
    X = pd.DataFrame(
        {
            "feature": [0, 1, 0, 1],
        }
    )

    y = pd.Series(
        [0, 1, 0, 1],
    )

    pipeline = Pipeline(
        [
            (
                "model",
                LogisticRegression(
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        X,
        y,
    )

    original_probabilities = predict_probabilities(
        pipeline,
        X,
    )

    model_path = tmp_path / "model.joblib"

    save_model(
        pipeline,
        model_path,
    )

    loaded_model = load_model(
        model_path,
    )

    loaded_probabilities = predict_probabilities(
        loaded_model,
        X,
    )

    pd.testing.assert_series_equal(
        original_probabilities,
        loaded_probabilities,
    )

