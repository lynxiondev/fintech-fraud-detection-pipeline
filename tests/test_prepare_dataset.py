import pandas as pd
import pytest

from src.ml.prepare_dataset import prepare_dataset


def test_prepare_dataset_splits_chronologically():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "is_fraud": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        }
    )

    train, test = prepare_dataset(df)

    assert len(train) == 8
    assert len(test) == 2

    assert train["timestamp"].max() < test["timestamp"].min()


def test_prepare_dataset_preserves_target():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "is_fraud": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        }
    )

    train, test = prepare_dataset(df)

    assert "is_fraud" in train.columns
    assert "is_fraud" in test.columns


def test_prepare_dataset_rejects_empty_dataset():
    df = pd.DataFrame(
        columns=["timestamp", "is_fraud"]
    )

    with pytest.raises(ValueError, match="Feature dataset is empty"):
        prepare_dataset(df)


def test_prepare_dataset_rejects_missing_required_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_dataset(df)