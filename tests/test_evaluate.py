import pandas as pd
import pytest

from src.ml.evaluate import (
    evaluate_at_alert_rate,
    evaluate_at_threshold,
    evaluate_predictions,
)


@pytest.fixture
def predictions():
    y_true = pd.Series(
        [0, 1, 0, 1, 0],
        index=[10, 20, 30, 40, 50],
    )

    probabilities = pd.Series(
        [0.10, 0.90, 0.20, 0.80, 0.30],
        index=[10, 20, 30, 40, 50],
    )

    return y_true, probabilities


def test_evaluate_predictions_returns_ranking_metrics(predictions):
    y_true, probabilities = predictions

    results = evaluate_predictions(
        y_true,
        probabilities,
    )

    assert "roc_auc" in results
    assert "pr_auc" in results

    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1


def test_evaluate_at_alert_rate(predictions):
    y_true, probabilities = predictions

    results = evaluate_at_alert_rate(
        y_true,
        probabilities,
        alert_rate=0.40,
    )

    assert results["alerts"] == 2
    assert results["frauds_captured"] == 2
    assert results["recall"] == 1.0
    assert results["precision"] == 1.0


def test_evaluate_at_threshold(predictions):
    y_true, probabilities = predictions

    results = evaluate_at_threshold(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert results["threshold"] == 0.50
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0


def test_alert_rate_must_be_valid(predictions):
    y_true, probabilities = predictions

    with pytest.raises(ValueError):
        evaluate_at_alert_rate(
            y_true,
            probabilities,
            alert_rate=0,
        )


def test_threshold_must_be_valid(predictions):
    y_true, probabilities = predictions

    with pytest.raises(ValueError):
        evaluate_at_threshold(
            y_true,
            probabilities,
            threshold=1.5,
        )