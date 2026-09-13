"""
Evaluate fraud detection model predictions.
"""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    """Calculate ranking metrics for fraud predictions."""

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }


def evaluate_at_alert_rate(
    y_true: pd.Series,
    probabilities: pd.Series,
    alert_rate: float,
) -> dict[str, float]:
    """Evaluate fraud detection at a fixed alert rate."""

    if not 0 < alert_rate <= 1:
        raise ValueError(
            "alert_rate must be greater than 0 and less than or equal to 1."
        )

    number_of_alerts = max(
        1,
        int(len(y_true) * alert_rate),
    )

    top_indices = (
        probabilities
        .nlargest(number_of_alerts)
        .index
    )

    alerts = y_true.loc[top_indices]

    frauds_captured = float(alerts.sum())
    total_frauds = float(y_true.sum())

    recall = (
        frauds_captured / total_frauds
        if total_frauds > 0
        else 0.0
    )

    precision = (
        frauds_captured / number_of_alerts
        if number_of_alerts > 0
        else 0.0
    )

    return {
        "alert_rate": alert_rate,
        "alerts": float(number_of_alerts),
        "frauds_captured": frauds_captured,
        "recall": recall,
        "precision": precision,
    }


def evaluate_at_threshold(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Evaluate predictions at a probability threshold."""

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
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
    }