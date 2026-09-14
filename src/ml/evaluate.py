"""
Evaluate fraud detection model predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PREDICTIONS_PATH = Path(
    "data/processed/predictions.csv"
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

    frauds_captured = float(
        alerts.sum()
    )

    total_frauds = float(
        y_true.sum()
    )

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


def main() -> None:
    """Evaluate the generated prediction dataset."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions dataset not found: "
            f"{PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    required_columns = {
        "is_fraud",
        "fraud_probability",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    y_true = predictions["is_fraud"]
    probabilities = predictions[
        "fraud_probability"
    ]

    ranking_metrics = evaluate_predictions(
        y_true,
        probabilities,
    )

    alert_10 = evaluate_at_alert_rate(
        y_true,
        probabilities,
        alert_rate=0.10,
    )

    threshold_50 = evaluate_at_threshold(
        y_true,
        probabilities,
        threshold=0.50,
    )

    print("Fraud model evaluation")
    print()

    print(
        f"ROC-AUC: "
        f"{ranking_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:  "
        f"{ranking_metrics['pr_auc']:.4f}"
    )

    print()

    print("Operational evaluation @ 10% alert rate")
    print(
        f"Alerts:          "
        f"{int(alert_10['alerts'])}"
    )
    print(
        f"Frauds captured: "
        f"{int(alert_10['frauds_captured'])}"
    )
    print(
        f"Recall:          "
        f"{alert_10['recall']:.4f}"
    )
    print(
        f"Precision:       "
        f"{alert_10['precision']:.4f}"
    )

    print()

    print("Threshold evaluation @ 0.50")
    print(
        f"Recall:          "
        f"{threshold_50['recall']:.4f}"
    )
    print(
        f"Precision:       "
        f"{threshold_50['precision']:.4f}"
    )


if __name__ == "__main__":
    main() 