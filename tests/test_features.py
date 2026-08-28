"""
Tests for temporal feature engineering.
"""

import pandas as pd

from src.features.build_features import build_features


def test_historical_features_do_not_use_current_transaction():
    """Historical features must exclude the current transaction."""

    df = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-03 10:00:00",
                ]
            ),
            "customer_id": [1, 1, 1],
            "amount": [100.0, 200.0, 300.0],
            "country": ["BR", "BR", "AR"],
            "device_type": ["mobile", "mobile", "desktop"],
            "currency": ["USD", "USD", "USD"],
            "merchant_category": [
                "groceries",
                "electronics",
                "travel",
            ],
            "payment_method": [
                "card",
                "card",
                "pix",
            ],
            "is_international": [0, 0, 1],
            "is_fraud": [0, 0, 1],
        }
    )

    result = build_features(df)

    # First transaction has no history.
    assert result.loc[0, "customer_transaction_count"] == 0
    assert pd.isna(result.loc[0, "customer_avg_amount"])

    # Second transaction can only see transaction 1.
    assert result.loc[1, "customer_transaction_count"] == 1
    assert result.loc[1, "customer_avg_amount"] == 100.0

    # Third transaction can see transactions 1 and 2.
    assert result.loc[2, "customer_transaction_count"] == 2
    assert result.loc[2, "customer_avg_amount"] == 150.0


def test_new_country_uses_only_previous_history():
    """New-country detection must use historical observations only."""

    df = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "customer_id": [1, 1, 1],
            "amount": [10.0, 20.0, 30.0],
            "country": ["BR", "BR", "AR"],
            "device_type": ["mobile", "mobile", "desktop"],
            "currency": ["USD", "USD", "USD"],
            "merchant_category": [
                "groceries",
                "groceries",
                "travel",
            ],
            "payment_method": [
                "card",
                "card",
                "pix",
            ],
            "is_international": [0, 0, 1],
            "is_fraud": [0, 0, 1],
        }
    )

    result = build_features(df)

    assert result.loc[0, "is_new_country"] == 1
    assert result.loc[1, "is_new_country"] == 0
    assert result.loc[2, "is_new_country"] == 1


def test_new_device_uses_only_previous_history():
    """New-device detection must use historical observations only."""

    df = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "customer_id": [1, 1, 1],
            "amount": [10.0, 20.0, 30.0],
            "country": ["BR", "BR", "BR"],
            "device_type": ["mobile", "mobile", "desktop"],
            "currency": ["USD", "USD", "USD"],
            "merchant_category": [
                "groceries",
                "groceries",
                "travel",
            ],
            "payment_method": [
                "card",
                "card",
                "pix",
            ],
            "is_international": [0, 0, 0],
            "is_fraud": [0, 0, 1],
        }
    )

    result = build_features(df)

    assert result.loc[0, "is_new_device"] == 1
    assert result.loc[1, "is_new_device"] == 0
    assert result.loc[2, "is_new_device"] == 1
