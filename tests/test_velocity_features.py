import pandas as pd

from src.features.build_features import build_features


def test_velocity_features_do_not_include_current_transaction():
    """Velocity features must only use transactions before the current one."""

    data = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:30:00",
                "2026-01-01 11:30:00",
            ],
            "customer_id": [1, 1, 1],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["ARS", "ARS", "ARS"],
            "merchant_category": [
                "groceries",
                "groceries",
                "groceries",
            ],
            "payment_method": [
                "card",
                "card",
                "card",
            ],
            "country": ["AR", "AR", "AR"],
            "device_type": [
                "mobile",
                "mobile",
                "mobile",
            ],
            "is_international": [0, 0, 0],
            "is_fraud": [0, 0, 1],
        }
    )

    result = build_features(data)

    first = result.iloc[0]
    second = result.iloc[1]
    third = result.iloc[2]

    # First transaction has no history.
    assert first["transactions_last_1h"] == 0
    assert first["amount_last_1h"] == 0

    # Second transaction sees only the first transaction.
    assert second["transactions_last_1h"] == 1
    assert second["amount_last_1h"] == 10.0

    # Third transaction is 60 minutes after the second.
    # The 10:30 transaction is outside the 1-hour window.
    assert third["transactions_last_1h"] == 0
    assert third["amount_last_1h"] == 0

    # The 24-hour window still includes both previous transactions.
    assert third["transactions_last_24h"] == 2
    assert third["amount_last_24h"] == 30.0


def test_velocity_features_ignore_current_transaction_amount():
    """Current transaction amount must never appear in velocity totals."""

    data = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:30:00",
            ],
            "customer_id": [1, 1],
            "amount": [100.0, 1000.0],
            "currency": ["ARS", "ARS"],
            "merchant_category": [
                "groceries",
                "electronics",
            ],
            "payment_method": [
                "card",
                "digital_wallet",
            ],
            "country": ["AR", "AR"],
            "device_type": [
                "mobile",
                "tablet",
            ],
            "is_international": [0, 0],
            "is_fraud": [0, 1],
        }
    )

    result = build_features(data)

    second = result.iloc[1]

    # Only the previous $100 transaction is included.
    # The current $1,000 must NOT be included.
    assert second["transactions_last_1h"] == 1
    assert second["amount_last_1h"] == 100.0
