import pandas as pd
import pytest

from src.pipeline.run_pipeline import run_pipeline


def test_pipeline_stops_when_data_quality_fails(tmp_path):
    """The pipeline should stop when input data fails validation."""

    df = pd.DataFrame(
        {
            "transaction_id": ["tx_001"],
            "timestamp": ["2026-01-01 10:00:00"],
            "customer_id": ["customer_001"],
            "amount": [-100.0],
            "currency": ["USD"],
            "merchant_category": ["groceries"],
            "payment_method": ["card"],
            "country": ["AR"],
            "device_type": ["mobile"],
            "is_international": [0],
            "is_fraud": [0],
        }
    )

    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "transactions_features.csv"

    df.to_csv(input_path, index=False)

    with pytest.raises(ValueError):
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
        )

    assert not output_path.exists()  

def test_pipeline_generates_features_for_valid_data(tmp_path):
    """The pipeline should generate a feature dataset for valid input."""

    df = pd.DataFrame(
        {
            "transaction_id": ["tx_001", "tx_002"],
            "timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
            ],
            "customer_id": ["customer_001", "customer_002"],
            "amount": [100.0, 250.0],
            "currency": ["USD", "USD"],
            "merchant_category": ["groceries", "electronics"],
            "payment_method": ["card", "digital_wallet"],
            "country": ["AR", "BR"],
            "device_type": ["mobile", "desktop"],
            "is_international": [0, 1],
            "is_fraud": [0, 1],
        }
    )

    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "transactions_features.csv"

    df.to_csv(input_path, index=False)

    run_pipeline(
        input_path=input_path,
        output_path=output_path,
    )

    assert output_path.exists()

    result = pd.read_csv(output_path)

    assert len(result) == 2
    assert "customer_transaction_count" in result.columns
    assert "international_digital_wallet" in result.columns 