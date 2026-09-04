import pandas as pd

from src.utils.data_quality import validate_transactions


def create_valid_dataset():
    """Create a minimal valid transaction dataset for testing."""

    return pd.DataFrame(
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


def save_dataset(tmp_path, df):
    """Save a test dataset to a temporary CSV file."""

    path = tmp_path / "transactions.csv"
    df.to_csv(path, index=False)

    return path


def test_valid_labeled_dataset_passes(tmp_path):
    """A valid labeled dataset should pass validation."""

    df = create_valid_dataset()
    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["warnings"] == []


def test_missing_transaction_id_fails(tmp_path):
    """Missing transaction IDs should fail validation."""

    df = create_valid_dataset()
    df.loc[0, "transaction_id"] = None

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "Missing transaction IDs" in result["errors"][0]


def test_duplicate_transaction_id_fails(tmp_path):
    """Duplicate transaction IDs should fail validation."""

    df = create_valid_dataset()
    df.loc[1, "transaction_id"] = "tx_001"

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "Duplicate transaction IDs" in result["errors"][0]


def test_invalid_timestamp_fails(tmp_path):
    """Invalid timestamps should fail validation."""

    df = create_valid_dataset()
    df.loc[0, "timestamp"] = "not-a-date"

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "Invalid timestamps" in result["errors"][0]


def test_non_positive_amount_fails(tmp_path):
    """Non-positive transaction amounts should fail validation."""

    df = create_valid_dataset()
    df.loc[0, "amount"] = 0

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "Transaction amounts must be positive" in result["errors"][0]


def test_invalid_is_international_fails(tmp_path):
    """Invalid is_international values should fail validation."""

    df = create_valid_dataset()
    df.loc[0, "is_international"] = 2

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "is_international" in result["errors"][0]


def test_missing_target_fails_for_labeled_dataset(tmp_path):
    """A labeled dataset must contain the fraud target."""

    df = create_valid_dataset()
    df = df.drop(columns=["is_fraud"])

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "Missing target column" in result["errors"][0]


def test_invalid_target_fails_for_labeled_dataset(tmp_path):
    """The fraud target must contain only 0 or 1."""

    df = create_valid_dataset()
    df.loc[0, "is_fraud"] = 2

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "FAIL"
    assert "is_fraud must contain only 0 or 1" in result["errors"][0]


def test_unlabeled_dataset_does_not_require_target(tmp_path):
    """Scoring data should not require the fraud target."""

    df = create_valid_dataset()
    df = df.drop(columns=["is_fraud"])

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=False)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["warnings"] == []

def test_unexpected_payment_method_returns_warning(tmp_path):
    df = create_valid_dataset()
    df.loc[0, "payment_method"] = "crypto_wallet"

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "WARNING"
    assert result["errors"] == []
    assert "Unexpected payment_method values: ['crypto_wallet']" in result["warnings"]


def test_unexpected_category_returns_warning(tmp_path):
    df = create_valid_dataset()
    df.loc[0, "merchant_category"] = "cryptocurrency"

    path = save_dataset(tmp_path, df)

    result = validate_transactions(path, labeled=True)

    assert result["status"] == "WARNING"
    assert result["errors"] == []
    assert "Unexpected merchant_category values: ['cryptocurrency']" in result["warnings"]
