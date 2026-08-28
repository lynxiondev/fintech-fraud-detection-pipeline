"""
Data quality checks for the transaction dataset.
"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "customer_id",
    "amount",
    "currency",
    "merchant_category",
    "payment_method",
    "country",
    "device_type",
    "is_international",
    "is_fraud",
}


def validate_transactions(path: str) -> None:
    """Validate the transaction dataset."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    if df["amount"].isna().any():
        raise ValueError(
            "Missing transaction amounts detected."
        )

    if (df["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError(
            "is_fraud must contain only 0 or 1."
        )

    print("Data quality checks passed.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    validate_transactions(
        "data/raw/transactions.csv"
    )
