"""
Data quality checks for transaction datasets.
"""

from pathlib import Path

import pandas as pd


# Columns required for transaction ingestion and ML scoring.
REQUIRED_INPUT_COLUMNS = {
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
}

# Label required only for training/evaluation datasets.
TARGET_COLUMN = "is_fraud"

# Expected categorical values for the current transaction domain.
EXPECTED_VALUES = {
    "currency": {"USD"},
    "merchant_category": {
        "electronics",
        "entertainment",
        "groceries",
        "online_services",
        "restaurants",
        "transport",
        "travel",
        "utilities",
    },
    "payment_method": {
        "bank_transfer",
        "card",
        "digital_wallet",
        "pix",
    },
    "country": {
        "AR",
        "BR",
        "CL",
        "CO",
        "MX",
        "PT",
        "US",
    },
    "device_type": {
        "desktop",
        "mobile",
        "tablet",
    },
}


def validate_transactions(path: str, labeled: bool = False) -> dict:
    """
    Validate a transaction dataset.

    Parameters
    ----------
    path:
        Path to the transaction CSV file.

    labeled:
        Whether the dataset contains the fraud label.
        Use True for training/evaluation datasets.

    Returns
    -------
    dict
        Validation result containing status, errors, and warnings.
    """

    file_path = Path(path)

    if not file_path.exists():
        return {
            "status": "FAIL",
            "errors": [f"Dataset not found: {file_path}"],
            "warnings": [],
        }

    df = pd.read_csv(file_path)

    errors = []
    warnings = []

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    missing_columns = REQUIRED_INPUT_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if labeled and TARGET_COLUMN not in df.columns:
        errors.append(
            f"Missing target column: {TARGET_COLUMN}"
        )

    # Stop here if required columns are missing.
    #
    # The remaining checks depend on those columns existing.
    if errors:
        return {
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Transaction ID validation
    # ------------------------------------------------------------------

    if df["transaction_id"].isna().any():
        errors.append(
            "Missing transaction IDs detected."
        )

    if df["transaction_id"].duplicated().any():
        errors.append(
            "Duplicate transaction IDs detected."
        )

    # ------------------------------------------------------------------
    # Timestamp validation
    # ------------------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    if timestamps.isna().any():
        errors.append(
            "Invalid timestamps detected."
        )

    # ------------------------------------------------------------------
    # Amount validation
    # ------------------------------------------------------------------

    if df["amount"].isna().any():
        errors.append(
            "Missing transaction amounts detected."
        )

    if (df["amount"] <= 0).any():
        errors.append(
            "Transaction amounts must be positive."
        )

    # ------------------------------------------------------------------
    # Domain validation
    # ------------------------------------------------------------------

    if not df["is_international"].isin([0, 1]).all():
        errors.append(
            "is_international must contain only 0 or 1."
        )

    # Expected categorical values
    for column, expected_values in EXPECTED_VALUES.items():
        observed_values = set(df[column].dropna().unique())
        unexpected_values = observed_values - expected_values

        if unexpected_values:
            warnings.append(
            f"Unexpected {column} values: "
            f"{sorted(unexpected_values)}"
            )    

    # ------------------------------------------------------------------
    # Training / evaluation validation
    # ------------------------------------------------------------------

    if labeled:
        if df[TARGET_COLUMN].isna().any():
            errors.append(
                "Missing fraud labels detected."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            errors.append(
                "is_fraud must contain only 0 or 1."
            )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    result = validate_transactions(
        "data/raw/transactions.csv",
        labeled=True,
    )

    print(f"Status: {result['status']}")

    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")

    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

