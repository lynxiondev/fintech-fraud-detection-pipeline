"""
Run the transaction data processing pipeline.
"""

from pathlib import Path

import pandas as pd

from src.features.build_features import build_features
from src.utils.data_quality import validate_transactions


INPUT_PATH = Path("data/raw/transactions.csv")
OUTPUT_PATH = Path("data/processed/transactions_features.csv")


def run_pipeline(
    input_path: Path = INPUT_PATH, 
    output_path: Path = OUTPUT_PATH,  
) -> None: 
    
    """Run data quality validation and feature engineering."""

    print("Starting transaction pipeline...")

    # ---------------------------------------------------------
    # 1. Data quality validation
    # ---------------------------------------------------------

    validation = validate_transactions(
        input_path,
        labeled=True,
    )

    print(f"Data quality status: {validation['status']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    if validation["status"] == "FAIL":
        print("Pipeline stopped: data quality validation failed.")

        for error in validation["errors"]:
            print(f"  - {error}")

        raise ValueError("Data quality validation failed.")

    # ---------------------------------------------------------
    # 2. Load raw data
    # ---------------------------------------------------------

    df = pd.read_csv(input_path)

    print(f"Loaded {len(df):,} transactions.")

    # ---------------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------------

    df_features = build_features(df)

    print(
        f"Feature engineering completed. "
        f"Rows: {len(df_features):,} "
        f"Columns: {len(df_features.columns)}"
    )

    # ---------------------------------------------------------
    # 4. Save processed dataset
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_features.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved processed dataset to: {OUTPUT_PATH}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline() 