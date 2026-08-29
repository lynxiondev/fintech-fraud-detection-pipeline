"""
Prepare the feature dataset for machine learning.

The split is chronological to simulate a real production scenario:
past transactions are used for training and future transactions
are reserved for evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/transactions_features.csv")

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


TARGET = "is_fraud"


EXCLUDED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    TARGET,
]


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train/test datasets."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


def main() -> None:
    """Prepare train and test datasets."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    train, test = prepare_dataset(df)

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("Temporal dataset split completed.")
    print()
    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")
    print()
    print(
        f"Train period: "
        f"{train['timestamp'].min()} → {train['timestamp'].max()}"
    )
    print(
        f"Test period:  "
        f"{test['timestamp'].min()} → {test['timestamp'].max()}"
    )
    print()
    print(
        f"Train fraud rate: "
        f"{train[TARGET].mean():.2%}"
    )
    print(
        f"Test fraud rate:  "
        f"{test[TARGET].mean():.2%}"
    )


if __name__ == "__main__":
    main()
