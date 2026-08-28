"""
Exploratory data analysis for the synthetic transaction dataset.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/transactions.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("Columns:")
    for column in df.columns:
        print(f"  - {column}: {df[column].dtype}")

    print()
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isna().sum())

    print()
    print("=" * 60)
    print("DUPLICATES")
    print("=" * 60)
    print(f"Duplicate transaction IDs: {df['transaction_id'].duplicated().sum()}")

    print()
    print("=" * 60)
    print("FRAUD DISTRIBUTION")
    print("=" * 60)

    fraud_counts = df["is_fraud"].value_counts()
    fraud_rate = df["is_fraud"].mean()

    print(fraud_counts)
    print(f"\nOverall fraud rate: {fraud_rate:.2%}")

    print()
    print("=" * 60)
    print("TRANSACTION AMOUNT")
    print("=" * 60)
    print(df["amount"].describe())

    print()
    print("=" * 60)
    print("FRAUD BY PAYMENT METHOD")
    print("=" * 60)

    payment_fraud = (
        df.groupby("payment_method")["is_fraud"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
    )

    payment_fraud["fraud_rate"] = payment_fraud["mean"]

    print(payment_fraud)

    print()
    print("=" * 60)
    print("FRAUD BY MERCHANT CATEGORY")
    print("=" * 60)

    merchant_fraud = (
        df.groupby("merchant_category")["is_fraud"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
    )

    merchant_fraud["fraud_rate"] = merchant_fraud["mean"]

    print(merchant_fraud)

    print()
    print("=" * 60)
    print("INTERNATIONAL VS DOMESTIC")
    print("=" * 60)

    international_fraud = (
        df.groupby("is_international")["is_fraud"]
        .agg(["count", "sum", "mean"])
    )

    international_fraud["fraud_rate"] = international_fraud["mean"]

    print(international_fraud)

    print()
    print("=" * 60)
    print("EDA COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
