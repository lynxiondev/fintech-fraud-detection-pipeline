"""
Temporal feature engineering for fraud detection.

All customer behavioral features are calculated using only
information available before the current transaction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/transactions.csv")
OUTPUT_PATH = Path("data/processed/transactions_features.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical customer features without temporal leakage."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Historical transaction count
    # ---------------------------------------------------------
    df["customer_transaction_count"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    # ---------------------------------------------------------
    # Historical average transaction amount
    # ---------------------------------------------------------
    df["customer_avg_amount"] = (
        df.groupby("customer_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ---------------------------------------------------------
    # Current amount relative to historical average
    # ---------------------------------------------------------
    df["amount_vs_customer_avg"] = (
        df["amount"]
        / df["customer_avg_amount"]
    )

    # ---------------------------------------------------------
    # Historical unique countries
    # ---------------------------------------------------------
    def historical_unique_count(
        series: pd.Series,
    ) -> pd.Series:
        seen: set[str] = set()
        result: list[int] = []

        for value in series:
            result.append(len(seen))
            seen.add(value)

        return pd.Series(
            result,
            index=series.index,
        )

    df["customer_unique_countries"] = (
        df.groupby("customer_id")["country"]
        .transform(historical_unique_count)
    )

    # ---------------------------------------------------------
    # Historical unique devices
    # ---------------------------------------------------------
    df["customer_unique_devices"] = (
        df.groupby("customer_id")["device_type"]
        .transform(historical_unique_count)
    )

    # ---------------------------------------------------------
    # New country indicator
    # ---------------------------------------------------------
    def mark_new_value(
        series: pd.Series,
    ) -> pd.Series:
        seen: set[str] = set()
        result: list[int] = []

        for value in series:
            result.append(int(value not in seen))
            seen.add(value)

        return pd.Series(
            result,
            index=series.index,
        )

    df["is_new_country"] = (
        df.groupby("customer_id")["country"]
        .transform(mark_new_value)
    )

    # ---------------------------------------------------------
    # New device indicator
    # ---------------------------------------------------------
    df["is_new_device"] = (
        df.groupby("customer_id")["device_type"]
        .transform(mark_new_value)
    )

    return df


def main() -> None:
    """Run the feature engineering pipeline."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    featured_df = build_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    featured_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Feature engineering completed.")
    print(f"Rows: {len(featured_df):,}")
    print(f"Columns: {len(featured_df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
