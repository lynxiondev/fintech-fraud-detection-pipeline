"""
Synthetic fintech transaction generator.

Generates realistic-looking transaction data for the fraud detection pipeline.
The data is synthetic and contains no real customer information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MERCHANT_CATEGORIES = [
    "groceries",
    "restaurants",
    "transport",
    "electronics",
    "travel",
    "entertainment",
    "utilities",
    "online_services",
]

PAYMENT_METHODS = [
    "card",
    "pix",
    "bank_transfer",
    "digital_wallet",
]

COUNTRIES = [
    "BR",
    "AR",
    "CL",
    "CO",
    "MX",
    "US",
    "PT",
]

DEVICE_TYPES = [
    "mobile",
    "desktop",
    "tablet",
]


def generate_transactions(
    n_transactions: int = 10_000,
    n_customers: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic fintech transaction dataset."""

    rng = np.random.default_rng(seed)

    customer_ids = rng.integers(
        low=1,
        high=n_customers + 1,
        size=n_transactions,
    )

    amounts = np.round(
        rng.lognormal(mean=3.5, sigma=1.0, size=n_transactions),
        2,
    )

    countries = rng.choice(
        COUNTRIES,
        size=n_transactions,
        p=[0.45, 0.15, 0.10, 0.08, 0.07, 0.10, 0.05],
    )

    home_countries = rng.choice(
        ["BR", "AR", "CL", "CO", "MX"],
        size=n_customers,
        p=[0.50, 0.20, 0.10, 0.10, 0.10],
    )

    customer_home_country = home_countries[customer_ids - 1]

    is_international = (
        countries != customer_home_country
    ).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_transactions + 1),
            "timestamp": pd.date_range(
                start="2026-01-01",
                periods=n_transactions,
                freq="5min",
            ),
            "customer_id": customer_ids,
            "amount": amounts,
            "currency": "USD",
            "merchant_category": rng.choice(
                MERCHANT_CATEGORIES,
                size=n_transactions,
            ),
            "payment_method": rng.choice(
                PAYMENT_METHODS,
                size=n_transactions,
                p=[0.45, 0.30, 0.15, 0.10],
            ),
            "country": countries,
            "device_type": rng.choice(
                DEVICE_TYPES,
                size=n_transactions,
                p=[0.75, 0.20, 0.05],
            ),
            "is_international": is_international,
        }
    )

    # Fraud probability increases with several behavioral signals.
    fraud_score = (
        0.01
        + 0.035 * (df["amount"] > 300).astype(float)
        + 0.025 * df["is_international"]
        + 0.025 * (
            df["payment_method"] == "digital_wallet"
        ).astype(float)
        + 0.02 * (
            df["merchant_category"] == "electronics"
        ).astype(float)
    )

    fraud_score = np.clip(fraud_score, 0, 0.50)

    df["is_fraud"] = rng.binomial(
        n=1,
        p=fraud_score,
        size=n_transactions,
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fintech transactions."
    )

    parser.add_argument(
        "--transactions",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/transactions.csv",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_transactions(
        n_transactions=args.transactions,
        n_customers=args.customers,
    )

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} transactions.")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
