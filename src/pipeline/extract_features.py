from pathlib import Path

from google.cloud import bigquery


PROJECT_ID = "fintech-fraud-detection-508019"
DATASET = "fintech_fraud"
TABLE = "fct_fraud_transactions"

OUTPUT_PATH = Path("data/processed/transactions_features.csv")


def extract_features() -> None:
    """Extract the final fraud feature mart from BigQuery into a local CSV."""

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        ORDER BY timestamp, transaction_id
    """

    df = client.query(query).to_dataframe()

    if df.empty:
        raise ValueError("BigQuery feature mart returned no rows.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Extracted {len(df)} rows and {len(df.columns)} columns.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_features()
