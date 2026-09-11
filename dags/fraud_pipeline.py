from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def start_pipeline():
    print("Starting fraud detection pipeline")


def process_data():
    print("Processing transaction data")


def finish_pipeline():
    print("Fraud detection pipeline finished")


with DAG(
    dag_id="fraud_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fintech", "fraud"],
) as dag:

    start = PythonOperator(
        task_id="start_pipeline",
        python_callable=start_pipeline,
    )

    process = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
    )

    finish = PythonOperator(
        task_id="finish_pipeline",
        python_callable=finish_pipeline,
    )

    start >> process >> finish
