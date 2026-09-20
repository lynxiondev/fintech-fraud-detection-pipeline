from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


# --- Configuración de paths ---
REPO_ROOT = Path("/workspaces/fintech-fraud-detection-pipeline")
DBT_PROJECT = REPO_ROOT / "dbt_project"

# Entorno virtual del proyecto (dependencias)
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DBT = REPO_ROOT / ".venv" / "bin" / "dbt"


def run_command(command: list[str], cwd: Path) -> None:
    """Run a command and fail the Airflow task if it returns an error."""
    print(f"Running: {' '.join(command)}")
    print(f"Working directory: {cwd}")

    # Capturar salida para mostrarla en Airflow UI
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    # Mostrar logs en Airflow
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def dbt_seed() -> None:
    run_command([str(DBT), "seed"], DBT_PROJECT)


def dbt_build() -> None:
    run_command([str(DBT), "build"], DBT_PROJECT)


def extract_features() -> None:
    run_command([str(PYTHON), "-m", "src.pipeline.extract_features"], REPO_ROOT)


def prepare_dataset() -> None:
    run_command([str(PYTHON), "-m", "src.ml.prepare_dataset"], REPO_ROOT)


def train_model() -> None:
    run_command([str(PYTHON), "-m", "src.ml.train"], REPO_ROOT)


def generate_predictions() -> None:
    run_command([str(PYTHON), "-m", "src.ml.predict"], REPO_ROOT)


def evaluate_model() -> None:
    run_command([str(PYTHON), "-m", "src.ml.evaluate"], REPO_ROOT)


# --- Configuración del DAG ---
default_args = {
    "owner": "lynxiondev",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="fraud_pipeline",
    default_args=default_args,
    description="End-to-end fraud detection pipeline with dbt and ML",
    schedule=None,  # Manual trigger
    catchup=False,
    tags=["fintech", "fraud", "dbt", "ml"],
    max_active_runs=1,
) as dag:

    seed = PythonOperator(
        task_id="dbt_seed",
        python_callable=dbt_seed,
    )

    build = PythonOperator(
        task_id="dbt_build",
        python_callable=dbt_build,
    )

    extract = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features,
    )

    prepare = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_dataset,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # Flujo secuencial
    seed >> build >> extract >> prepare >> train >> predict >> evaluate