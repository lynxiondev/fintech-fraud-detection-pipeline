# 🛡️ Fintech Fraud Detection Pipeline [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/) [![Airflow](https://img.shields.io/badge/Airflow-3.x-017CEE.svg)](https://airflow.apache.org/) [![dbt](https://img.shields.io/badge/dbt-1.12-FF694B.svg)](https://www.getdbt.com/) [![BigQuery](https://img.shields.io/badge/BigQuery-GCP-669DF6.svg)](https://cloud.google.com/bigquery) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) Pipeline de datos *end-to-end* para la detección de fraude en transacciones fintech. Este proyecto orquesta la ingesta, transformación, entrenamiento y evaluación de un modelo de Machine Learning utilizando un stack moderno de ingeniería de datos. ## 📖 Descripción del Proyecto El objetivo de este pipeline es procesar transacciones financieras, generar features de comportamiento (como frecuencia de transacciones en la última hora/día) y predecir la probabilidad de fraude. El flujo está completamente automatizado mediante **Apache Airflow**, garantizando reproducibilidad, pruebas de calidad de datos con **dbt** y un enfoque modular en **Python**. ## 🏗️ Arquitectura del Pipeline El flujo de trabajo sigue una secuencia lineal y robusta: ```mermaid graph LR A[Raw Data / Seed] -->|dbt seed| B(BigQuery Staging) B -->|dbt build| C{Data Quality Tests} C -->|Pass| D[Feature Mart] D -->|Python Extract| E[Local CSV Features] E -->|Python Split| F[Train / Test Datasets] F -->|Python Train| G[(Logistic Regression Model)] G -->|Python Predict| H[Predictions CSV] H -->|Python Evaluate| I[Model Metrics Report] style C fill:#d4edda,stroke:#28a745,stroke-width:2px style G fill:#cce5ff,stroke:#007bff,stroke-width:2px style I fill:#fff3cd,stroke:#ffc107,stroke-width:2px

📂 Estructura del Proyecto

fintech-fraud-detection-pipeline/
├── dags/                   # Definición del DAG de Airflow (fraud_pipeline.py)
├── dbt_project/            # Proyecto dbt (modelos, tests, seeds, macros)
│   ├── models/             # Staging y Feature Mart (int_, fct_)
│   └── tests/              # Pruebas de calidad de datos (not_null, unique, accepted_values)
├── src/                    # Código fuente de Python
│   ├── pipeline/           # Lógica de extracción de features desde BigQuery
│   ├── ml/                 # Módulos de ML: prepare_dataset, train, predict, evaluate
│   └── utils/              # Utilidades y validaciones
├── data/                   # Datos procesados localmente (features, predictions)
├── models/                 # Artefactos del modelo entrenado (.joblib)
├── .venv/                  # Entorno virtual del proyecto (dependencias de datos/ML)
└── airflow_venv/           # Entorno virtual exclusivo de Airflow

⚙️ Prerrequisitos

    Python 3.12+ instalado.
    Cuenta de Google Cloud Platform (GCP) con un proyecto activo y la API de BigQuery habilitada.
    Autenticación GCP:

       gcloud auth application-default login

    Doble entorno virtual (recomendado para aislamiento):
       .venv: Contiene las dependencias del proyecto (dbt, google-cloud-bigquery, scikit-learn, pandas).
       airflow_venv: Contiene exclusivamente las dependencias de Apache Airflow 3.x.

🚀 Cómo Ejecutar el Pipeline

1. Iniciar los servicios de Airflow
Asegúrate de estar en el entorno de Airflow (source airflow_venv/bin/activate) y levanta los servicios en terminales separadas:

    airflow api-server    # Terminal 1
    airflow scheduler     # Terminal 2
    airflow dag-processor # Terminal 3

2. Ejecutar el DAG
Puedes ejecutar el pipeline de forma manual desde la CLI:

    airflow dags trigger fraud_pipeline

(Opcional) Para pruebas rápidas y depuración sin necesidad del scheduler:

    airflow dags test fraud_pipeline 2026-09-21


3. Monitorear
Visita la UI de Airflow en http://localhost:8080 para ver el gráfico de ejecución, los logs en tiempo real y el estado de cada tarea.
📊 Resultados y Métricas del Modelo
En la ejecución de referencia con 10,000 transacciones sintéticas:

    División temporal: 8,000 filas para entrenamiento, 2,000 para prueba.
    Tasa de fraude: ~3.3% (Train), ~3.1% (Test).
    Modelo: Regresión Logística.
    ROC-AUC: 0.6255
    PR-AUC: 0.0570
    Evaluación Operativa (tasa de alerta del 10%):
        Alertas generadas: 200
        Fraudes capturados: 12
        Recall: 19.05% | Precisión: 6.00%

    Nota: Las métricas reflejan un baseline funcional. En un entorno de producción, se recomienda experimentar con modelos como XGBoost/LightGBM y técnicas de manejo de desbalanceo (SMOTE, class weights).

🛠️ Mejoras Futuras

    Implementar un modelo de Gradient Boosting (XGBoost) para mejorar el PR-AUC.
    Migrar la extracción de datos de CSV local a una tabla de predicciones en BigQuery.
    Agregar notificaciones de Slack/Email en caso de fallo del DAG o degradación del modelo.
    Orquestar el reentrenamiento automático del modelo (MLOps).

📄 Licencia
Este proyecto está bajo la Licencia MIT. Siéntete libre de usarlo, modificarlo y aprender de él.

Desarrollado con ❤️ y mucho café por @lynxiondev.