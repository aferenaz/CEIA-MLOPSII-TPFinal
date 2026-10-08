"""Configuración compartida (Airflow, API, gRPC). Todo sale de variables de entorno."""
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
GOUT_DB_URL = os.getenv("GOUT_DB_URL", "postgresql+psycopg2://airflow:airflow@localhost:5432/gout_db")
DATA_BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
MODEL_NAME = os.getenv("MODEL_NAME", "gout_demand_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")
RAW_KEY = os.getenv("RAW_KEY", "raw/pedidos_anon.csv")
TEST_YEAR = int(os.getenv("TEST_YEAR", "2025"))

# Tablas en gout_db
TABLE_HISTORY = "ventas_historicas"     # serie limpia por (FECHA, SUCURSAL, SKU)
TABLE_FERIADOS = "feriados"             # FECHA, ES_FERIADO
TABLE_CLIMA = "clima"                   # FECHA, TEMP_MAX, TEMP_MIN
TABLE_TRAIN = "features_train"          # features + target, años < TEST_YEAR
TABLE_TEST = "features_test"            # features + target, TEST_YEAR
TABLE_PREDICTIONS = "predicciones"      # log de predicciones servidas (insumo para drift)
TABLE_ALERTS = "alertas_desvio"         # salida del consumidor Kafka
TABLE_DRIFT = "metricas_drift"          # salida del DAG de monitoreo
