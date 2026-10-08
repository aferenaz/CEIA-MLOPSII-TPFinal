"""Carga del modelo en producción desde el registry de MLflow (alias configurable)."""
from functools import lru_cache

import mlflow

from gout_common import config


@lru_cache(maxsize=1)
def load_prod_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(uri)


def reload_model():
    load_prod_model.cache_clear()
    return load_prod_model()
