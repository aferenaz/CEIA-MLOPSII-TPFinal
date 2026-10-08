"""Entrenamiento y promoción (champion/challenger), independiente de Airflow.

run_train   : features de gout_db -> pipeline XGBoost -> métricas en test -> MLflow (run + registry)
run_promote : compara challenger vs. champion (alias prod) sobre el MISMO test actual;
              si el challenger mejora el RMSE (o no hay champion), pasa a prod.
"""
import logging
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gout_common import config
from gout_common.features import FEATURES, TARGET
from gout_common.pipeline import XGB_PARAMS, build_pipeline, postprocess

log = logging.getLogger(__name__)
EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'gout_demand')


def _metricas(y_real, y_pred) -> dict:
    return {'rmse': float(np.sqrt(mean_squared_error(y_real, y_pred))),
            'mae': float(mean_absolute_error(y_real, y_pred)),
            'r2': float(r2_score(y_real, y_pred))}


def load_train_test(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_sql(f'SELECT * FROM {config.TABLE_TRAIN}', engine, parse_dates=['FECHA'])
    test = pd.read_sql(f'SELECT * FROM {config.TABLE_TEST}', engine, parse_dates=['FECHA'])
    return train, test


def run_train(engine, **xgb_overrides) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    train, test = load_train_test(engine)
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]

    with mlflow.start_run(run_name='xgboost') as run:
        pipe = build_pipeline(**xgb_overrides)
        pipe.fit(X_tr, y_tr)
        y_pred = postprocess(pipe.predict(X_te))
        met = _metricas(y_te, y_pred)

        mlflow.log_params({**XGB_PARAMS, **xgb_overrides})
        mlflow.log_params({'features': ','.join(FEATURES), 'test_year': config.TEST_YEAR,
                           'n_train': len(train), 'n_test': len(test),
                           'train_desde': str(train['FECHA'].min().date()),
                           'train_hasta': str(train['FECHA'].max().date())})
        mlflow.log_metrics({f'test_{k}': v for k, v in met.items()})

        # Artefactos: métricas por sucursal (TOTAL > 0, como la memoria) e importancia de atributos
        ev = test[['SUCURSAL']].assign(y_real=y_te.values, y_pred=y_pred)
        ev = ev[ev['y_real'] > 0]
        por_suc = pd.DataFrame([{'SUCURSAL': s, 'registros': len(g), **_metricas(g['y_real'], g['y_pred'])}
                                for s, g in ev.groupby('SUCURSAL')])
        imp = pd.DataFrame({'feature': FEATURES,
                            'importancia': pipe.named_steps['xgb'].feature_importances_}) \
            .sort_values('importancia', ascending=False)
        with tempfile.TemporaryDirectory() as d:
            por_suc.to_csv(f'{d}/metricas_por_sucursal.csv', index=False)
            imp.to_csv(f'{d}/importancia_atributos.csv', index=False)
            mlflow.log_artifacts(d, artifact_path='evaluacion')

        firma = infer_signature(X_tr.head(200), pipe.predict(X_tr.head(200)))
        info = mlflow.sklearn.log_model(pipe, artifact_path='model', signature=firma,
                                        input_example=X_tr.head(5),
                                        registered_model_name=config.MODEL_NAME)
        version = info.registered_model_version

    out = {'run_id': run.info.run_id, 'version': str(version), **met}
    log.info('Challenger entrenado: %s', out)
    return out


def run_promote(engine, version: str) -> dict:
    """Champion/challenger sobre el test actual. Devuelve la decisión."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    _, test = load_train_test(engine)
    X_te, y_te = test[FEATURES], test[TARGET]

    def evaluar(v):
        m = mlflow.sklearn.load_model(f'models:/{config.MODEL_NAME}/{v}')
        return _metricas(y_te, postprocess(m.predict(X_te)))

    chall = evaluar(version)
    try:
        champ_v = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS).version
    except Exception:
        champ_v = None

    if champ_v is None:
        promover, champ = True, None
    elif str(champ_v) == str(version):
        promover, champ = False, chall
    else:
        champ = evaluar(champ_v)
        promover = chall['rmse'] < champ['rmse']

    client.set_model_version_tag(config.MODEL_NAME, version, 'test_rmse', f"{chall['rmse']:.4f}")
    if promover:
        client.set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version)
        client.set_model_version_tag(config.MODEL_NAME, version, 'estado', 'champion')
        if champ_v:
            client.set_model_version_tag(config.MODEL_NAME, champ_v, 'estado', 'retirado')
    else:
        client.set_model_version_tag(config.MODEL_NAME, version, 'estado', 'rechazado')

    out = {'challenger': version, 'champion_anterior': champ_v, 'promovido': promover,
           'rmse_challenger': chall['rmse'], 'rmse_champion': champ['rmse'] if champ else None}
    log.info('Promoción: %s', out)
    return out


def notify_api_reload() -> bool:
    """Avisa a la API que recargue el modelo. No falla si la API no está levantada."""
    import requests
    url = os.getenv('API_URL', 'http://api:8800')
    try:
        requests.post(f'{url}/model/reload', timeout=10).raise_for_status()
        return True
    except Exception as e:
        log.warning('No se pudo notificar a la API (%s): %s', url, e)
        return False
