"""Lógica del ETL, independiente de Airflow (el DAG solo orquesta; esto se puede testear aparte).

extraer   : CSV anonimizado en MinIO -> validación -> gout_db.ventas_historicas
externas  : feriados + clima para el rango de fechas -> gout_db.feriados / gout_db.clima
features  : build_features + partición temporal -> gout_db.features_train / features_test
"""
import io
import logging

import pandas as pd
from sqlalchemy import create_engine

from gout_common import config
from gout_common.external import fetch_clima, fetch_feriados
from gout_common.features import FEATURES, TARGET, build_features, temporal_split

log = logging.getLogger(__name__)

HISTORY_COLS = ['FECHA', 'SUCURSAL', 'SKU', 'FECHA_ELAB', 'FECHA_VTO',
                'PEDIDO', 'STOCK', 'DEVOLUCIONES', 'TOTAL']
QTY_COLS = ['PEDIDO', 'STOCK', 'DEVOLUCIONES', 'TOTAL']


def get_engine(url: str | None = None):
    return create_engine(url or config.GOUT_DB_URL)


def read_raw_from_s3(bucket: str = config.DATA_BUCKET, key: str = config.RAW_KEY) -> pd.DataFrame:
    """El endpoint de MinIO sale de AWS_ENDPOINT_URL_S3 (definido en el docker-compose)."""
    import boto3  # import diferido: en Colab no hace falta S3
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    faltan = set(HISTORY_COLS) - set(df.columns)
    if faltan:
        raise ValueError(f'Faltan columnas en el CSV: {sorted(faltan)}')
    df = df[HISTORY_COLS].copy()
    for c in ['FECHA', 'FECHA_ELAB', 'FECHA_VTO']:
        df[c] = pd.to_datetime(df[c])
    # igual que la memoria: cantidades numéricas, nulos -> 0, enteros
    for c in QTY_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    dup = df.duplicated(['FECHA', 'SUCURSAL', 'SKU']).sum()
    if dup:
        raise ValueError(f'{dup} filas duplicadas por (FECHA, SUCURSAL, SKU)')
    neg = (df[QTY_COLS] < 0).any(axis=1).sum()
    if neg:
        raise ValueError(f'{neg} filas con cantidades negativas')
    return df.sort_values(['SUCURSAL', 'SKU', 'FECHA']).reset_index(drop=True)


def run_extract(engine, raw: pd.DataFrame | None = None) -> dict:
    df = validate_history(read_raw_from_s3() if raw is None else raw)
    df.to_sql(config.TABLE_HISTORY, engine, if_exists='replace', index=False, chunksize=10_000)
    stats = {'filas': len(df), 'sucursales': df['SUCURSAL'].nunique(), 'skus': df['SKU'].nunique(),
             'desde': str(df['FECHA'].min().date()), 'hasta': str(df['FECHA'].max().date())}
    log.info('Historia cargada: %s', stats)
    return stats


def run_externals(engine) -> dict:
    rango = pd.read_sql(f'SELECT MIN("FECHA") AS d, MAX("FECHA") AS h FROM {config.TABLE_HISTORY}', engine)
    desde, hasta = pd.to_datetime(rango.loc[0, 'd']), pd.to_datetime(rango.loc[0, 'h'])
    feriados = fetch_feriados(range(desde.year, hasta.year + 1))
    clima = fetch_clima(f'{desde.year}-01-01', f'{hasta.year}-12-31')
    feriados.to_sql(config.TABLE_FERIADOS, engine, if_exists='replace', index=False)
    clima.to_sql(config.TABLE_CLIMA, engine, if_exists='replace', index=False)
    stats = {'feriados': len(feriados), 'dias_clima': len(clima),
             'clima_real': bool(clima['TEMP_MAX'].nunique() > 12)}   # respaldo = 12 valores mensuales
    log.info('Externas cargadas: %s', stats)
    return stats


def load_externals(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    fer = pd.read_sql(f'SELECT * FROM {config.TABLE_FERIADOS}', engine, parse_dates=['FECHA'])
    cli = pd.read_sql(f'SELECT * FROM {config.TABLE_CLIMA}', engine, parse_dates=['FECHA'])
    return fer, cli


def run_features(engine, test_year: int = config.TEST_YEAR) -> dict:
    hist = pd.read_sql(f'SELECT * FROM {config.TABLE_HISTORY}', engine, parse_dates=['FECHA'])
    fer, cli = load_externals(engine)
    df = build_features(hist, fer, cli)
    train, test = temporal_split(df, test_year)
    if train.empty or test.empty:
        raise ValueError(f'Partición vacía con test_year={test_year}: train={len(train)}, test={len(test)}')
    train.to_sql(config.TABLE_TRAIN, engine, if_exists='replace', index=False, chunksize=10_000)
    test.to_sql(config.TABLE_TEST, engine, if_exists='replace', index=False, chunksize=10_000)
    stats = {'train': len(train), 'test': len(test), 'features': len(FEATURES), 'target': TARGET,
             'nulos_lags_train': int(train[['LAG_1', 'LAG_2', 'ROLLING_MEAN_3']].isna().sum().sum())}
    log.info('Features: %s', stats)
    return stats
