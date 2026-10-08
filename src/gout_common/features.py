"""Ingeniería de features ÚNICA para entrenamiento y serving (evita training-serving skew).

Replica la memoria v2 (sección 5): 13 atributos, target TOTAL, partición 2023–2024 / 2025.
Diferencia: la serie es (SUCURSAL, SKU) con SKU único; en la memoria era (SUCURSAL, PRODUCTO),
y los nombres repetidos (ECLAIR x4, PAN x3) mezclaban variantes en los rezagos.

La codificación, la imputación de rezagos y el escalado NO están acá: van dentro del
Pipeline de sklearn (pipeline.py), así se ajustan solo en train y viajan con el modelo.
"""
import pandas as pd

from gout_common.external import TEMP_MEDIA_MENSUAL

KEYS = ['SUCURSAL', 'SKU']
TARGET = 'TOTAL'
CAT_COLS = ['SKU', 'SUCURSAL']                                   # -> PRODUCTO_ENC, SUCURSAL_ENC
NUM_COLS_1 = ['DIA_SEMANA', 'MES', 'AÑO', 'STOCK', 'DEVOLUCIONES']
LAG_COLS = ['LAG_1', 'LAG_2', 'ROLLING_MEAN_3']
NUM_COLS_2 = ['ES_FERIADO', 'TEMP_MAX', 'TEMP_MIN']
FEATURES = CAT_COLS + NUM_COLS_1 + LAG_COLS + NUM_COLS_2           # mismo orden que la memoria


def _add_calendar_and_external(df: pd.DataFrame, feriados: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['DIA_SEMANA'] = df['FECHA'].dt.dayofweek
    df['MES'] = df['FECHA'].dt.month
    df['AÑO'] = df['FECHA'].dt.year

    df = df.drop(columns=['ES_FERIADO', 'TEMP_MAX', 'TEMP_MIN'], errors='ignore')
    df = df.merge(feriados[['FECHA', 'ES_FERIADO']], on='FECHA', how='left')
    df['ES_FERIADO'] = df['ES_FERIADO'].fillna(0).astype(int)

    df = df.merge(clima[['FECHA', 'TEMP_MAX', 'TEMP_MIN']], on='FECHA', how='left')
    df['TEMP_MAX'] = df['TEMP_MAX'].fillna(df['MES'].map(lambda m: TEMP_MEDIA_MENSUAL.get(m, 20) + 3))
    df['TEMP_MIN'] = df['TEMP_MIN'].fillna(df['MES'].map(lambda m: TEMP_MEDIA_MENSUAL.get(m, 20) - 3))
    return df


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Rezagos por posición dentro de la serie (fechas de despacho), no por días calendario."""
    df = df.sort_values(KEYS + ['FECHA'], kind='stable').reset_index(drop=True)
    g = df.groupby(KEYS)[TARGET]
    df['LAG_1'] = g.shift(1)
    df['LAG_2'] = g.shift(2)
    df['ROLLING_MEAN_3'] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    return df


def build_features(history: pd.DataFrame, feriados: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """history: FECHA, SUCURSAL, SKU, STOCK, DEVOLUCIONES, TOTAL. Devuelve FECHA + FEATURES + TOTAL."""
    df = _add_lags(_add_calendar_and_external(history, feriados, clima))
    return df[['FECHA'] + FEATURES + [TARGET]]


def temporal_split(df: pd.DataFrame, test_year: int = 2025):
    """Partición cronológica estricta: años < test_year -> train; test_year -> test."""
    anio = pd.to_datetime(df['FECHA']).dt.year
    return df[anio < test_year].copy(), df[anio == test_year].copy()


def build_inference_row(history: pd.DataFrame, fecha, sucursal: str, sku: str,
                        stock: int, devoluciones: int,
                        feriados: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """Fila de features para predecir (fecha, sucursal, sku) usando solo historia anterior a fecha.
    STOCK y DEVOLUCIONES son del mismo día (los informa la sucursal al pedir)."""
    fecha = pd.Timestamp(fecha)
    serie = history[(history['SUCURSAL'] == sucursal) & (history['SKU'] == sku) &
                    (pd.to_datetime(history['FECHA']) < fecha)].sort_values('FECHA')
    ult = serie[TARGET].tail(3).tolist()
    row = pd.DataFrame([{'FECHA': fecha, 'SUCURSAL': sucursal, 'SKU': sku,
                         'STOCK': stock, 'DEVOLUCIONES': devoluciones}])
    row = _add_calendar_and_external(row, feriados, clima)
    row['LAG_1'] = ult[-1] if len(ult) >= 1 else float('nan')
    row['LAG_2'] = ult[-2] if len(ult) >= 2 else float('nan')
    row['ROLLING_MEAN_3'] = sum(ult) / 3 if len(ult) == 3 else float('nan')
    return row[FEATURES]
