"""Variables externas del modelo (memoria v2, sección 3): feriados nacionales y temperatura de Buenos Aires.

En el TP se descargan en el DAG de ETL y se guardan en gout_db; el serving las lee de ahí
(no se llama a APIs externas en cada predicción).
"""
import pandas as pd
import requests

FERIADOS_BACKUP = [
    # 2023
    '2023-01-01', '2023-02-20', '2023-02-21', '2023-04-02', '2023-04-06',
    '2023-04-07', '2023-05-01', '2023-05-25', '2023-06-17', '2023-06-19',
    '2023-07-09', '2023-08-21', '2023-10-13', '2023-11-20', '2023-12-08', '2023-12-25',
    # 2024
    '2024-01-01', '2024-02-12', '2024-02-13', '2024-03-24', '2024-03-28',
    '2024-03-29', '2024-04-02', '2024-05-01', '2024-05-25', '2024-06-17',
    '2024-06-20', '2024-07-09', '2024-08-17', '2024-10-11', '2024-11-18',
    '2024-12-08', '2024-12-25',
    # 2025
    '2025-01-01', '2025-03-03', '2025-03-04', '2025-03-24', '2025-04-02',
    '2025-04-17', '2025-04-18', '2025-05-01', '2025-05-25', '2025-06-16',
    '2025-06-20', '2025-07-09', '2025-08-17', '2025-10-12', '2025-11-17',
    '2025-12-08', '2025-12-25',
]

# Temperatura media histórica mensual de Buenos Aires (respaldo)
TEMP_MEDIA_MENSUAL = {1: 29.0, 2: 28.0, 3: 25.0, 4: 20.0, 5: 15.0, 6: 11.0,
                      7: 10.0, 8: 12.0, 9: 15.0, 10: 19.0, 11: 24.0, 12: 27.0}


def fetch_feriados(anios) -> pd.DataFrame:
    """FECHA, ES_FERIADO=1. API date.nager.at; si falla, listado de respaldo."""
    fechas = set()
    for anio in anios:
        try:
            resp = requests.get(f'https://date.nager.at/api/v3/PublicHolidays/{anio}/AR', timeout=10)
            resp.raise_for_status()
            fechas.update(f['date'] for f in resp.json())
        except Exception:
            fechas.update(d for d in FERIADOS_BACKUP if d.startswith(str(anio)))
    return pd.DataFrame({'FECHA': pd.to_datetime(sorted(fechas)), 'ES_FERIADO': 1})


def fetch_clima(start: str, end: str) -> pd.DataFrame:
    """FECHA, TEMP_MAX, TEMP_MIN diarias (Open-Meteo archive); si falla, respaldo mensual ±3 °C."""
    try:
        url = ('https://archive-api.open-meteo.com/v1/archive'
               '?latitude=-34.6037&longitude=-58.3816'
               f'&start_date={start}&end_date={end}'
               '&daily=temperature_2m_max,temperature_2m_min'
               '&timezone=America%2FArgentina%2FBuenos_Aires')
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        d = resp.json()['daily']
        return pd.DataFrame({'FECHA': pd.to_datetime(d['time']),
                             'TEMP_MAX': d['temperature_2m_max'],
                             'TEMP_MIN': d['temperature_2m_min']})
    except Exception:
        df = pd.DataFrame({'FECHA': pd.date_range(start, end)})
        media = df['FECHA'].dt.month.map(TEMP_MEDIA_MENSUAL)
        return df.assign(TEMP_MAX=media + 3, TEMP_MIN=media - 3)
