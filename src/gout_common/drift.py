"""S7 — Monitoreo: drift de datos + degradación del modelo. Independiente de Airflow.

Drift de datos (PSI por atributo): se compara una ventana reciente contra una referencia ESTACIONAL
(las filas de entrenamiento de los mismos meses), para no confundir estacionalidad con drift
(p. ej. temperatura de junio vs. todo el año).
    PSI < 0.1 estable · 0.1–0.2 moderado · >= 0.2 drift

Degradación: MAE online (evaluacion_online: pedido real vs. pronóstico, generado por el streaming)
contra el MAE de test del modelo en prod. Degrada si supera al de test en más de TOLERANCIA_MAE.

Decisión: reentrenar si hay drift en la proporción de atributos >= UMBRAL_DATASET o si degrada.
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gout_common import config
from gout_common.features import TARGET, build_features

log = logging.getLogger(__name__)

# ES_FERIADO y las variables de calendario no se monitorean: las fija el calendario, no los datos
MONITOREADOS = ['STOCK', 'DEVOLUCIONES', 'LAG_1', 'LAG_2', 'ROLLING_MEAN_3',
                'TEMP_MAX', 'TEMP_MIN', TARGET]
PSI_DRIFT = float(os.getenv('PSI_DRIFT', '0.2'))
PSI_MODERADO = 0.1
UMBRAL_DATASET = float(os.getenv('UMBRAL_DATASET', '0.3'))    # proporción de atributos con drift
TOLERANCIA_MAE = float(os.getenv('TOLERANCIA_MAE', '0.2'))    # +20 % sobre el MAE de test
VENTANA_DIAS = int(os.getenv('VENTANA_DIAS', '30'))
TABLE_ONLINE = 'evaluacion_online'


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia (robusto a discretos)."""
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if ref.empty or cur.empty:
        return float('nan')
    if ref.nunique() <= 10:                               # discreta/binaria: PSI por categoría
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        q = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
        return float(np.sum((q - p) * np.log(q / p)))
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:                                   # variable constante en la referencia
        cortes = np.array([ref.min() - 0.5, ref.min() + 0.5])
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref)
    q = np.histogram(cur, cortes)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_de_datos(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in MONITOREADOS:
        v = psi(referencia[c], actual[c])
        estado = 'drift' if v >= PSI_DRIFT else 'moderado' if v >= PSI_MODERADO else 'estable'
        filas.append({'atributo': c, 'psi': round(v, 4), 'estado': estado,
                      'media_ref': round(referencia[c].mean(), 3), 'media_actual': round(actual[c].mean(), 3)})
    return pd.DataFrame(filas)


def ventana_actual(engine) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Últimos VENTANA_DIAS con pedidos evaluados online; si no hay, los últimos de la historia."""
    for tabla in (TABLE_ONLINE, config.TABLE_HISTORY):
        try:
            h = pd.read_sql(f'SELECT MAX("FECHA") AS h FROM {tabla}', engine)['h'][0]
        except Exception:
            continue
        if h is not None:
            hasta = pd.Timestamp(h)
            return hasta - pd.Timedelta(days=VENTANA_DIAS - 1), hasta
    raise ValueError('No hay datos para definir la ventana')


def datos_ventana(engine, desde, hasta) -> pd.DataFrame:
    """Features de la ventana, calculadas con la misma función que el entrenamiento."""
    from gout_common.etl import load_externals
    hist = pd.read_sql(f'SELECT * FROM {config.TABLE_HISTORY}', engine, parse_dates=['FECHA'])
    fer, cli = load_externals(engine)
    df = build_features(hist, fer, cli)
    return df[(df['FECHA'] >= desde) & (df['FECHA'] <= hasta)]


def referencia_estacional(engine, meses) -> pd.DataFrame:
    train = pd.read_sql(f'SELECT * FROM {config.TABLE_TRAIN}', engine, parse_dates=['FECHA'])
    ref = train[train['FECHA'].dt.month.isin(meses)]
    return ref if len(ref) else train


def degradacion(engine, desde, hasta) -> dict:
    import mlflow
    from mlflow import MlflowClient
    try:
        on = pd.read_sql(f'SELECT * FROM {TABLE_ONLINE}', engine, parse_dates=['FECHA'])
        on = on[(on['FECHA'] >= desde) & (on['FECHA'] <= hasta)]
    except Exception:
        on = pd.DataFrame()
    if on.empty:
        return {'n_online': 0, 'mae_online': None, 'mae_test': None, 'degrada': False}
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    c = MlflowClient()
    mv = c.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    mae_test = c.get_run(mv.run_id).data.metrics['test_mae']
    err = on['PEDIDO'] - on['PRONOSTICO']
    mae_on = float(err.abs().mean())
    return {'n_online': len(on), 'mae_online': round(mae_on, 4), 'mae_test': round(mae_test, 4),
            'sesgo_online': round(float(err.mean()), 4), 'modelo_version': mv.version,
            'degrada': mae_on > mae_test * (1 + TOLERANCIA_MAE)}


def reporte_html(res: dict, tabla: pd.DataFrame) -> str:
    color = {'drift': '#f8d7da', 'moderado': '#fff3cd', 'estable': '#d1e7dd'}
    filas = ''.join(
        f"<tr style='background:{color[r.estado]}'><td>{r.atributo}</td><td>{r.psi}</td>"
        f"<td>{r.estado}</td><td>{r.media_ref}</td><td>{r.media_actual}</td></tr>" for r in tabla.itertuples())
    d = res['degradacion']
    return f"""<html><head><meta charset='utf-8'><title>Drift Gout</title></head>
<body style='font-family:sans-serif;max-width:900px;margin:auto'>
<h2>Monitoreo Gout — {res['desde']} a {res['hasta']}</h2>
<p><b>Decisión: {'REENTRENAR' if res['reentrenar'] else 'sin acción'}</b> — {res['motivo']}</p>
<h3>Drift de datos (referencia estacional: meses {res['meses_ref']}, {res['n_ref']:,} filas; actual {res['n_actual']:,} filas)</h3>
<table border='1' cellpadding='6' style='border-collapse:collapse'>
<tr><th>Atributo</th><th>PSI</th><th>Estado</th><th>Media ref.</th><th>Media actual</th></tr>{filas}</table>
<p>Atributos con drift: {res['n_drift']} de {len(tabla)} (umbral de dataset {UMBRAL_DATASET:.0%}, PSI ≥ {PSI_DRIFT}).</p>
<h3>Desempeño online</h3>
<p>Pedidos evaluados: {d['n_online']:,} · MAE online: {d['mae_online']} · MAE test (modelo v{d.get('modelo_version')}): {d['mae_test']}
 · sesgo (real − pronóstico): {d.get('sesgo_online')} · tolerancia +{TOLERANCIA_MAE:.0%}</p>
<p style='color:#666'>Generado {res['ts']}</p></body></html>"""


def guardar_reporte(html: str, nombre: str) -> str:
    """En Docker va al bucket de MinIO (REPORTS_BUCKET); si no, a un directorio local."""
    bucket = os.getenv('REPORTS_BUCKET')
    if bucket:
        import boto3
        key = f'reportes_drift/{nombre}'
        boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=html.encode(), ContentType='text/html')
        return f's3://{bucket}/{key}'
    d = os.getenv('REPORTS_DIR', 'reportes_drift')
    os.makedirs(d, exist_ok=True)
    ruta = os.path.join(d, nombre)
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(html)
    return ruta


def run_drift(engine, actual: pd.DataFrame | None = None, desde=None, hasta=None, guardar: bool = True) -> dict:
    """Si 'actual' viene dado (p. ej. un escenario simulado), se usa en lugar de la ventana real."""
    if desde is None or hasta is None:
        desde, hasta = ventana_actual(engine)
    if actual is None:
        actual = datos_ventana(engine, desde, hasta)
    meses = sorted(actual['FECHA'].dt.month.unique().tolist())
    ref = referencia_estacional(engine, meses)
    tabla = drift_de_datos(ref, actual)
    n_drift = int((tabla['estado'] == 'drift').sum())
    deg = degradacion(engine, desde, hasta)

    drift_dataset = n_drift / len(tabla) >= UMBRAL_DATASET
    motivos = []
    if drift_dataset:
        motivos.append(f'drift en {n_drift} atributos: ' + ', '.join(tabla.loc[tabla.estado == 'drift', 'atributo']))
    if deg['degrada']:
        motivos.append(f"MAE online {deg['mae_online']} > MAE test {deg['mae_test']} +{TOLERANCIA_MAE:.0%}")
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    res = {'ts': ts, 'desde': str(pd.Timestamp(desde).date()), 'hasta': str(pd.Timestamp(hasta).date()),
           'meses_ref': meses, 'n_ref': len(ref), 'n_actual': len(actual), 'n_drift': n_drift,
           'drift_dataset': drift_dataset, 'degradacion': deg,
           'reentrenar': bool(drift_dataset or deg['degrada']),
           'motivo': '; '.join(motivos) or 'sin drift ni degradación'}

    if guardar:
        res['reporte'] = guardar_reporte(reporte_html(res, tabla), f"drift_{ts:%Y%m%d_%H%M%S}.html")
        pd.DataFrame([{'TS': ts, 'DESDE': res['desde'], 'HASTA': res['hasta'], 'N_ACTUAL': len(actual),
                       'N_DRIFT': n_drift, 'PSI': json.dumps(dict(zip(tabla.atributo, tabla.psi))),
                       'MAE_ONLINE': deg['mae_online'], 'MAE_TEST': deg['mae_test'],
                       'REENTRENAR': res['reentrenar'], 'MOTIVO': res['motivo'],
                       'REPORTE': res['reporte']}]) \
            .to_sql(config.TABLE_DRIFT, engine, if_exists='append', index=False)
    res['tabla'] = tabla
    log.info('Drift %s→%s: reentrenar=%s (%s)', res['desde'], res['hasta'], res['reentrenar'], res['motivo'])
    return res
