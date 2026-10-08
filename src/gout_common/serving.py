"""Lógica de predicción ÚNICA para REST, GraphQL y gRPC (así la comparación de protocolos es justa).

- Modelo: alias prod del registry de MLflow.
- Rezagos: últimas 3 fechas de la serie en gout_db.ventas_historicas (anteriores a la fecha pedida).
- Feriados y clima: tablas de gout_db, en memoria. Si la fecha no está, respaldo (0 / media mensual).
- STOCK y DEVOLUCIONES: los informa la sucursal al pedir.
- Cada predicción se registra en gout_db.predicciones (insumo del monitoreo de drift).
"""
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import text

from gout_common import config
from gout_common.etl import get_engine, load_externals
from gout_common.features import FEATURES, build_inference_row
from gout_common.pipeline import postprocess

log = logging.getLogger(__name__)


class NotFound(ValueError):
    """Sucursal o SKU inexistente."""


@dataclass
class Item:
    sku: str
    stock: int = 0
    devoluciones: int = 0


CHECK_ALIAS_CADA_S = 60   # cada cuánto se verifica si cambió el modelo en prod


class ForecastService:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self._lock = threading.Lock()
        self.model = None
        self.version = None
        self._ultimo_check = 0.0
        self.reload()

    # ------------------------------------------------------------------ carga
    def reload(self) -> str:
        with self._lock:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mv = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
            self.model = mlflow.sklearn.load_model(f'models:/{config.MODEL_NAME}/{mv.version}')
            self.version = str(mv.version)
            self.feriados, self.clima = load_externals(self.engine)
            cat = pd.read_sql(f'SELECT DISTINCT "SUCURSAL", "SKU" FROM {config.TABLE_HISTORY}', self.engine)
            self.sucursales = set(cat['SUCURSAL'])
            self.skus = set(cat['SKU'])
            self._ultimo_check = time.monotonic()
        log.info('Modelo %s v%s cargado', config.MODEL_NAME, self.version)
        return self.version

    def _refrescar_si_cambio(self):
        """Si train_gout promovió otra versión, la carga (sirve para gRPC, que no recibe /model/reload)."""
        if time.monotonic() - self._ultimo_check < CHECK_ALIAS_CADA_S:
            return
        self._ultimo_check = time.monotonic()
        try:
            v = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS).version
            if str(v) != self.version:
                self.reload()
        except Exception as e:
            log.warning('No se pudo verificar el alias %s: %s', config.MODEL_ALIAS, e)

    # ------------------------------------------------------------------ helpers
    def _validar(self, sucursal: str, skus):
        if sucursal not in self.sucursales:
            raise NotFound(f'Sucursal desconocida: {sucursal}')
        faltan = [s for s in skus if s not in self.skus]
        if faltan:
            raise NotFound(f'SKU desconocido: {", ".join(faltan)}')

    def _historia(self, fecha: pd.Timestamp, sucursal: str, skus: list[str]) -> pd.DataFrame:
        """Solo lo necesario: filas anteriores a fecha de las series pedidas (luego se toman las 3 últimas)."""
        q = text(f'SELECT "FECHA", "SUCURSAL", "SKU", "TOTAL" FROM {config.TABLE_HISTORY} '
                 f'WHERE "SUCURSAL" = :suc AND "FECHA" < :f')
        h = pd.read_sql(q, self.engine, params={'suc': sucursal, 'f': fecha.to_pydatetime()},
                        parse_dates=['FECHA'])
        h = h[h['SKU'].isin(skus)].sort_values('FECHA')
        return h.groupby('SKU').tail(3)

    def _log(self, filas: list[dict]):
        try:
            pd.DataFrame(filas).to_sql(config.TABLE_PREDICTIONS, self.engine, if_exists='append', index=False)
        except Exception as e:                       # el log nunca debe tirar la predicción
            log.warning('No se pudo registrar la predicción: %s', e)

    # ------------------------------------------------------------------ API pública
    def predict_batch(self, fecha, sucursal: str, items: list[Item], protocolo: str = 'python') -> list[dict]:
        self._refrescar_si_cambio()
        fecha = pd.Timestamp(fecha)
        self._validar(sucursal, [i.sku for i in items])
        t0 = time.perf_counter()
        hist = self._historia(fecha, sucursal, [i.sku for i in items])
        X = pd.concat([build_inference_row(hist, fecha, sucursal, i.sku, i.stock, i.devoluciones,
                                           self.feriados, self.clima) for i in items], ignore_index=True)
        with self._lock:
            unidades = postprocess(self.model.predict(X[FEATURES]))
            version = self.version
        ms = (time.perf_counter() - t0) * 1000

        out = [{'fecha': fecha.date(), 'sucursal': sucursal, 'sku': i.sku, 'unidades': int(u),
                'modelo_version': version} for i, u in zip(items, unidades)]
        ahora = datetime.now(timezone.utc).replace(tzinfo=None)
        self._log([{'TS': ahora, 'PROTOCOLO': protocolo, 'FECHA': fecha, 'SUCURSAL': sucursal, 'SKU': i.sku,
                    'STOCK': i.stock, 'DEVOLUCIONES': i.devoluciones, 'PRED': int(u),
                    'MODELO_VERSION': version, 'MS_MODELO': ms / len(items)}
                   for i, u in zip(items, unidades)])
        return out

    def predict(self, fecha, sucursal: str, sku: str, stock: int = 0, devoluciones: int = 0,
                protocolo: str = 'python') -> dict:
        return self.predict_batch(fecha, sucursal, [Item(sku, stock, devoluciones)], protocolo)[0]


_service: ForecastService | None = None


def get_service() -> ForecastService:
    """Singleton perezoso: el servicio arranca aunque todavía no haya modelo en prod."""
    global _service
    if _service is None:
        _service = ForecastService()
    return _service


def reload_service() -> str:
    global _service
    if _service is None:
        _service = ForecastService()
        return _service.version
    return _service.reload()
