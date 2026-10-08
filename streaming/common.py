"""Piezas comunes del streaming (S4), sin dependencia de Kafka para poder testearlas.

Escenario: cada mensaje es un pedido que una sucursal carga para una fecha (con su stock y devoluciones).
El consumidor pide el pronóstico por gRPC y compara el pedido real contra lo esperado:
si el desvío supera el umbral, emite una alerta de pedido atípico (SOBREPEDIDO / SUBPEDIDO)
para revisarlo antes del despacho. Todas las comparaciones quedan en gout_db (insumo del monitoreo)."""
import json
import os
from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy import create_engine, text

GOUT_DB_URL = os.getenv('GOUT_DB_URL', 'postgresql+psycopg2://airflow:airflow@localhost:5432/gout_db')
TABLE_HISTORY = 'ventas_historicas'
TABLE_ALERTS = 'alertas_desvio'
TABLE_ONLINE = 'evaluacion_online'

UMBRAL_ABS = int(os.getenv('UMBRAL_ABS', '4'))          # unidades
UMBRAL_REL = float(os.getenv('UMBRAL_REL', '0.5'))      # proporción del pronóstico


@dataclass
class Pedido:
    fecha: str
    sucursal: str
    sku: str
    stock: int
    devoluciones: int
    pedido: int                     # TOTAL real cargado por la sucursal

    def key(self) -> bytes:
        return f'{self.sucursal}|{self.sku}'.encode()

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_bytes(cls, b: bytes) -> 'Pedido':
        return cls(**json.loads(b))


def pedidos_a_reproducir(engine, desde: str, hasta: str) -> list[Pedido]:
    """Pedidos reales del período, en orden cronológico (fecha, sucursal, sku)."""
    q = text(f'SELECT "FECHA", "SUCURSAL", "SKU", "STOCK", "DEVOLUCIONES", "TOTAL" FROM {TABLE_HISTORY} '
             f'WHERE "FECHA" >= :d AND "FECHA" <= :h')
    df = pd.read_sql(q, engine, params={'d': pd.Timestamp(desde).to_pydatetime(),
                                        'h': pd.Timestamp(hasta).to_pydatetime()}, parse_dates=['FECHA'])
    df = df.sort_values(['FECHA', 'SUCURSAL', 'SKU'])
    return [Pedido(str(r.FECHA.date()), r.SUCURSAL, r.SKU, int(r.STOCK), int(r.DEVOLUCIONES), int(r.TOTAL))
            for r in df.itertuples()]


def evaluar(p: Pedido, pronostico: int, version: str) -> dict:
    desvio = p.pedido - pronostico
    umbral = max(UMBRAL_ABS, UMBRAL_REL * pronostico)
    tipo = None
    if abs(desvio) >= umbral:
        tipo = 'SOBREPEDIDO' if desvio > 0 else 'SUBPEDIDO'
    return {'FECHA': pd.Timestamp(p.fecha), 'SUCURSAL': p.sucursal, 'SKU': p.sku,
            'STOCK': p.stock, 'DEVOLUCIONES': p.devoluciones, 'PEDIDO': p.pedido,
            'PRONOSTICO': pronostico, 'DESVIO': desvio, 'UMBRAL': umbral,
            'ALERTA': tipo, 'MODELO_VERSION': version}


def guardar(engine, filas: list[dict]):
    if not filas:
        return
    df = pd.DataFrame(filas)
    df.to_sql(TABLE_ONLINE, engine, if_exists='append', index=False)
    alertas = df[df['ALERTA'].notna()]
    if not alertas.empty:
        alertas.to_sql(TABLE_ALERTS, engine, if_exists='append', index=False)


def get_engine():
    return create_engine(GOUT_DB_URL)
