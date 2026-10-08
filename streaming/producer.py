"""S4 — Productor. Re-emite fecha a fecha los pedidos reales del período elegido al tópico de Kafka,
simulando la carga diaria de pedidos de las sucursales. Clave = sucursal|sku (orden por serie)."""
import logging
import os
import time
from itertools import groupby

from confluent_kafka import Producer

from streaming.common import get_engine, pedidos_a_reproducir

log = logging.getLogger('producer')

TOPIC = os.getenv('KAFKA_TOPIC_VENTAS', 'ventas_gout')
BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:29092')
DESDE = os.getenv('REPLAY_DESDE', '2025-06-01')
HASTA = os.getenv('REPLAY_HASTA', '2025-06-30')
PAUSA_POR_FECHA_S = float(os.getenv('REPLAY_PAUSA_S', '1'))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    pedidos = pedidos_a_reproducir(get_engine(), DESDE, HASTA)
    log.info('Reproduciendo %d pedidos (%s → %s) en %s', len(pedidos), DESDE, HASTA, TOPIC)
    prod = Producer({'bootstrap.servers': BOOTSTRAP, 'linger.ms': 5})
    for fecha, grupo in groupby(pedidos, key=lambda p: p.fecha):
        n = 0
        for p in grupo:
            prod.produce(TOPIC, key=p.key(), value=p.to_bytes())
            n += 1
        prod.flush()
        log.info('%s: %d pedidos enviados', fecha, n)
        time.sleep(PAUSA_POR_FECHA_S)
    log.info('Fin de la reproducción')


if __name__ == '__main__':
    main()
