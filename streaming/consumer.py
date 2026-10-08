"""S4 — Consumidor. Por cada pedido pide el pronóstico al servicio gRPC (S3), calcula el desvío
y, si supera el umbral, publica una alerta en el tópico de alertas. Todo queda en gout_db."""
import json
import logging
import os

import grpc
from confluent_kafka import Consumer, Producer

from streaming import forecast_pb2 as pb
from streaming import forecast_pb2_grpc as pb_grpc
from streaming.common import Pedido, evaluar, get_engine, guardar

log = logging.getLogger('consumer')

TOPIC_IN = os.getenv('KAFKA_TOPIC_VENTAS', 'ventas_gout')
TOPIC_OUT = os.getenv('KAFKA_TOPIC_ALERTAS', 'alertas_desvio')
BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:29092')
GRPC_TARGET = os.getenv('GRPC_TARGET', 'localhost:50051')
GUARDAR_CADA = int(os.getenv('GUARDAR_CADA', '50'))
MAX_INACTIVO_S = float(os.getenv('MAX_INACTIVO_S', '0'))   # 0 = escuchar siempre


def procesar(p: Pedido, stub) -> dict:
    r = stub.Predict(pb.PredictRequest(fecha=p.fecha, sucursal=p.sucursal, sku=p.sku,
                                       stock=p.stock, devoluciones=p.devoluciones), timeout=10)
    return evaluar(p, r.unidades, r.modelo_version)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    engine = get_engine()
    stub = pb_grpc.ForecastStub(grpc.insecure_channel(GRPC_TARGET))
    cons = Consumer({'bootstrap.servers': BOOTSTRAP, 'group.id': 'gout-desvios',
                     'auto.offset.reset': 'earliest', 'enable.auto.commit': False})
    prod = Producer({'bootstrap.servers': BOOTSTRAP})
    cons.subscribe([TOPIC_IN])
    log.info("Escuchando '%s' → alertas en '%s'", TOPIC_IN, TOPIC_OUT)

    buffer, inactivo, total, alertas = [], 0.0, 0, 0
    try:
        while True:
            msg = cons.poll(1.0)
            if msg is None:
                inactivo += 1.0
                if buffer:
                    guardar(engine, buffer); cons.commit(asynchronous=False); buffer = []
                if MAX_INACTIVO_S and inactivo >= MAX_INACTIVO_S:
                    break
                continue
            if msg.error():
                log.warning('Error Kafka: %s', msg.error()); continue
            inactivo = 0.0
            p = Pedido.from_bytes(msg.value())
            try:
                fila = procesar(p, stub)
            except grpc.RpcError as e:
                log.warning('gRPC %s para %s: %s', e.code(), p.key(), e.details()); continue
            buffer.append(fila); total += 1
            if fila['ALERTA']:
                alertas += 1
                prod.produce(TOPIC_OUT, key=p.key(),
                             value=json.dumps({**fila, 'FECHA': p.fecha}, default=str).encode())
                log.info('ALERTA %s %s %s: pedido %d vs pronóstico %d',
                         fila['ALERTA'], p.sucursal, p.sku, p.pedido, fila['PRONOSTICO'])
            if len(buffer) >= GUARDAR_CADA:
                guardar(engine, buffer); prod.flush(); cons.commit(asynchronous=False); buffer = []
    finally:
        guardar(engine, buffer); prod.flush(); cons.close()
        log.info('Procesados %d pedidos, %d alertas', total, alertas)


if __name__ == '__main__':
    main()
