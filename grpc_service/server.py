"""S3 — servidor gRPC. Misma lógica que REST y GraphQL (gout_common.serving), otro protocolo."""
import logging
import os
from concurrent import futures

import grpc

from gout_common.serving import Item, NotFound, get_service, reload_service
from grpc_service import forecast_pb2 as pb
from grpc_service import forecast_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)


def _resp(p: dict) -> pb.PredictResponse:
    return pb.PredictResponse(fecha=str(p["fecha"]), sucursal=p["sucursal"], sku=p["sku"],
                              unidades=p["unidades"], modelo_version=p["modelo_version"])


class ForecastServicer(pb_grpc.ForecastServicer):
    def _servicio(self, context):
        try:
            return get_service()
        except Exception as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, f"Modelo no disponible: {e}")

    def Predict(self, request, context):
        s = self._servicio(context)
        try:
            return _resp(s.predict(request.fecha, request.sucursal, request.sku,
                                   request.stock, request.devoluciones, protocolo="grpc"))
        except NotFound as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    def PredictBatch(self, request, context):
        s = self._servicio(context)
        if not request.items:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "items vacío")
        try:
            preds = s.predict_batch(request.fecha, request.sucursal,
                                    [Item(i.sku, i.stock, i.devoluciones) for i in request.items],
                                    protocolo="grpc")
        except NotFound as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return pb.BatchResponse(predicciones=[_resp(p) for p in preds])

    def Reload(self, request, context):
        try:
            return pb.ReloadResponse(modelo_version=reload_service())
        except Exception as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))


def serve():
    logging.basicConfig(level=logging.INFO)
    port = os.getenv("GRPC_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_ForecastServicer_to_server(ForecastServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("gRPC escuchando en :%s", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
