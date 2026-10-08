"""S1 — REST (FastAPI) + S2 — GraphQL (montado en /graphql). La lógica está en gout_common.serving."""
from fastapi import FastAPI, HTTPException
from strawberry.fastapi import GraphQLRouter

from app.graphql_schema import schema
from app.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse
from gout_common.serving import Item, NotFound, get_service, reload_service

app = FastAPI(title="Gout — Pronóstico de demanda", version="0.2.0")
app.include_router(GraphQLRouter(schema), prefix="/graphql")


def _servicio():
    try:
        return get_service()
    except Exception as e:  # sin modelo en prod o sin datos
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {e}")


@app.get("/health")
def health():
    try:
        return {"status": "ok", "modelo_version": get_service().version}
    except Exception as e:
        return {"status": "sin_modelo", "detalle": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    s = _servicio()
    try:
        return s.predict(req.fecha, req.sucursal, req.sku, req.stock, req.devoluciones, protocolo="rest")
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    s = _servicio()
    try:
        preds = s.predict_batch(req.fecha, req.sucursal,
                                [Item(i.sku, i.stock, i.devoluciones) for i in req.items], protocolo="rest")
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"fecha": req.fecha, "sucursal": req.sucursal, "predicciones": preds}


@app.post("/model/reload")
def model_reload():
    """Lo llama train_gout cuando promueve un modelo nuevo a prod."""
    try:
        return {"modelo_version": reload_service()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
