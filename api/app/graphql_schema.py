"""S2 — GraphQL. Pronóstico de una sucursal para una fecha, por SKU (misma lógica que REST)."""
from datetime import date

import strawberry
from graphql import GraphQLError

from gout_common.serving import Item, NotFound, get_service


@strawberry.input
class ItemInput:
    sku: str
    stock: int = 0
    devoluciones: int = 0


@strawberry.type
class Pronostico:
    sku: str
    unidades: int


@strawberry.type
class PronosticoSucursal:
    sucursal: str
    fecha: date
    modelo_version: str
    pronosticos: list[Pronostico]


@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> str:
        return "ok"

    @strawberry.field
    def modelo_version(self) -> str:
        return get_service().version

    @strawberry.field
    def pronostico(self, sucursal: str, fecha: date, items: list[ItemInput]) -> PronosticoSucursal:
        try:
            preds = get_service().predict_batch(
                fecha, sucursal, [Item(i.sku, i.stock, i.devoluciones) for i in items], protocolo="graphql")
        except NotFound as e:
            raise GraphQLError(str(e))
        return PronosticoSucursal(
            sucursal=sucursal, fecha=fecha, modelo_version=preds[0]["modelo_version"],
            pronosticos=[Pronostico(sku=p["sku"], unidades=p["unidades"]) for p in preds])


schema = strawberry.Schema(query=Query)
