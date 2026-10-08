from datetime import date
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    fecha: date
    sucursal: str = Field(examples=["S01"])
    sku: str = Field(examples=["P001"])
    stock: int = Field(0, ge=0, description="Stock informado por la sucursal al pedir")
    devoluciones: int = Field(0, ge=0, description="Devoluciones informadas por la sucursal al pedir")


class PredictResponse(BaseModel):
    fecha: date
    sucursal: str
    sku: str
    unidades: int
    modelo_version: str


class BatchItem(BaseModel):
    sku: str = Field(examples=["P001"])
    stock: int = Field(0, ge=0)
    devoluciones: int = Field(0, ge=0)


class BatchRequest(BaseModel):
    fecha: date
    sucursal: str = Field(examples=["S01"])
    items: list[BatchItem] = Field(min_length=1)


class BatchResponse(BaseModel):
    fecha: date
    sucursal: str
    predicciones: list[PredictResponse]
