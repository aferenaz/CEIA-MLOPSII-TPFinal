# Gout MLOps — Pronóstico de demanda

TP final de MLOps2 (CEIA–FIUBA). Pronóstico diario de unidades a despachar por sucursal y SKU
para una cadena de pastelería sin TACC, sobre datos anonimizados (sucursales S01–S09, SKUs P001–P020).

## Componentes

| Servicio | Rol | Puerto |
|---|---|---|
| Airflow | ETL, entrenamiento, monitoreo (DAGs `etl_gout`, `train_gout`, `drift_gout`) | 8080 |
| MLflow | Tracking y model registry (alias `prod`) | 5005 |
| MinIO | Data lake (bucket `data`) y artefactos (bucket `mlflow`) | 9000 / 9001 |
| PostgreSQL | Metadatos de Airflow y MLflow + `gout_db` (features, predicciones, alertas, drift) | 5432 |
| API | S1 REST (`/predict`, `/predict/batch`) + S2 GraphQL (`/graphql`) | 8800 |
| gRPC | S3 `Forecast.Predict` / `PredictBatch` | 50051 |
| Kafka | S4 streaming: ventas → consumidor → alertas de desvío | 29092 |
| Monitoreo (DAG `drift_gout`) | S7 drift de datos (PSI vs. referencia estacional) y degradación online → reentrenamiento | — |

## Estructura

```
├── docker-compose.yaml
├── .env.example            # copiar a .env
├── data/                   # pedidos_anon.csv, resumen_anon.csv (se suben a MinIO al iniciar)
├── notebooks/              # anonimizar_gout.ipynb
├── src/gout_common/        # código compartido: config, features, carga de modelo
├── airflow/dags/           # etl_gout, train_gout, drift_gout
├── api/app/                # FastAPI + Strawberry (GraphQL)
├── grpc_service/           # .proto + servidor
├── streaming/              # productor y consumidor Kafka
└── dockerfiles/            # un Dockerfile por servicio
```

La ingeniería de features vive en `src/gout_common/features.py` y la usan tanto el entrenamiento
como el serving, para evitar diferencias entre ambos.

## Uso

### Sin Docker local: GitHub Codespaces
El repo trae `.devcontainer/` (máquina de 4 núcleos / 16 GB, Docker incluido, `.env` creado solo).
En GitHub: **Code → Codespaces → Create codespace on main**, y en la terminal:

```bash
docker compose --profile all up -d --build
docker compose ps          # esperar a que todo esté healthy
```
Los puertos se abren desde la pestaña **Ports**.

### Con Docker local

```bash
cp .env.example .env              # en Linux/Mac: AIRFLOW_UID=$(id -u)
# copiar data/pedidos_anon.csv y data/resumen_anon.csv
docker compose --profile all up -d --build
```

1. Airflow (http://localhost:8080, airflow/airflow): correr `etl_gout` y luego `train_gout`.
2. MLflow (http://localhost:5005): verificar el modelo con alias `prod`.
3. API: http://localhost:8800/docs (REST) y http://localhost:8800/graphql.
4. Streaming (con modelo en prod): `docker compose --profile streaming up -d`.

Perfiles disponibles: `all`, `airflow`, `mlflow`, `api`, `grpc`, `streaming`.
