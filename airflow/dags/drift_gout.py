"""S7 — Monitoreo semanal: drift de datos (PSI vs. referencia estacional) + degradación online.
Guarda el reporte HTML en MinIO y las métricas en gout_db. Si corresponde, dispara
etl_gout -> train_gout (el champion/challenger evita promover un modelo peor).
La lógica vive en gout_common.drift (testeable sin Airflow)."""
import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(dag_id="drift_gout", schedule="@weekly", start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, tags=["gout", "monitoring"])
def drift_gout():
    @task
    def calcular_drift():
        from gout_common.drift import run_drift
        from gout_common.etl import get_engine
        res = run_drift(get_engine())
        return {k: v for k, v in res.items() if k not in ("tabla", "ts")}

    @task.branch
    def decidir(res: dict):
        return "reentrenar" if res["reentrenar"] else "sin_accion"

    @task
    def sin_accion():
        pass

    reentrenar = TriggerDagRunOperator(task_id="reentrenar", trigger_dag_id="etl_gout",
                                       conf={"disparar_entrenamiento": True})

    decidir(calcular_drift()) >> [reentrenar, sin_accion()]


drift_gout()
