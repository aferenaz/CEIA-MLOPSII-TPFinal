"""Entrenamiento: features de gout_db -> XGBoost (pipeline de la memoria) -> MLflow.
Champion/challenger: el nuevo modelo pasa a alias 'prod' solo si mejora el RMSE en test.
La lógica vive en gout_common.train (testeable sin Airflow)."""
import pendulum
from airflow.decorators import dag, task


@dag(dag_id="train_gout", schedule=None, start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, tags=["gout", "train"])
def train_gout():
    @task
    def entrenar():
        from gout_common.etl import get_engine
        from gout_common.train import run_train
        return run_train(get_engine())

    @task
    def evaluar_y_promover(resultado: dict):
        from gout_common.etl import get_engine
        from gout_common.train import run_promote
        return run_promote(get_engine(), resultado["version"])

    @task.short_circuit
    def hubo_promocion(decision: dict):
        return decision["promovido"]

    @task
    def recargar_api():
        from gout_common.train import notify_api_reload
        notify_api_reload()

    decision = evaluar_y_promover(entrenar())
    hubo_promocion(decision) >> recargar_api()


train_gout()
