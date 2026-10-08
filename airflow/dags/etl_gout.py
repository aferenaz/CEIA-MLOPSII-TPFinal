"""ETL: CSV anonimizado en MinIO -> historia validada -> variables externas -> features + split temporal.
Todo queda en gout_db. La lógica vive en gout_common.etl (testeable sin Airflow)."""
import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(dag_id="etl_gout", schedule=None, start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, tags=["gout", "etl"],
     params={"disparar_entrenamiento": False})
def etl_gout():
    @task
    def extraer():
        from gout_common.etl import get_engine, run_extract
        return run_extract(get_engine())

    @task
    def externas():
        from gout_common.etl import get_engine, run_externals
        return run_externals(get_engine())

    @task
    def features():
        from gout_common.etl import get_engine, run_features
        return run_features(get_engine())

    @task.short_circuit
    def entrenar_despues(**context):
        return bool(context["params"]["disparar_entrenamiento"])

    entrenar = TriggerDagRunOperator(task_id="disparar_train_gout", trigger_dag_id="train_gout")

    extraer() >> externas() >> features() >> entrenar_despues() >> entrenar


etl_gout()
