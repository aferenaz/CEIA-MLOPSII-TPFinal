FROM apache/airflow:2.10.5-python3.11
COPY dockerfiles/airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
COPY --chown=airflow:root src/gout_common /opt/airflow/gout_common_pkg/gout_common
ENV PYTHONPATH="/opt/airflow/gout_common_pkg"
