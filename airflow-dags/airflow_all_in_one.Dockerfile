FROM apache/spark:4.1.0-preview4-scala2.13-java21-python3-ubuntu
USER root
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY spark_job/ /app/spark_job/
COPY modules/ /app/modules/
COPY dbt/ /app/dbt_project/
COPY feature_repo/ /app/feature_repo/

# Download GCS connector jar trực tiếp vào thư mục Spark jars
RUN wget -P /opt/spark/jars \
    https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.20/gcs-connector-hadoop3-2.2.20-shaded.jar

CMD ["/bin/bash"]