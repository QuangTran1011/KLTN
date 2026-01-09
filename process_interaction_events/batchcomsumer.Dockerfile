FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip install confluent-kafka pyarrow google-cloud-storage google-cloud-bigquery

COPY offlinestore_batch_consumer.py gcs_batch_consumer.py ./

CMD ["python", "consumer.py"]
