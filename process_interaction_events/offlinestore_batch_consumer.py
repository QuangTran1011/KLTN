import json
from datetime import datetime
from confluent_kafka import Consumer
from google.cloud import bigquery
import os

KAFKA_CONF = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP"),
    "group.id": "offline-batch-consumer",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

TOPIC = os.getenv("KAFKA_TOPIC", "user-interactions")

BQ_PROJECT = os.getenv("PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("RAW_TABLE_ID")

def main():
    consumer = Consumer(KAFKA_CONF)
    consumer.subscribe([TOPIC])

    # ---- wait assignment ----
    while True:
        consumer.poll(0.1)
        assignment = consumer.assignment()
        if assignment:
            break

    # ---- snapshot end offsets ----
    end_offsets = {}
    for tp in assignment:
        low, high = consumer.get_watermark_offsets(tp)
        end_offsets[tp.partition] = high - 1

    rows = []
    done = set()

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            if len(done) == len(end_offsets):
                break
            continue

        if msg.error():
            continue

        partition = msg.partition()
        offset = msg.offset()

        data = json.loads(msg.value())

        rows.append(data)

        if offset >= end_offsets[partition]:
            done.add(partition)

        if len(done) == len(end_offsets):
            break

    consumer.close()

    if not rows:
        print("No data")
        return

    # ---- write to BigQuery ----
    client = bigquery.Client(project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    errors = client.insert_rows_json(table_id, rows)

    if errors:
        raise RuntimeError(errors)

    print(f"Inserted {len(rows)} rows to BigQuery")

if __name__ == "__main__":
    main()
