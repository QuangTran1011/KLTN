import json
from datetime import datetime
from confluent_kafka import Consumer, TopicPartition
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
import os
import time

KAFKA_CONF = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP"),
    "group.id": "offline-batch-consumer",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

TOPIC = os.getenv("KAFKA_TOPIC", "user-interactions")
BUCKET = os.getenv("GCS_BUCKET", "kltn--data")
GCS_PREFIX = os.getenv("GCS_PREFIX", "partitiondata")

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
        end_offsets[tp.partition] = high - 1  # last readable offset

    print("End offsets snapshot:", end_offsets)

    rows = []
    done_partitions = set()

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            if len(done_partitions) == len(end_offsets):
                break
            continue

        if msg.error():
            continue

        partition = msg.partition()
        offset = msg.offset()

        rows.append(json.loads(msg.value()))

        if offset >= end_offsets[partition]:
            done_partitions.add(partition)

        if len(done_partitions) == len(end_offsets):
            break

    consumer.close()

    if not rows:
        print("No data")
        return

    # ---- write parquet ----
    table = pa.Table.from_pylist(rows)
    today = datetime.utcnow().strftime("%Y_%m_%d_%H%M%S")
    filename = f"recsys_data_upto_{today}.parquet"
    local_path = f"/tmp/{filename}"

    pq.write_table(table, local_path)

    # ---- upload to GCS ----
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(f"{GCS_PREFIX}/{filename}")
    blob.upload_from_filename(local_path)

    print(f"Uploaded {filename} to GCS")

if __name__ == "__main__":
    main()
