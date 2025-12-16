import pandas as pd
import redis
import json
import glob
import os
import pyarrow.parquet as pq
import numpy as np

DATA_PATH = "/home/quangtran/Documents/KLTN2/sampled_metadata"
redis_host = "127.0.0.1"
redis_port = 6379

REDIS_KEY_PREFIX = "item:"


r = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=0,
    decode_responses=True
)


def load_parquet_from_folder(folder_path):
    """Đọc tất cả file Parquet trong một folder."""
    folder_path = os.path.abspath(folder_path)
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    tables = [pq.read_table(f) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)


def upload_to_redis(df: pd.DataFrame):
    pipe = r.pipeline(transaction=False)
    count = 0

    for _, row in df.iterrows():
        item_id = row["parent_asin"]

        # Convert np.ndarray → list 
        categories_list = row["categories"].tolist() if isinstance(row["categories"], np.ndarray) else row["categories"]

        metadata = {
            "parent_asin": row["parent_asin"],
            "main_category": row["main_category"],
            "categories": categories_list,
            "price": row["price"],
        }

        key = f"{REDIS_KEY_PREFIX}{item_id}"
        pipe.set(key, json.dumps(metadata))
        count += 1

    pipe.execute()
    print(f"Đã upload {count} items vào Redis.")


def run():
    print(" Loading parquet files...")
    df = load_parquet_from_folder(DATA_PATH)
    df = df[["parent_asin", "main_category", "categories", "price"]]

    print("Uploading to Redis...")
    upload_to_redis(df)


if __name__ == "__main__":
    run()
