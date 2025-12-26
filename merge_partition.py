# this file paritition raw data to simulate schedule update by partition


import pandas as pd
import io
import os
import re
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

client = Minio(
    os.getenv("S3_ENDPOINT_URL").replace("http://", ""),
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    secure=False,
)

bucket = "partitiondata"
input_prefix = "output"
output_prefix = "merged_partition"

# folder local để lưu parquet
local_output_dir = "./parquet_output"
os.makedirs(local_output_dir, exist_ok=True)

objects = sorted(
    [o.object_name for o in client.list_objects(bucket, prefix=input_prefix, recursive=True)]
)

buffer = []
count = 0
file_idx = 1
start_month = None

for obj_name in objects:
    resp = client.get_object(bucket, obj_name)
    if obj_name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(resp.read()))
    else:
        resp.close()
        resp.release_conn()
        continue
    resp.close()
    resp.release_conn()

    m = re.search(r"year_month=(\d{4}-\d{2})", obj_name)
    if m:
        this_month = m.group(1)
    else:
        continue

    if start_month is None:
        start_month = this_month

    buffer.append(df)
    count += len(df)

    if count >= 100000:
        merged_df = pd.concat(buffer, ignore_index=True)
        end_month = this_month

        out_name = f"{output_prefix}_{start_month}_{end_month}_id{file_idx}.parquet"
        out_path = os.path.join(local_output_dir, out_name)

        merged_df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}, size={len(merged_df)}")

        # reset
        buffer = []
        count = 0
        file_idx += 1
        start_month = None

# flush phần dư cuối cùng
if buffer:
    merged_df = pd.concat(buffer, ignore_index=True)
    end_month = this_month
    out_name = f"{output_prefix}_{start_month}_{end_month}_id{file_idx}.parquet"
    out_path = os.path.join(local_output_dir, out_name)
    merged_df.to_parquet(out_path, index=False)
    print(f"Saved {out_path}, size={len(merged_df)}")
