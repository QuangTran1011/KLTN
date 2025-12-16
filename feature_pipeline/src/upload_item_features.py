import pandas as pd
import pyarrow.parquet as pq
from google.cloud import bigquery
import glob
import os

PROJECT_ID = "big-potential-478810-t1"
DATASET_ID = "kltn"
TABLE_ID = "item_meta_data"

DATA_PATH = "/home/quangtran/Documents/KLTN2/sampled_metadata"

def load_parquet_from_folder(folder_path):
    """Đọc tất cả file Parquet trong một folder local"""
    folder_path = os.path.abspath(folder_path)
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    tables = [pq.read_table(f) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

def upload():
    print("📥 Đang đọc dữ liệu từ folder...")
    full_df = load_parquet_from_folder(DATA_PATH)
    full_df["image"] = full_df["images"].apply(
    lambda x: next((i for i in x.get("large", []) if i is not None), None)
)
    full_df = full_df[['parent_asin', 'main_category', 'categories', 'price', 'image', 'title']]
    # 🚀 Upload lên BigQuery

    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  
        autodetect=True
    )

    job = client.load_table_from_dataframe(full_df, table_ref, job_config=job_config)
    job.result()  

    print(f"✅ Đã upload {len(full_df)} dòng lên bảng {DATASET_ID}.{TABLE_ID}")

if __name__ == "__main__":
    upload()
