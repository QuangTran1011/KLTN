import pandas as pd
import pyarrow.parquet as pq
import gcsfs
from google.cloud import bigquery
# import os
# import glob

PROJECT_ID = "turing-thought-481409-d8"
DATASET_ID = "kltn"
TABLE_ID = "traindatareviewtest"

TRAIN_PATH = "gs://kltn--data/train_df20/*.parquet"
TEST_PATH = "gs://kltn--data/val_df20/*.parquet"

def load_parquet_from_gcs(path_pattern):
    """Đọc tất cả file Parquet matching pattern từ GCS"""
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(path_pattern)
    tables = [pq.read_table(fs.open(f, "rb")) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

# def load_parquet_from_folder(folder_path):            
#     """Đọc tất cả file Parquet trong một folder local"""
#     folder_path = os.path.abspath(folder_path)
#     files = glob.glob(os.path.join(folder_path, "*.parquet"))
#     tables = [pq.read_table(f) for f in files]
#     return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

def parse_dt(df, cols=["timestamp"]):
    return df.assign(
        **{
            col: lambda df: pd.to_datetime(df[col].astype(int), unit="ms", utc=True)
            for col in cols
        }
    )

def concat_and_upload():
    print("📥 Đang đọc dữ liệu từ GCS...")
    df_train = load_parquet_from_gcs(TRAIN_PATH)
    df_test = load_parquet_from_gcs(TEST_PATH)

    print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")

    # Gộp 2 tập
    df_all = pd.concat([df_train, df_test], ignore_index=True).pipe(parse_dt)

    # Xóa cột images nếu tồn tại
    if "images" in df_all.columns:
        df_all = df_all.drop(columns=["images"])
        print("🗑️ Đã xóa cột 'images' trước khi upload")

    print(f"Tổng cộng: {len(df_all)} dòng sau khi concat")

    # Upload lên BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  
        autodetect=True
    )

    job = client.load_table_from_dataframe(df_all, table_ref, job_config=job_config)
    job.result()  # Chờ job hoàn tất

    print(f"Đã upload {len(df_all)} dòng lên bảng {DATASET_ID}.{TABLE_ID}")

if __name__ == "__main__":
    concat_and_upload()
