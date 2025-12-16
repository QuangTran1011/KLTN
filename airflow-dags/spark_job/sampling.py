from loguru import logger
from pydantic import BaseModel
from pyspark.sql import SparkSession
from modules.sample import SparkRichnessDataSampler
from google.cloud import storage
from datetime import datetime
import re


class Args(BaseModel):
    random_seed: int = 41
    user_col: str = "user_id"
    item_col: str = "parent_asin"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"
    bucket: str = "kltn--data"
    prefix: str = "partitiondata/"
    min_user_interactions: int = 2
    min_item_interactions: int = 3
    window_size: int = 5

args = Args()


spark = (
    SparkSession.builder
        .appName("job_sampling")
        .config("spark.jars.packages", 
                "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.20")
        .getOrCreate()
)

spark.conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
spark.sparkContext.setCheckpointDir("gs://kltn--data/spark_checkpoints")

pattern = r"recsys_data_upto_(\d{4})_(\d{2})_(\d{2})\.parquet"

client = storage.Client()
bucket = client.bucket(args.bucket)
blobs = bucket.list_blobs(prefix=args.prefix)

parquet_files = [b.name for b in blobs if re.match(pattern, b.name.split("/")[-1])]

def extract_date(filename):
    m = re.match(pattern, filename.split("/")[-1])
    y, mth, d = m.groups()
    return datetime(int(y), int(mth), int(d))

parquet_files_sorted = sorted(parquet_files, key=extract_date, reverse=True)

files_to_read = parquet_files_sorted[:args.window_size]

files_to_read_gcs = [f"gs://{args.bucket}/{f}" for f in files_to_read]
logger.info(f"Files to read: {files_to_read_gcs}")


df = spark.read.parquet(*files_to_read_gcs)

sampler = SparkRichnessDataSampler(
    user_col=args.user_col,
    item_col=args.item_col,
    ts_col=args.timestamp_col,
    random_seed=args.random_seed,
    min_item_interactions=args.min_item_interactions,
    min_user_interactions=args.min_user_interactions,
    debug=True
)

train_df, val_df, cold_users, cold_items = sampler.sample(df)


train_df.write.mode("overwrite").parquet(f"gs://{args.bucket}/train_df20")
val_df.write.mode("overwrite").parquet(f"gs://{args.bucket}/val_df20")
cold_users.write.mode("overwrite").parquet(f"gs://{args.bucket}/coldstart_users20")
cold_items.write.mode("overwrite").parquet(f"gs://{args.bucket}/coldstart_items20")

