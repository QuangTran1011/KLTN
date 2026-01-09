from loguru import logger
from pydantic import BaseModel
from pyspark.sql import SparkSession
import os

class Args(BaseModel):
    item_col: str = "parent_asin"
    bucket: str = os.getenv("BUCKET")
    prefix: str = "train_df20"

args = Args()

spark = (
    SparkSession.builder
        .appName("job_metadata")
        .config("spark.jars.packages", 
                "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.20")
        .getOrCreate()
)

spark.conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
spark.sparkContext.setCheckpointDir("gs://kltn--data/spark_checkpoints")

df = spark.read.parquet(f"gs://{args.bucket}/{args.prefix}/")
metadata_raw_df = spark.read.parquet(f"gs://{args.bucket}/raw_item_metadata/")

sampled_metadata_df = metadata_raw_df.join(
    df.select("parent_asin").distinct(),
    on="parent_asin",
    how="semi"
)

sampled_metadata_df.write.mode("overwrite").parquet("gs://kltn--data/sampled_metadata/")

from google.cloud import storage
from datetime import datetime

def upload_flag(bucket_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    today = datetime.utcnow().strftime("%Y_%m_%d")
    blob = bucket.blob(f"_flags/training_ready_{today}.txt")
    blob.upload_from_string("done")

upload_flag(args.bucket)