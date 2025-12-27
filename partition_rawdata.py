# this file paritition raw data to simulate schedule update by partition

from pyspark.sql import SparkSession
from dotenv import load_dotenv
from pyspark.sql.functions import col, from_unixtime, date_format

import os

load_dotenv() 

spark = SparkSession.builder.appName("PartitionData").config(
        "spark.jars.packages",
        "org.apache.hadoop:hadoop-aws:3.4.1,com.amazonaws:aws-java-sdk-bundle:1.12.262"
    )\
        .config("spark.driver.memory", "10g")  \
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT_URL")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

df = spark.read.json("s3a://rawdata/raw_review.jsonl")

df = df.withColumn("ts", (col("timestamp")/1000).cast("long"))
df = df.withColumn("ts", from_unixtime(col("ts")))

df = df.withColumn("year_month", date_format(col("ts"), "yyyy-MM"))

df.write.mode("overwrite").partitionBy("year_month").parquet("s3a://partitiondata/output")