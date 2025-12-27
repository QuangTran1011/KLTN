import os
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from google.cloud import bigquery

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
dataset = os.getenv("BIGQUERY_DATASET")
table_name = "traindatareviewtest"

client = bigquery.Client(project=project_id)

def get_curr_oltp_max_timestamp():
    query = f"SELECT MAX(timestamp) AS max_timestamp FROM `{project_id}.{dataset}.{table_name}`"
    df = client.query(query).to_dataframe()
    max_timestamp = df["max_timestamp"].iloc[0]
    if pd.notnull(max_timestamp):
        max_timestamp = pd.to_datetime(max_timestamp).isoformat()
    return max_timestamp

logger.info(f"Max timestamp in BigQuery: <ts>{get_curr_oltp_max_timestamp()}</ts>")
