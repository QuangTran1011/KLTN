import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from loguru import logger
import numpy as np
import mlflow
import sys
import json
sys.path.insert(0, "../..")

from src.id_mapper import IDMapper

mlflow.set_tracking_uri("http://34.69.242.168:8080/")

mlf_client = mlflow.MlflowClient()
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/item2vec@newly"
)

tag_df = pd.read_parquet('/home/quangtran/Documents/KLTN2/tag_embeded.parquet')
embedding_dim = len(tag_df.iloc[0]['tag_embedding'])

def fix_embedding(emb, target_dim=embedding_dim):
    if emb is None:
        return np.zeros(target_dim, dtype=np.float32)
    return np.array(emb, dtype=np.float32).flatten()

tag_df["tag_embedding"] = tag_df["tag_embedding"].apply(fix_embedding)

collection_name = "item_tag_embedding"

id_mapping_raw = model.unwrap_python_model().id_mapping

with open("/home/quangtran/Documents/KLTN2/data/idm.json", "w") as f:
    json.dump(id_mapping_raw, f)

id_mapping = IDMapper().load("/home/quangtran/Documents/KLTN2/data/idm.json")

ann_index = QdrantClient("localhost:6333", timeout=120.0, prefer_grpc=True)

collection_exists = ann_index.collection_exists(collection_name)
if collection_exists:
    logger.info(f"Deleting existing Qdrant collection {collection_name}...")
    ann_index.delete_collection(collection_name)

create_collection_result = ann_index.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=id_mapping.get_item_index(row.parent_asin),
        vector=row.tag_embedding.tolist(),
        payload={}
    )
    for row in tag_df.itertuples(index=False)
]

ann_index.upsert(
    collection_name=collection_name,
    points=points,
)


