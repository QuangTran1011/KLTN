import torch
import sys
sys.path.insert(0, "../..")
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from loguru import logger
import mlflow

from src.id_mapper import IDMapper

mlflow.set_tracking_uri("http://34.69.242.168:8080/")

collection_name = "item2vec"

mlf_client = mlflow.MlflowClient()
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/item2vec@newly"
)
skipgram_model = model.unwrap_python_model().model
embedding_0 = skipgram_model.embeddings(torch.tensor(0))
embedding_dim = embedding_0.size()[0]

id_mapping = model.unwrap_python_model().id_mapping
all_items = list(id_mapping['item_to_index'].values())

embeddings = skipgram_model.embeddings(torch.tensor(all_items)).detach().numpy()
ann_index = QdrantClient("localhost:6333", timeout=120.0, prefer_grpc=True)

collection_exists = ann_index.collection_exists(collection_name)
if collection_exists:
    logger.info(f"Deleting existing Qdrant collection {collection_name}...")
    ann_index.delete_collection(collection_name)

create_collection_result = ann_index.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

upsert_result = ann_index.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(id=idx, vector=vector.tolist(), payload={})
        for idx, vector in enumerate(embeddings)
    ],
)
assert str(upsert_result.status) == "completed"
