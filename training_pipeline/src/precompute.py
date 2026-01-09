import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from loguru import logger
import mlflow
from tqdm.auto import tqdm
import redis
import json
import os 

mlflow_url = os.environ.get("mlflow_url")  
qdrant_url = os.environ.get("qdrant_url")        #"qdrant.serving.svc.cluster.local:6333"
collection_name = "item2vec"
top_K = 100
redis_key_prefix = "output:i2i:"
redis_host = os.environ.get("redis_host")  
redis_port = os.environ.get("redis_port")  

mlflow.set_tracking_uri(mlflow_url)
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
ann_index = QdrantClient(qdrant_url, timeout=120.0, prefer_grpc=True)

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




# Redis client
r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
assert r.ping(), f"Redis at {redis_host}:{redis_port} is not running"

# Retrieve all vectors once
records = ann_index.retrieve(collection_name=collection_name, ids=all_items, with_vectors=True)
vectors = {record.id: record.vector for record in records}

# Prepare Redis pipeline
pipe = r.pipeline(transaction=False)

# Iterate item by item
for idx in tqdm(all_items, desc="Computing recommendations"):
    query_vector = vectors[idx]
    
    # Search neighbors
    neighbor_records = ann_index.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_K + 1,
    )
    neighbors = [r.id for r in neighbor_records if r.id != idx]  

    if not neighbors:
        continue

    # Compute SkipGram scores for each neighbor
    scores = skipgram_model(
        torch.tensor([idx] * len(neighbors)), 
        torch.tensor(neighbors)
    ).detach().numpy().astype(float)
    
    # Rerank
    neighbors, scores = zip(*sorted(zip(neighbors, scores), key=lambda x: x[1], reverse=True))
    
    # Map back to original item IDs
    target_item = id_mapping.index_to_item[idx]
    rec_item_ids = [id_mapping.index_to_item[n] for n in neighbors]

    # Save to Redis
    key = redis_key_prefix + target_item
    pipe.set(key, json.dumps({"rec_item_ids": rec_item_ids, "rec_scores": list(scores)}))

# Execute all Redis commands at once
pipe.execute()
