import torch
import sys
sys.path.insert(0, "../..")
from qdrant_client import QdrantClient
from loguru import logger
import time 
from src.id_mapper import IDMapper
from tqdm.auto import tqdm
import redis
import json
import os
import mlflow

qdrant_url = "localhost:6333"
# qdrant_url = "qdrant.serving.svc.cluster.local:6333"
collection_name = "item2vec"
top_K = 100
redis_key_prefix = "output:i2i:"
redis_host = "localhost"
redis_port = 6378
# redis_host = "redis-master.serving.svc.cluster.local"
# redis_port = 6379

mlflow.set_tracking_uri("http://34.69.242.168:8080/")

mlf_client = mlflow.MlflowClient()
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/item2vec@newly"
)
skipgram_model = model.unwrap_python_model().model


# Qdrant client
ann_index = QdrantClient(url=qdrant_url, timeout=120.0, prefer_grpc=True)
if not ann_index.collection_exists(collection_name):
    raise Exception(f"Required Qdrant collection {collection_name} does not exist")

# Redis client
r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
assert r.ping(), f"Redis at {redis_host}:{redis_port} is not running"

# ID mapping
id_mapping = IDMapper().load("/home/quangtran/Documents/KLTN2/data/idm.json")
all_items = list(id_mapping.item_to_index.values())

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
    t0 = time.time()
    scores = skipgram_model(
        torch.tensor([idx] * len(neighbors)), 
        torch.tensor(neighbors)
    ).detach().numpy().astype(float)
    t1 = time.time()
    
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
