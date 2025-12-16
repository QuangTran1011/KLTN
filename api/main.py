from typing import Any, Dict, List, Optional
import json
from loguru import logger
from fastapi import FastAPI, HTTPException, Query
import os
import redis 
import asyncio
import httpx


app = FastAPI()

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "localhost")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)   #6566

serving_url = f"{MODEL_SERVER_URL}/predict"
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)
redis_output_i2i_key_prefix = "output:i2i:"
redis_feature_recent_items_key_prefix = "feature:user:recent_items:"
redis_output_popular_key = "output:popular"
redis_recent_key_prefix = "feature:user:recent_items:"
REDIS_KEY_PREFIX = "item:"


def get_recommendations_from_redis(
    redis_key: str, count: Optional[int]
) -> Dict[str, Any]:
    rec_data = redis_client.get(redis_key)
    if not rec_data:
        error_message = f"[DEBUG] No recommendations found for key: {redis_key}"
        logger.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]
    return {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores}

@app.get("/recs/i2i")
async def get_recommendations_i2i(
    item_id: str = Query(..., description="ID of the item to get recommendations for"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    redis_key = f"{redis_output_i2i_key_prefix}{item_id}"
    recommendations = get_recommendations_from_redis(redis_key, count)
    return {
        "item_id": item_id,
        "recommendations": recommendations,
    }

@app.get(
    "/recs/u2i/last_item_i2i",
    summary="Get recommendations for users based on their most recent items",
)
async def get_recommendations_u2i_last_item_i2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(f"Getting recent items for user_id: {user_id}")

    # Step 1: Get the recent items for the user
    item_sequences = redis_client.get(f"{redis_recent_key_prefix}{user_id}")
    last_item_id = item_sequences.split('__')[-1]

    logger.debug(f"Most recently interacted item: {last_item_id}")

    # Step 2: Call the i2i endpoint internally to get recommendations for that item
    recommendations = await get_recommendations_i2i(last_item_id, count, debug)

    # Step 3: Format and return the output
    result = {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recommendations["recommendations"],
    }

    return result

@app.get("/recs/u2i/rerank", summary="Get recommendations for users with rerank")
async def get_recommendations_u2i_rerank(
    user_id: str = Query(..., description="ID of the user"),
    top_k_retrieval: Optional[int] = Query(100, description="Number of items to retrieve"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    # Lấy popular + i2i recommendations
    popular_recs, last_item_i2i_recs = await asyncio.gather(
        get_recommendations_popular(count=top_k_retrieval, debug=False),
        get_recommendations_u2i_last_item_i2i(user_id=user_id, count=top_k_retrieval, debug=False),
    )

    #Gộp item và loại bỏ trùng lặp
    all_items = list(set(popular_recs["recommendations"]["rec_item_ids"]).union(
                     set(last_item_i2i_recs["recommendations"]["rec_item_ids"])))
    
    #Lấy item_sequence của user
    # user_feat = await get_user_feature(user_id)
    raw_user_feat = await get_user_feature(user_id)
    user_feat = parse_feature(raw_user_feat)

    item_sequence = user_feat["user_rating_list_10_recent_asin"].split(',')
    # item_sequence = redis_client.get(f"{redis_recent_key_prefix}{user_id}").split('__')

    #Loại bỏ các item đã xuất hiện trong item_sequence
    all_items = [item for item in all_items if item not in item_sequence]

    if not all_items:
        return {"user_id": user_id, "recommendations": {"rec_item_ids": [], "rec_scores": []}}

    #Gọi trực tiếp score prediction v2
    reranked_result = await score_seq_rating_prediction_v2(user_id=user_id, item_ids=all_items, debug=debug)

    #Cắt top-k
    reranked_result["recommendations"]["rec_item_ids"] = reranked_result["recommendations"]["rec_item_ids"][:count]
    reranked_result["recommendations"]["rec_scores"] = reranked_result["recommendations"]["rec_scores"][:count]

    #Thêm item_sequence vào metadata
    reranked_result["features"] = {"item_sequence": item_sequence}

    return reranked_result


@app.get("/recs/popular")
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
    return {"recommendations": recommendations}

@app.post("/score/seq_rating_prediction")
async def score_seq_rating_prediction_v2(
    user_id: str,
    item_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    """
    Predict scores for a user and a list of items using full features (user + item + metadata)
    """
    #Build payload đầy đủ
    payload = await build_payload(user_id, item_ids)
    
    if debug:
        logger.debug(f"[DEBUG] Payload for prediction: {json.dumps(payload, indent=2)}")

    #Gọi model để predict
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                serving_url,
                json=payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
        resp.raise_for_status()
        result = resp.json()
        if debug:
            logger.debug(f"[DEBUG] Model response: {json.dumps(result, indent=2)}")
    except httpx.HTTPError as e:
        error_message = f"Error connecting to model server: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    #Rerank items theo score trả về
    scores = result.get("scores", [])
    returned_item_ids = result.get("item_ids", item_ids)
    if not scores or len(scores) != len(returned_item_ids):
        raise HTTPException(status_code=500, detail="Mismatch sizes between scores and items")

    item_scores = list(zip(returned_item_ids, scores))
    item_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_item_ids, sorted_scores = zip(*item_scores)

    return {
        "user_id": user_id,
        "recommendations": {
            "rec_item_ids": list(sorted_item_ids),
            "rec_scores": list(sorted_scores),
        },
        "metadata": {"rerank": True},
    }
    
@app.get("/recs/get_user_feature")
async def get_user_feature(user_id: str):
    base_url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"

    fresh_payload = {
        "features": [
            "user_rating_stats_fresh:user_rating_list_10_recent_asin",
            "user_rating_stats_fresh:user_rating_list_10_recent_asin_timestamp"
        ],
        "entities": { "user_id": [user_id] }
    }

    async with httpx.AsyncClient() as client:
        fresh_resp = await client.post(base_url, json=fresh_payload)
        fresh_resp.raise_for_status()
        fresh = fresh_resp.json()

    # --- CHECK FRESH RỖNG ---
    # Feature 1
    v1 = fresh["results"][1]["values"][0]

    if v1 is None:
        non_fresh_payload = {
            "features": [
                "user_rating_stats:user_rating_list_10_recent_asin",
                "user_rating_stats:user_rating_list_10_recent_asin_timestamp"
            ],
            "entities": { "user_id": [user_id] }
        }

        async with httpx.AsyncClient() as client:
            nf = await client.post(base_url, json=non_fresh_payload)
            nf.raise_for_status()
            return nf.json()

    return fresh

    
async def get_item_feature(item_id: str):
    url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"

    payload = {
        "features": [
            "parent_asin_rating_stats:parent_asin_rating_cnt_365d",
            "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_365d",
            "parent_asin_rating_stats:parent_asin_rating_cnt_90d",
            "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_90d",
            "parent_asin_rating_stats:parent_asin_rating_cnt_30d",
            "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_30d",
            "parent_asin_rating_stats:parent_asin_rating_cnt_7d",
            "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_7d",
        ],
        "entities": {
            "parent_asin": [item_id]
        }
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
    
def get_item_metadata(item_id: str):
    data = redis_client.get(f"{REDIS_KEY_PREFIX}{item_id}")
    if not data:
        raise ValueError(f"No metadata found for item {item_id}")
    return json.loads(data)

def parse_feature(data):
    feature_names = data["metadata"]["feature_names"]
    results = data["results"]

    parsed = {}

    for i, name in enumerate(feature_names):
        parsed[name] = results[i]["values"][0]

    return parsed

def safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except:
        return default

def safe_str(x, default=""):
    if x is None:
        return default
    return str(x)

# def safe_list_str(x):
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return [safe_str(v) for v in x]
#     return [safe_str(x)]


async def build_payload(user_id: str, item_ids: list):
    #Parse user feature đúng format
    raw_user_feat = await get_user_feature(user_id)
    user_feat = parse_feature(raw_user_feat)

    item_sequence = user_feat["user_rating_list_10_recent_asin"]
    item_sequence_ts = user_feat["user_rating_list_10_recent_asin_timestamp"]

    logger.debug(
        "item_sequence_ts={} | type={}",
        item_sequence_ts,
        type(item_sequence_ts),
    )

    #Lấy item features + metadata từ Redis, parse đúng
    main_category = []
    categories = []
    price = []
    parent_asin_rating_cnt_365d = []
    parent_asin_rating_avg_prev_rating_365d = []
    parent_asin_rating_cnt_90d = []
    parent_asin_rating_avg_prev_rating_90d = []
    parent_asin_rating_cnt_30d = []
    parent_asin_rating_avg_prev_rating_30d = []
    parent_asin_rating_cnt_7d = []
    parent_asin_rating_avg_prev_rating_7d = []
    parent_asin_list = []

    for item_id in item_ids:
        raw_feat = await get_item_feature(item_id)
        feat = parse_feature(raw_feat)

        meta = get_item_metadata(item_id)

        # metadata
        main_category.append(safe_str(meta.get("main_category")))
        categories.append(safe_str("__".join(meta.get("categories",[]))))
        price.append(safe_str(meta.get("price"), "0"))

        # numeric features
        parent_asin_rating_cnt_365d.append(safe_float(feat.get("parent_asin_rating_cnt_365d")))
        parent_asin_rating_avg_prev_rating_365d.append(safe_float(feat.get("parent_asin_rating_avg_prev_rating_365d")))

        parent_asin_rating_cnt_90d.append(safe_float(feat.get("parent_asin_rating_cnt_90d")))
        parent_asin_rating_avg_prev_rating_90d.append(safe_float(feat.get("parent_asin_rating_avg_prev_rating_90d")))

        parent_asin_rating_cnt_30d.append(safe_float(feat.get("parent_asin_rating_cnt_30d")))
        parent_asin_rating_avg_prev_rating_30d.append(safe_float(feat.get("parent_asin_rating_avg_prev_rating_30d")))

        parent_asin_rating_cnt_7d.append(safe_float(feat.get("parent_asin_rating_cnt_7d")))
        parent_asin_rating_avg_prev_rating_7d.append(safe_float(feat.get("parent_asin_rating_avg_prev_rating_7d")))

        parent_asin_list.append(item_id)


    #Final payload
    payload = {
        "input_data": {
            "user_id": [user_id] * len(item_ids),
            "item_sequence": [item_sequence] * len(item_ids),
            "item_sequence_ts": [item_sequence_ts] * len(item_ids),

            "main_category": main_category,
            "categories": categories,
            "price": price,

            "parent_asin_rating_cnt_365d": parent_asin_rating_cnt_365d,
            "parent_asin_rating_avg_prev_rating_365d": parent_asin_rating_avg_prev_rating_365d,
            "parent_asin_rating_cnt_90d": parent_asin_rating_cnt_90d,
            "parent_asin_rating_avg_prev_rating_90d": parent_asin_rating_avg_prev_rating_90d,
            "parent_asin_rating_cnt_30d": parent_asin_rating_cnt_30d,
            "parent_asin_rating_avg_prev_rating_30d": parent_asin_rating_avg_prev_rating_30d,
            "parent_asin_rating_cnt_7d": parent_asin_rating_cnt_7d,
            "parent_asin_rating_avg_prev_rating_7d": parent_asin_rating_avg_prev_rating_7d,

            "parent_asin": parent_asin_list
        }
    }

    return payload
