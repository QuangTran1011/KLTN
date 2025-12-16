import json
import time
from datetime import datetime
from typing import List

import requests
from loguru import logger

FEAST_ONLINE_SERVER_HOST = "localhost"
FEAST_ONLINE_SERVER_PORT = 8815


def get_user_item_sequence(user_id: str):
    # Define the URL
    url = "http://localhost:8000/recs/get_user_feature"  

    headers = {
        "accept": "application/json",
    }
    params = {"user_id": user_id}

    try:
        # Make the GET request
        response = requests.get(url, headers=headers, params=params)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        # Return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle exceptions (e.g., network errors, HTTP errors)
        logger.error(f"An error occurred: {e}")
        return None


def parse_feature(data):
    """Parse feature data from Feast response"""
    feature_names = data["metadata"]["feature_names"]
    results = data["results"]

    parsed = {}

    for i, name in enumerate(feature_names):
        parsed[name] = results[i]["values"][0]

    return parsed


def push_new_item_sequence(
    user_id: str, new_items: List[str], sequence_length: int = 10
):
    """
    Push new item sequence to Feast online store
    
    Args:
        user_id: User ID
        new_items: List of new item IDs to add
        sequence_length: Maximum length of sequence to keep
    """
    try:
        # Get current user features
        response = get_user_item_sequence(user_id)
        
        if response is None:
            logger.error(f"Failed to get user features for {user_id}")
            return
        
        user_feat = parse_feature(response)

        # Parse current item sequences - CONVERT STRING TO LIST
        item_sequences_str = user_feat.get("user_rating_list_10_recent_asin", "")
        
        # Handle empty string or parse comma-separated string
        if item_sequences_str:
            if isinstance(item_sequences_str, str):
                item_sequences = item_sequences_str.split(",")
            else:
                item_sequences = list(item_sequences_str)
        else:
            item_sequences = []
        
        logger.info(f"Current item sequences for {user_id}: {item_sequences}")
        
        # Add new items to sequence
        new_item_sequences = item_sequences + new_items
        
        # Keep only last N items
        new_item_sequences = new_item_sequences[-sequence_length:]
        
        # Convert back to comma-separated string
        new_item_sequences_str = ",".join(new_item_sequences)
        
        logger.info(f"New item sequences: {new_item_sequences_str}")

        # Parse timestamps - CONVERT STRING TO LIST
        item_sequence_tss_str = user_feat.get("user_rating_list_10_recent_asin_timestamp", "")
        
        if item_sequence_tss_str:
            if isinstance(item_sequence_tss_str, str):
                item_sequence_tss = [int(ts) for ts in item_sequence_tss_str.split(",") if ts]
            else:
                item_sequence_tss = list(item_sequence_tss_str)
        else:
            item_sequence_tss = []
        
        # Add new timestamps (one for each new item)
        current_timestamp = int(time.time())
        new_item_sequence_tss = item_sequence_tss + [current_timestamp] * len(new_items)
        
        # Keep only last N timestamps
        new_item_sequence_tss = new_item_sequence_tss[-sequence_length:]
        
        # Convert back to comma-separated string
        new_item_sequence_tss_str = ",".join([str(ts) for ts in new_item_sequence_tss])
        
        logger.info(f"New timestamps: {new_item_sequence_tss_str}")

        # Prepare event data for Feast push API
        event_dict = {
            "user_id": [user_id],
            "timestamp": [str(datetime.now())],
            "dedup_rn": [1],  
            "user_rating_cnt_90d": [user_feat.get("user_rating_cnt_90d", 1)],  
            "user_rating_avg_prev_rating_90d": [user_feat.get("user_rating_avg_prev_rating_90d", 4.5)],  
            "user_rating_list_10_recent_asin": [new_item_sequences_str],
            "user_rating_list_10_recent_asin_timestamp": [new_item_sequence_tss_str],
        }
        
        push_data = {
            "push_source_name": "user_rating_stats_push_source",
            "df": event_dict,
            "to": "online",
        }
        
        logger.info(f"Event data to be pushed to feature store: {event_dict}")
        
        # Push to Feast online store
        r = requests.post(
            f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/push",
            data=json.dumps(push_data),
        )

        if r.status_code == 200:
            logger.info(f"Successfully pushed new item sequence for user {user_id}")
        else:
            logger.error(f"Error pushing to Feast: {r.status_code} {r.text}")
            
    except Exception as e:
        logger.error(f"Error in push_new_item_sequence: {e}", exc_info=True)
        raise


# push_new_item_sequence(
#     'AEHS443XTMNSX4TNW2YYLWND6FTQ',
#     ['B079L5T84M']
# )