import pandas as pd
import redis
from tqdm.auto import tqdm
import json 
import os

redis_recent_key_prefix = "feature:user:recent_items:"
redis_popular_key = "output:popular"
redis_host = "127.0.0.1"
redis_port = 6379
top_K = 100


val_df = pd.read_parquet("gs://kltn--data/feature_data/val_features_neg_df2.parquet")
train_df = pd.read_parquet("gs://kltn--data/feature_data/train_features_neg_df2.parquet")

full_df = pd.concat([train_df, val_df], axis=0)

# print(full_df.iloc[-1])

latest_df = full_df.assign(
    recency=lambda df: df.groupby("user_id")['timestamp'].rank(
        method="first", ascending=False
    )
).loc[lambda df: df["recency"].eq(1)]

popular_recs = (
    full_df.groupby('parent_asin').size().sort_values(ascending=False).head(top_K)
)
popular_item = popular_recs.index[0]

r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

pipe = r.pipeline(transaction=False)  
count = 0

for i, row in tqdm(latest_df.iterrows(), total=latest_df.shape[0]):
    user_id = str(row['user_id'])
    seq = row['user_rating_list_10_recent_asin']

    if seq is None or pd.isna(seq):
        item_sequences = popular_item
    else:
        item_sequences = seq.replace(",", "__")
    
    pipe.set(redis_recent_key_prefix + user_id, item_sequences)
    count += 1

# gửi tất cả lệnh cùng lúc
pipe.execute()
print(f"Đã upload {count} user sequences vào Redis.")


key = redis_popular_key
value = json.dumps(
    {
        "rec_item_ids": popular_recs.index.tolist(),
        "rec_scores": popular_recs.values.tolist(),
    }
)
r.set(key, value)
