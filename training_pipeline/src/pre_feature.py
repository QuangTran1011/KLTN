import os
import pandas as pd
from feast import FeatureStore
from loguru import logger
from pydantic import BaseModel
import pyarrow.parquet as pq
import gcsfs
from datetime import timedelta, datetime

from .data_prep_utils import handle_dtypes, parse_dt
from .ranker.negative_sampling import generate_negative_samples_mp
from .id_mapper import IDMapper, map_indice

class Args(BaseModel):
    run_name: str = "000-prep-data"
    testing: bool = True
    notebook_persist_dp: str = None
    random_seed: int = 41

    user_col: str = "user_id"
    item_col: str = "parent_asin"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"
    neg_to_pos_ratio: int = 1

    def init(self):
        self.notebook_persist_dp = os.path.abspath(f"data/{self.run_name}")
        if not self.testing:
            os.makedirs(self.notebook_persist_dp, exist_ok=True)

        return self


args = Args().init()

def load_parquet_from_gcs(path_pattern):
    """Đọc tất cả file Parquet matching pattern từ GCS"""
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(path_pattern)
    tables = [pq.read_table(fs.open(f, "rb")) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

TRAIN_PATH = "gs://kltn--data/train_df20/*.parquet"
TEST_PATH = "gs://kltn--data/val_df20/*.parquet"

train_df = load_parquet_from_gcs(TRAIN_PATH).drop(columns=[ 'asin', 'helpful_vote', 'images',
       'text', 'title', 'verified_purchase', 'ts']).drop_duplicates(subset=[args.user_col, args.item_col, args.timestamp_col])
val_df = load_parquet_from_gcs(TEST_PATH).drop(columns=[ 'asin', 'helpful_vote', 'images',
       'text', 'title', 'verified_purchase', 'ts']).drop_duplicates(subset=[args.user_col, args.item_col, args.timestamp_col])

ts_max_ms = train_df[args.timestamp_col].max()
val_timestamp = datetime.fromtimestamp(ts_max_ms / 1000) + timedelta(seconds=1)
val_timestamp = pd.Timestamp(val_timestamp).tz_localize("UTC")

full_df = (
    pd.concat([train_df, val_df], axis=0)
    .pipe(parse_dt)
    .pipe(handle_dtypes)
    .assign(timestamp_unix=lambda df: df[args.timestamp_col].astype("int64") // 10**9)
)

unique_user_ids = sorted(train_df[args.user_col].unique())
unique_item_ids = sorted(train_df[args.item_col].unique())
idm = IDMapper()
idm.fit(unique_user_ids, unique_item_ids)

train_df = train_df.pipe(map_indice, idm, args.user_col, args.item_col)
val_df = val_df.pipe(map_indice, idm, args.user_col, args.item_col)
idm_persist_fp = "./data/idm.json"
os.makedirs(os.path.dirname(idm_persist_fp), exist_ok=True) 
idm.save(idm_persist_fp)

store = FeatureStore(
    repo_path="/app/src/feature_repo"
)
item_features = [
    "parent_asin_rating_stats:parent_asin_rating_cnt_365d",
    "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_365d",
    "parent_asin_rating_stats:parent_asin_rating_cnt_90d",
    "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_90d",
    "parent_asin_rating_stats:parent_asin_rating_cnt_30d",
    "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_30d",
    "parent_asin_rating_stats:parent_asin_rating_cnt_7d",
    "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_7d",
]

features_df = store.get_historical_features(full_df[[args.item_col, args.timestamp_col]].drop_duplicates(), item_features).to_df()
full_features_df = pd.merge(
    full_df, features_df, on=[args.item_col, args.timestamp_col], how="left"
).pipe(map_indice, idm, args.user_col, args.item_col)


user_features = [
    "user_rating_stats:user_rating_cnt_90d",
    "user_rating_stats:user_rating_avg_prev_rating_90d",
    "user_rating_stats:user_rating_list_10_recent_asin",
    "user_rating_stats:user_rating_list_10_recent_asin_timestamp",
]

features_df = store.get_historical_features(full_df[[args.user_col, args.timestamp_col]].drop_duplicates(), user_features).to_df()
full_features_df = pd.merge(
    full_features_df, features_df, on=[args.user_col, args.timestamp_col], how="left"
)

item_features_df = full_features_df.drop_duplicates(subset=[args.item_col])[
    [args.item_col, "item_indice"]
]

ufs = [
    "user_rating_list_10_recent_asin_timestamp",
    "user_id",
    "user_rating_cnt_90d",
    "user_rating_avg_prev_rating_90d",
    "user_rating_list_10_recent_asin",
]

neg_df = generate_negative_samples_mp(
    full_features_df,
    "user_indice",
    "item_indice",
    args.rating_col,
    neg_label=0,
    neg_to_pos_ratio=args.neg_to_pos_ratio,
    seed=args.random_seed,
    features=ufs,
    num_processes=7
)

neg_df = neg_df.pipe(
    lambda df: pd.merge(
        df, item_features_df, how="left", on="item_indice", validate="m:1"
    )
)
neg_ts_features_df = store.get_historical_features(neg_df[[args.item_col, args.timestamp_col]].drop_duplicates(), item_features).to_df()
neg_df = pd.merge(
    neg_df, neg_ts_features_df, on=[args.item_col, args.timestamp_col], how="left"
)

full_features_df = (
    pd.concat([full_features_df, neg_df], axis=0)
    .reset_index(drop=True)
    .sample(frac=1, replace=False, random_state=args.random_seed)
)

train_neg_df = full_features_df.loc[lambda df: df[args.timestamp_col].lt(val_timestamp)]
val_neg_df = full_features_df.loc[lambda df: df[args.timestamp_col].ge(val_timestamp)]

val_persist_fp = "gs://kltn--data/feature_data/val_features_neg_df2.parquet"
val_neg_df.to_parquet(val_persist_fp, index=False)
train_persist_fp = "gs://kltn--data/feature_data/train_features_neg_df2.parquet"
train_neg_df.to_parquet(train_persist_fp, index=False)