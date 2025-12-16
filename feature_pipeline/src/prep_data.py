import os
import sys
import gc
import dill
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gcsfs
from datasets import load_dataset
from feast import FeatureStore
from loguru import logger
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_prep_utils import chunk_transform, handle_dtypes, parse_dt
from src.features.tfm import (
    categories_pipeline_steps,
    price_pipeline_steps,
    rating_agg_pipeline_steps,
)
from src.features.timestamp_bucket import calc_sequence_timestamp_bucket
from src.id_mapper import IDMapper, map_indice


class Args(BaseModel):
    run_name: str = "000-prep-data"
    testing: bool = True
    notebook_persist_dp: str = None
    random_seed: int = 41

    user_col: str = "user_id"
    item_col: str = "parent_asin"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"

    tfm_chunk_size: int = 5000
    sequence_length: int = 10

    def init(self):
        self.notebook_persist_dp = os.path.abspath(f"data/{self.run_name}")
        if not self.testing:
            os.makedirs(self.notebook_persist_dp, exist_ok=True)
        return self


args = Args().init()


# ---------------------------------------------------------------------
# Utility: read parquet from GCS in chunks
# ---------------------------------------------------------------------
def load_parquet_from_gcs(path_pattern):
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(path_pattern)
    logger.info(f"Found {len(files)} files under {path_pattern}")
    tables = [pq.read_table(fs.open(f, "rb")) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)


# ---------------------------------------------------------------------
# Step 1: Load raw metadata (Amazon dataset)
# ---------------------------------------------------------------------
def load_metadata():
    logger.info("Loading Amazon metadata ...")
    metadata_raw = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Books",
        split="full",
        trust_remote_code=True,
    )
    df = metadata_raw.to_pandas()
    logger.info(f"Metadata shape: {df.shape}")
    return df


# ---------------------------------------------------------------------
# Step 2: Load train/val, preprocess and merge
# ---------------------------------------------------------------------
def load_and_prepare_data():
    logger.info("Loading train/val data from GCS ...")
    TRAIN_PATH = "gs://kltn-data/train_df20/*.parquet"
    TEST_PATH = "gs://kltn-data/val_df20/*.parquet"

    drop_cols = [
        "asin", "helpful_vote", "images", "text", "title",
        "verified_purchase", "ts"
    ]

    train_df = (
        load_parquet_from_gcs(TRAIN_PATH)
        .drop(columns=drop_cols)
        .drop_duplicates(subset=[args.user_col, args.item_col, args.timestamp_col])
    )
    val_df = (
        load_parquet_from_gcs(TEST_PATH)
        .drop(columns=drop_cols)
        .drop_duplicates(subset=[args.user_col, args.item_col, args.timestamp_col])
    )

    full_df = (
        pd.concat([train_df, val_df], axis=0)
        .pipe(parse_dt)
        .pipe(handle_dtypes)
        .assign(timestamp_unix=lambda df: df[args.timestamp_col].astype("int64") // 10**9)
    )
    logger.info(f"Full data shape: {full_df.shape}")

    return train_df, val_df, full_df


# ---------------------------------------------------------------------
# Step 3: ID mapping
# ---------------------------------------------------------------------
def prepare_id_mapper(train_df, val_df):
    unique_user_ids = sorted(train_df[args.user_col].unique())
    unique_item_ids = sorted(train_df[args.item_col].unique())
    idm = IDMapper()
    idm.fit(unique_user_ids, unique_item_ids)
    idm_persist_fp = "./data/idm.json"
    idm.save(idm_persist_fp)
    logger.info(f"Saved IDMapper to {idm_persist_fp}")
    return idm


# ---------------------------------------------------------------------
# Step 4: Fetch features from Feast
# ---------------------------------------------------------------------
def fetch_features(full_df, store, idm):
    logger.info("Fetching item and user features from Feast ...")

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
    features_df = store.get_historical_features(
        full_df[[args.item_col, args.timestamp_col]].drop_duplicates(), item_features
    ).to_df()

    full_features_df = pd.merge(
        full_df, features_df, on=[args.item_col, args.timestamp_col], how="left"
    ).pipe(map_indice, idm, args.user_col, args.item_col)

    del features_df
    gc.collect()

    user_features = [
        "user_rating_stats:user_rating_cnt_90d",
        "user_rating_stats:user_rating_avg_prev_rating_90d",
        "user_rating_stats:user_rating_list_10_recent_asin",
        "user_rating_stats:user_rating_list_10_recent_asin_timestamp",
    ]
    features_df = store.get_historical_features(
        full_df[[args.user_col, args.timestamp_col]].drop_duplicates(), user_features
    ).to_df()

    full_features_df = pd.merge(
        full_features_df, features_df, on=[args.user_col, args.timestamp_col], how="left"
    )

    del features_df
    gc.collect()

    return full_features_df


# ---------------------------------------------------------------------
# Step 5: Prepare item/user sequences
# ---------------------------------------------------------------------
def add_sequence_features(full_features_df, idm):
    def convert_asin_to_idx(inp: str, sequence_length=10, padding_value=-1):
        if inp is None:
            return [padding_value] * sequence_length
        asins = inp.split(",")
        indices = [idm.get_item_index(item_id) for item_id in asins]
        padding_needed = sequence_length - len(indices)
        return np.pad(indices, (padding_needed, 0), "constant", constant_values=padding_value)

    def pad_timestamp_sequence(inp: str, sequence_length=10, padding_value=-1):
        if inp is None:
            return [padding_value] * sequence_length
        inp_list = [int(x) for x in inp.split(",")]
        padding_needed = sequence_length - len(inp_list)
        return np.pad(inp_list, (padding_needed, 0), "constant", constant_values=padding_value)

    return full_features_df.assign(
        item_sequence=lambda df: df["user_rating_list_10_recent_asin"].apply(convert_asin_to_idx),
        item_sequence_ts=lambda df: df["user_rating_list_10_recent_asin_timestamp"].apply(pad_timestamp_sequence),
        item_sequence_ts_bucket=lambda df: df.apply(calc_sequence_timestamp_bucket, axis=1),
    )


# ---------------------------------------------------------------------
# Step 6: Build feature pipeline
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Step 6: Build feature pipeline
# ---------------------------------------------------------------------
def build_item_metadata_pipeline(train_df, val_df, metadata_raw_df, full_features_df):
    import psutil

    def log_ram(msg):
        mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
        logger.info(f"[RAM] {msg}: {mem:.2f} GB")

    val_timestamp = pd.to_datetime(
        val_df[args.timestamp_col].astype(int), unit="ms", utc=True
    ).min()

    train_df_final = full_features_df.loc[lambda df: df["timestamp"].lt(val_timestamp)]
    val_df_final = full_features_df.loc[lambda df: df["timestamp"].ge(val_timestamp)]

    meta_cols = ["main_category", "title", "description", "categories", "price"]
    rating_agg_cols = [
        "parent_asin_rating_stats:parent_asin_rating_cnt_365d".split(":")[1],
        "parent_asin_rating_stats:parent_asin_rating_avg_prev_rating_365d".split(":")[1],
    ]
    tfm = [
        ("main_category", OneHotEncoder(handle_unknown="ignore"), ["main_category"]),
        ("categories", Pipeline(categories_pipeline_steps()), "categories"),
        ("price", Pipeline(price_pipeline_steps()), "price"),
        ("rating_agg", Pipeline(rating_agg_pipeline_steps()), rating_agg_cols),
    ]

    train_features_df = pd.merge(
        train_df_final,
        metadata_raw_df[[args.item_col] + meta_cols],
        how="left",
        on=args.item_col,
    )
    val_features_df = pd.merge(
        val_df_final,
        metadata_raw_df[[args.item_col] + meta_cols],
        how="left",
        on=args.item_col,
    )

    # Giải phóng metadata_raw_df sớm sau khi merge xong
    log_ram("Before deleting metadata_raw_df")
    del metadata_raw_df
    gc.collect()
    log_ram("After deleting metadata_raw_df")

    preprocessing_pipeline = ColumnTransformer(
        transformers=tfm, remainder="drop"
    )

    item_metadata_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            ("normalizer", StandardScaler()),
        ]
    )

    fit_df = train_features_df.drop_duplicates(subset=[args.item_col])
    item_metadata_pipeline.fit(fit_df)
    del fit_df, train_df_final, val_df_final, full_features_df
    gc.collect()

    with open("./data/item_metadata_pipeline.dill", "wb") as f:
        dill.dump(item_metadata_pipeline, f)
    logger.info("Saved item_metadata_pipeline.dill")

    return train_features_df, val_features_df



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("===== START PREPROCESSING =====")
    metadata_raw_df = load_metadata()
    train_df, val_df, full_df = load_and_prepare_data()
    idm = prepare_id_mapper(train_df, val_df)
    store = FeatureStore(
        repo_path="./feature_pipeline/feature_store/feature_repo"
    )
    full_features_df = fetch_features(full_df, store, idm)
    full_features_df = add_sequence_features(full_features_df, idm)

    train_features_df, val_features_df = build_item_metadata_pipeline(
        train_df, val_df, metadata_raw_df, full_features_df
    )

    train_persist_fp = "gs://kltn-data/feature_data/train_features_df.parquet"
    val_persist_fp = "gs://kltn-data/feature_data/val_features_df.parquet"

    train_features_df.to_parquet(train_persist_fp, index=False)
    val_features_df.to_parquet(val_persist_fp, index=False)

    logger.info("Saved train/val parquet to GCS.")
    logger.info("===== DONE =====")

    del train_features_df, val_features_df, metadata_raw_df
    gc.collect()


if __name__ == "__main__":
    main()
