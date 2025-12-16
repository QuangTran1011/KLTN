import os

import lightning as L
import numpy as np
import pandas as pd
import dill
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import gcsfs
from scipy.sparse import vstack

import mlflow

from typing import List
from .dataset import UserItemBinaryDFDataset
from .id_mapper import IDMapper
from .ranker.model import Ranker
from .ranker.trainer import LitRanker

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep_utils import chunk_transform
from .features.tfm import (
    categories_pipeline_steps,
    price_pipeline_steps,
    rating_agg_pipeline_steps,

)
from .features.timestamp_bucket import calc_sequence_timestamp_bucket
from .ranker.inference import RankerInferenceWrapper



class Args(BaseModel):
    testing: bool = False
    author: str = "quangtran"
    log_to_mlflow: bool = True
    experiment_name: str = "RecSys - Ranker"
    run_name: str = "use-slm-tags"
    notebook_persist_dp: str = None
    random_seed: int = 41
    device: str = None
    item_feature_cols: List[str] = [
        "main_category",
        "categories",
        "price",
        "parent_asin_rating_cnt_365d",
        "parent_asin_rating_avg_prev_rating_365d",
        "parent_asin_rating_cnt_90d",
        "parent_asin_rating_avg_prev_rating_90d",
        "parent_asin_rating_cnt_30d",
        "parent_asin_rating_avg_prev_rating_30d",
        "parent_asin_rating_cnt_7d",
        "parent_asin_rating_avg_prev_rating_7d",
    ]

    qdrant_url: str = None
    qdrant_collection_name: str = "item_desc_sbert"

    max_epochs: int = 1
    batch_size: int = 128
    tfm_chunk_size: int = 10000
    neg_to_pos_ratio: int = 1

    user_col: str = "user_id"
    item_col: str = "parent_asin"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"

    top_K: int = 100
    top_k: int = 10

    embedding_dim: int = 128
    item_sequence_ts_bucket_size: int = 10
    bucket_embedding_dim: int = 16
    dropout: float = 0.3
    early_stopping_patience: int = 5
    learning_rate: float = 0.001
    l2_reg: float = 1e-6

    mlf_item2vec_model_name: str = "item2vec"
    mlf_model_name: str = "rankertest"
    min_roc_auc: float = 0.7

    best_checkpoint_path: str = None

    def init(self):
        self.notebook_persist_dp = os.path.abspath(f"data/{self.run_name}")
        os.makedirs(self.notebook_persist_dp, exist_ok=True)

        if not (mlflow_uri := os.environ.get("MLFLOW_TRACKING_URI")):
            logger.warning(
                f"Environment variable MLFLOW_TRACKING_URI is not set. Setting self.log_to_mlflow to false."
            )
            self.log_to_mlflow = False

        if self.log_to_mlflow:
            logger.info(
                f"Setting up MLflow experiment {self.experiment_name} - run {self.run_name}..."
            )
            self._mlf_logger = MLFlowLogger(
                experiment_name=self.experiment_name,
                run_name=self.run_name,
                tracking_uri=mlflow_uri,
                log_model=False,
            )

        if self.device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        return self


args = Args().init()

def init_model(
    n_users,
    n_items,
    embedding_dim,
    item_sequence_ts_bucket_size,
    bucket_embedding_dim,
    item_feature_size,
    dropout,
    item_embedding=None,
):
    model = Ranker(
        n_users,
        n_items,
        embedding_dim,
        item_sequence_ts_bucket_size=item_sequence_ts_bucket_size,
        bucket_embedding_dim=bucket_embedding_dim,
        item_feature_size=item_feature_size,
        dropout=dropout,
        item_embedding=item_embedding,
    )
    return model

mlf_client = mlflow.MlflowClient()
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{args.mlf_item2vec_model_name}@newly"
)
skipgram_model = model.unwrap_python_model().model
embedding_0 = skipgram_model.embeddings(torch.tensor(0))
embedding_dim = embedding_0.size()[0]
pretrained_item_embedding = skipgram_model.embeddings

train_df = pd.read_parquet("gs://kltn--data/feature_data/train_features_neg_df2.parquet")
val_df = pd.read_parquet("gs://kltn--data/feature_data/val_features_neg_df2.parquet")
train_df['timestamp_unix'] = train_df[args.timestamp_col].astype("int64") // 10**9
val_df['timestamp_unix'] = val_df[args.timestamp_col].astype("int64") // 10**9

unique_user_ids = sorted(train_df[args.user_col].unique())
unique_item_ids = sorted(train_df[args.item_col].unique())
idm = IDMapper()
idm.fit(unique_user_ids, unique_item_ids)
idm_persist_fp = "./data/idm.json"
os.makedirs(os.path.dirname(idm_persist_fp), exist_ok=True) 
idm.save(idm_persist_fp)

def convert_asin_to_idx(inp: str, sequence_length=10, padding_value=-1):
    if inp is None:
        return [padding_value] * sequence_length
    asins = inp.split(",")
    indices = [idm.get_item_index(item_id) for item_id in asins]
    padding_needed = sequence_length - len(indices)
    output = np.pad(
        indices,
        (padding_needed, 0),  # Add padding at the beginning
        "constant",
        constant_values=padding_value,
    )
    return output


def pad_timestamp_sequence(inp: str, sequence_length=10, padding_value=-1):
    if inp is None:
        return [padding_value] * sequence_length
    inp_list = [int(x) for x in inp.split(",")]
    padding_needed = sequence_length - len(inp_list)
    output = np.pad(
        inp_list,
        (padding_needed, 0),  # Add padding at the beginning
        "constant",
        constant_values=padding_value,
    )
    return output


train_df = train_df.assign(
    item_sequence=lambda df: df["user_rating_list_10_recent_asin"].apply(
        convert_asin_to_idx
    ),
    item_sequence_ts=lambda df: df["user_rating_list_10_recent_asin_timestamp"].apply(
        pad_timestamp_sequence
    ),
    item_sequence_ts_bucket=lambda df: df.apply(calc_sequence_timestamp_bucket, axis=1),
)

val_df = val_df.assign(
    item_sequence=lambda df: df["user_rating_list_10_recent_asin"].apply(
        convert_asin_to_idx
    ),
    item_sequence_ts=lambda df: df["user_rating_list_10_recent_asin_timestamp"].apply(
        pad_timestamp_sequence
    ),
    item_sequence_ts_bucket=lambda df: df.apply(calc_sequence_timestamp_bucket, axis=1),
)

tagging_df = pd.read_parquet('gs://kltn--data/tag_embedding_data/tag_embeddings.parquet')
target_dim = 384  

def fix_embedding(emb, target_dim=target_dim):
    if emb is None or (isinstance(emb, float) and np.isnan(emb)):
        return np.zeros(target_dim, dtype=np.float32)
    
    arr = np.array(emb, dtype=np.float32).flatten()
    
    if arr.size > target_dim:
        arr = arr[:target_dim]
    elif arr.size < target_dim:
        arr = np.pad(arr, (0, target_dim - arr.size), 'constant', constant_values=0)
    
    return arr

train_df = pd.merge(train_df, tagging_df, how='left', on=args.item_col)
val_df = pd.merge(val_df, tagging_df, how='left', on=args.item_col)
train_df["tag_embedding"] = train_df["tag_embedding"].apply(fix_embedding)
val_df["tag_embedding"] = val_df["tag_embedding"].apply(fix_embedding)

def load_parquet_from_gcs(path_pattern):
    """Đọc tất cả file Parquet matching pattern từ GCS"""
    fs = gcsfs.GCSFileSystem()
    files = fs.glob(path_pattern)
    tables = [pq.read_table(fs.open(f, "rb")) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

meta_path = "gs://kltn--data/sampled_metadata/*.parquet"
metadata_raw_df = load_parquet_from_gcs(meta_path)

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

# Define the transformations for the columns
rating_agg_cols = [feature.split(":")[1] for feature in item_features]

tfm = [
    ("main_category", OneHotEncoder(handle_unknown="ignore"), ["main_category"]),
    # ("title", Pipeline(title_pipeline_steps()), ["title"]),
    # ("description", Pipeline(description_pipeline_steps()), "description"),
    (
        "categories",
        Pipeline(categories_pipeline_steps()),
        "categories",
    ),  # Count Vectorizer for multi-label categorical
    (
        "price",
        Pipeline(price_pipeline_steps()),
        "price",
    ),  # Normalizing price
    (
        "rating_agg",
        Pipeline(rating_agg_pipeline_steps()),
        rating_agg_cols,
    ),
]
meta_cols = ["main_category", "categories", "price"]
cols = meta_cols + rating_agg_cols
train_df = pd.merge(
    train_df, metadata_raw_df[[args.item_col] + meta_cols], how="left", on=args.item_col
)
val_df = pd.merge(
    val_df, metadata_raw_df[[args.item_col] + meta_cols], how="left", on=args.item_col
)
preprocessing_pipeline = ColumnTransformer(
    transformers=tfm, remainder="drop"  # Drop any columns not specified in transformers
)

# Create a pipeline object
item_metadata_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessing_pipeline),
        (
            "normalizer",
            StandardScaler(with_mean=False),
        ),  # Normalize the numerical outputs since it's an important preconditions for any Deep Learning models
    ]
)

# Fit the pipeline
# Drop duplicated item so that the Pipeline only fit the unique item features
fit_df = train_df.drop_duplicates(subset=[args.item_col])
item_metadata_pipeline.fit(fit_df)

item_metadata_pipeline_fp= "./data/item_metadata_pipeline.dill"
with open(item_metadata_pipeline_fp, "wb") as f:
    dill.dump(item_metadata_pipeline, f)

user_indices = train_df["user_indice"].unique()
item_indices = train_df["item_indice"].unique()
train_item_features = chunk_transform(
    train_df, item_metadata_pipeline, chunk_size=args.tfm_chunk_size
)
val_item_features = chunk_transform(
    val_df, item_metadata_pipeline, chunk_size=args.tfm_chunk_size
)
train_item_features = vstack([x[0] for x in train_item_features])
val_item_features = vstack([x[0] for x in val_item_features])

rating_dataset = UserItemBinaryDFDataset(
    train_df,
    "user_indice",
    "item_indice",
    args.rating_col,
    args.timestamp_col,
    item_feature=train_item_features,
)
val_rating_dataset = UserItemBinaryDFDataset(
    val_df,
    "user_indice",
    "item_indice",
    args.rating_col,
    args.timestamp_col,
    item_feature=val_item_features,
)

train_loader = DataLoader(
    rating_dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=1
)
val_loader = DataLoader(
    val_rating_dataset, batch_size=1024, shuffle=False, drop_last=True, num_workers=1
)

n_items = len(item_indices)
n_users = len(user_indices)
item_feature_size = 747

model = init_model(
    n_users,
    n_items,
    args.embedding_dim,
    args.item_sequence_ts_bucket_size,
    args.bucket_embedding_dim,
    item_feature_size,
    args.dropout,
)

all_items_df = train_df.drop_duplicates(subset=["item_indice"])
all_items_indices = all_items_df["item_indice"].values
all_items_features = item_metadata_pipeline.transform(all_items_df)
all_items_features = vstack([x[0] for x in all_items_features])
all_items_features = all_items_features.toarray().astype(np.float32)
tagging_all_item = np.vstack(all_items_df["tag_embedding"].values)
all_items_features = np.hstack([all_items_features, tagging_all_item])

early_stopping = EarlyStopping(
    monitor="val_roc_auc", patience=args.early_stopping_patience, mode="max", verbose=False
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{args.notebook_persist_dp}/checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="val_roc_auc",
    mode="max",
)

model = init_model(
    n_users,
    n_items,
    args.embedding_dim,
    args.item_sequence_ts_bucket_size,
    args.bucket_embedding_dim,
    item_feature_size,
    dropout=args.dropout,
    item_embedding=pretrained_item_embedding,
)
lit_model = LitRanker(
    model,
    learning_rate=args.learning_rate,
    l2_reg=args.l2_reg,
    log_dir=args.notebook_persist_dp,
    # evaluate_ranking=True,
    idm=idm,
    all_items_indices=all_items_indices,
    all_items_features=all_items_features,
    args=args,
    neg_to_pos_ratio=args.neg_to_pos_ratio,
    checkpoint_callback=checkpoint_callback,
    accelerator=args.device,
)

log_dir = f"{args.notebook_persist_dp}/logs/run"

# train model
trainer = L.Trainer(
    default_root_dir=log_dir,
    max_epochs=args.max_epochs,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator=args.device if args.device else "auto",
    logger=args._mlf_logger if args.log_to_mlflow else None,
    # enable_progress_bar=False
)
trainer.fit(
    model=lit_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

logger.info(f"Loading best checkpoint from {checkpoint_callback.best_model_path}...")
args.best_checkpoint_path = checkpoint_callback.best_model_path

best_trainer = LitRanker.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model=init_model(
        n_users,
        n_items,
        args.embedding_dim,
        args.item_sequence_ts_bucket_size,
        args.bucket_embedding_dim,
        item_feature_size,
        dropout=0,
        item_embedding=pretrained_item_embedding,
    ),
)
best_model = best_trainer.model.to(lit_model.device)

if args.log_to_mlflow:
    # Persist id_mapping so that at inference we can predict based on item_ids (string) instead of item_index
    run_id = trainer.logger.run_id
    mlf_client = trainer.logger.experiment
    mlf_client.log_artifact(run_id, idm_persist_fp)
    # Persist item_feature_metadata pipeline
    mlf_client.log_artifact(run_id, item_metadata_pipeline_fp)

inferrer = RankerInferenceWrapper(best_model)

def generate_sample_item_features():
    sample_row = train_df.iloc[0].fillna(0.0)
    output = dict()
    for col in args.item_feature_cols[3:]:
        v = sample_row[col]    
        if isinstance(v, np.ndarray):
            v = "__".join(
                sample_row[col].tolist()
            )  # Workaround to avoid MLflow Got error: Per-column arrays must each be 1-dimensional
        output[col] = [v]
    return output
sample_input = {
    args.user_col: [idm.get_user_id(0)],
    "item_sequence": [",".join([idm.get_item_id(0), idm.get_item_id(1)])],
    "item_sequence_ts": [
        "1095133116,109770848"
    ],
    "main_category" : ['Books'],
    "categories" : ['Books__History'],
    "price" : ['20.99'],
    **generate_sample_item_features(),
    args.item_col: [idm.get_item_id(0)],
}
sample_output = inferrer.infer([0], [[0, 1]], [[2, 0]], [all_items_features[0]], [0])

if args.log_to_mlflow:
    run_id = trainer.logger.run_id
    sample_output_np = sample_output
    signature = infer_signature(sample_input, sample_output_np)
    idm_filename = 'idm.json'
    item_metadata_pipeline_filename = "item_metadata_pipeline.dill"
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            python_model=inferrer,
            artifact_path="inferrer",
            artifacts={
                # We log the id_mapping to the predict function so that it can accept item_id and automatically convert ot item_indice for PyTorch model to use
                "idm": mlflow.get_artifact_uri(idm_filename),
                "item_metadata_pipeline": mlflow.get_artifact_uri(
                    item_metadata_pipeline_filename
                ),
            },
            # model_config={"use_sbert_features": args.rc.use_sbert_features},
            signature=signature,
            input_example=sample_input,
            registered_model_name=args.mlf_model_name,
        )

if args.log_to_mlflow:
    new_mlf_run = trainer.logger.experiment.get_run(trainer.logger.run_id)
    new_metrics = new_mlf_run.data.metrics
    roc_auc = new_metrics.get("roc_auc", 0.0)

    threshold = 0.5

    if roc_auc >= threshold:
        # Lấy version vừa log
        model_version = (
            mlf_client.get_registered_model(args.mlf_model_name)
            .latest_versions[0]
            .version
        )

        # Gán alias 'newly' để deploy
        mlf_client.set_registered_model_alias(
            name=args.mlf_model_name,
            version=model_version,
            alias="newly",
        )

        # Optional: tag author
        mlf_client.set_model_version_tag(
            name=args.mlf_model_name,
            version=model_version,
            key="author",
            value=args.author,
        )

        logger.info(
            f"Assigned alias 'newly' to version {model_version} (roc_auc={roc_auc:.4f})."
        )
    else:
        logger.info(
            f"roc_auc={roc_auc:.4f} < {threshold}, skip assigning alias 'newly'."
        )
