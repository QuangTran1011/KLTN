import os
import sys
from typing import Any
import json
import pandas as pd

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from mlflow.models.signature import infer_signature
from pydantic import BaseModel, PrivateAttr
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import gcsfs

import mlflow

from .id_mapper import IDMapper
from .SkipGram.dataset import SkipGramDataset
from .SkipGram.inference import SkipGramInferenceWrapper
from .SkipGram.model import SkipGram
from .SkipGram.trainer import LitSkipGram

class Args(BaseModel):
    user_col: str = 'user_id'
    item_col: str = 'parent_asin'
    timestamp_col: str = "timestamp"

    testing: bool = False
    author: str = "quangtran"
    log_to_mlflow: bool = True
    _mlf_logger: Any = PrivateAttr()
    experiment_name: str = "Item2Vec"
    run_name: str = "001-report-best-checkpoint"
    notebook_persist_dp: str = None
    random_seed: int = 41
    device: str = None

    max_epochs: int = 1    
    batch_size: int = 1024

    num_negative_samples: int = 2
    window_size: int = 1

    embedding_dim: int = 128
    early_stopping_patience: int = 5
    learning_rate: float = 0.003
    l2_reg: float = 1e-5

    mlf_model_name: str = "item2vec"
    min_roc_auc: float = 0.7

    def init(self):
        pvc_root = "/mnt/pvc"     
        self.notebook_persist_dp = os.path.join(pvc_root, self.run_name)
        os.makedirs(self.notebook_persist_dp, exist_ok=True)

        if not (mlflow_uri := os.environ.get("MLFLOW_URI")):
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

unique_user_ids = sorted(train_df[args.user_col].unique())
unique_item_ids = sorted(train_df[args.item_col].unique())
idm = IDMapper()
idm.fit(unique_user_ids, unique_item_ids)
idm_persist_fp = "./data/idm.json"
os.makedirs(os.path.dirname(idm_persist_fp), exist_ok=True) 
idm.save(idm_persist_fp)

def get_sequence(df, user_col=args.user_col, item_col=args.item_col):
    return (
        df.groupby(user_col)[item_col]
        .agg(list)
        .loc[lambda s: s.apply(len) > 1] 
    ).values.tolist()

item_sequence = train_df.pipe(get_sequence)
val_item_sequence = val_df.pipe(get_sequence)

train_dataset = SkipGramDataset(
    item_sequence,
    window_size=args.window_size,
    negative_samples=args.num_negative_samples,
    id_to_idx=idm.item_to_index,
)
val_dataset = SkipGramDataset(
    val_item_sequence,
    train_dataset.interacted,
    train_dataset.item_freq,
    window_size=args.window_size,
    negative_samples=args.num_negative_samples,
    id_to_idx=idm.item_to_index,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,         
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
)

def init_model(n_items, embedding_dim):
    model = SkipGram(n_items, embedding_dim)
    return model

n_items = len(unique_item_ids)

early_stopping = EarlyStopping(
    monitor="val_roc_auc", patience=args.early_stopping_patience, mode="max", verbose=False
)

mlflow.pytorch.autolog(log_models=False, checkpoint=False)
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{args.notebook_persist_dp}/checkpoints",
    filename="best_checkpoint",
    save_top_k=1,
    monitor="val_roc_auc",
    mode="max",
)

# model
model = init_model(n_items, args.embedding_dim)
lit_model = LitSkipGram(
    model,
    learning_rate=args.learning_rate,
    l2_reg=args.l2_reg,
    log_dir=args.notebook_persist_dp,
    accelerator=args.device,
    checkpoint_callback=checkpoint_callback,
)

log_dir = f"{args.notebook_persist_dp}/logs/run"

# # train model
trainer = L.Trainer(
    default_root_dir=log_dir,
    accelerator="cpu",
    devices=1,       
    strategy="ddp",
    max_epochs=args.max_epochs,
    callbacks=[early_stopping, checkpoint_callback],
    # accelerator=args.device if args.device else "auto",
    logger=args._mlf_logger if args.log_to_mlflow else None,
    enable_progress_bar=False
)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

best_trainer = LitSkipGram.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model=init_model(n_items, args.embedding_dim),
)

best_model = best_trainer.model
inferrer = SkipGramInferenceWrapper(best_model)
sample_input = {
    "item_1_ids": [idm.index_to_item[0]],
    "item_2_ids": [idm.index_to_item[1]],
}
sample_output = inferrer.infer([0], [1])

if args.log_to_mlflow:
    run_id = trainer.logger.run_id
    mlf_client = trainer.logger.experiment
    mlf_client.log_artifact(run_id, idm_persist_fp)

if args.log_to_mlflow:
    run_id = trainer.logger.run_id
    sample_output_np = sample_output
    signature = infer_signature(sample_input, sample_output_np)
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            python_model=inferrer,
            artifact_path="inferrer",
            artifacts={"id_mapping": mlflow.get_artifact_uri("idm.json")},
            signature=signature,
            input_example=sample_input,
            registered_model_name=args.mlf_model_name,
        )

if args.log_to_mlflow:
    new_mlf_run = trainer.logger.experiment.get_run(trainer.logger.run_id)
    new_metrics = new_mlf_run.data.metrics
    roc_auc = new_metrics.get("roc_auc", 0.0)

    threshold = 0.0  # 0.0 to test pipeline because epoch = 1 (not train)

    if roc_auc >= threshold:
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
