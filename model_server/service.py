import os
import sys

import bentoml
from loguru import logger
from mlflow import MlflowClient


model_cfg = {
    "rankertest": {
        "name": "rankertest",
        "deploy_alias": "newly",
        "model_uri": "models:/rankertest@newly",
    },
}

for name, cfg in model_cfg.items():
    bentoml.mlflow.import_model(
        name,
        model_uri=cfg["model_uri"],
        signatures={
            "predict": {"batchable": True},
        },
    )


@bentoml.service(name="ranker_service")
class RankerService:
    model_name = "rankertest"

    def __init__(self):
        self.model = None
        self.model_version = None
        self.ready = False

    @bentoml.on_startup
    def load_model(self):
        bento_model = bentoml.models.get(self.model_name)
        self.model = bentoml.mlflow.load_model(bento_model)

        deploy_alias = model_cfg[self.model_name]["deploy_alias"]
        mlf_client = MlflowClient()
        self.model_version = mlf_client.get_model_version_by_alias(
            self.model_name, deploy_alias
        ).version

        self.ready = True
        logger.info(
            f"Model loaded. name={self.model_name}, "
            f"alias={deploy_alias}, version={self.model_version}"
        )

    @bentoml.api
    def ready_check(self) -> bool:
        return self.ready

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        rv["metadata"] = {
            "model_version": self.model_version,
            "model_name": self.model_name,
        }
        return rv