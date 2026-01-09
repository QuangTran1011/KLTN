from mlflow.tracking import MlflowClient
import subprocess
import yaml
import os

YAML_PATH = os.getenv("INFER_SERVICE_YAML", "ranker-inferenceservice.yaml")
NAMESPACE = os.getenv("K8S_NAMESPACE", "serving")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "rankertest")

client = MlflowClient(MLFLOW_TRACKING_URI)

versions = client.search_model_versions("name={MLFLOW_TRACKING_URI}")

for mv in versions:
    full = client.get_model_version(name=mv.name, version=mv.version)
    aliases = full.aliases or []

    if "newly" in aliases and "deployed" not in aliases:
        print("Found new model:", mv.version)

        with open(YAML_PATH) as f:
            data = yaml.safe_load(f)

        # Update MODEL_VERSION
        envs = data["spec"]["predictor"]["containers"][0]["env"]
        for env in envs:
            if env["name"] == "MODEL_VERSION":
                env["value"] = str(mv.version)

        with open(YAML_PATH, "w") as f:
            yaml.safe_dump(data, f)

        subprocess.run(
            ["kubectl", "apply", "-f", YAML_PATH, "-n", NAMESPACE],
            check=True
        )

        client.set_registered_model_alias(
            name=mv.name,
            version=mv.version,
            alias="deployed",
        )

        print(f"Rolled out model version {mv.version}")
