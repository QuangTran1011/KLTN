from mlflow.tracking import MlflowClient
import subprocess

client = MlflowClient("http://34.67.211.252:8080")

# Lấy danh sách version
versions = client.search_model_versions("name='rankertest'")

for mv in versions:
    full = client.get_model_version(name=mv.name, version=mv.version)

    aliases = full.aliases or []

    if "newly" in aliases:
        print("Found new model:", mv.version)
        subprocess.run(["kubectl", "apply", "-f", f"/home/quangtran/k8s/ranker-model-server.yaml", '-n', 'serving'], check=True)
    
        client.set_model_version_alias(
            name="rankertest",
            version=mv.version,
            alias="deployed"
        )

