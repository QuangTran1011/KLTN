import os
import tempfile
from google.cloud import storage
from kfp import Client

# =====================
# Config
# =====================
GCS_PIPELINE_PATH = os.environ.get(
    "GCS_PIPELINE_PATH",
    "gs://kltn--data/config/training_pipeline.yaml"
)

PIPELINE_NAME = os.environ.get(
    "PIPELINE_NAME",
    "training-pipeline"
)

EXPERIMENT_NAME = os.environ.get(
    "EXPERIMENT_NAME",
    "training-experiment"
)

KFP_HOST = os.environ.get(
    "KFP_HOST",
    "http://ml-pipeline.kubeflow.svc.cluster.local:8080"
)

PIPELINE_ARGS = {
    "bucket_name": "kltn--data",
    "yaml_skipgram_gcs_path": "config/skipgramptjob.yaml",
    "yaml_ranker_gcs_path": "config/rankingptjob.yaml",
}

def download_from_gcs(gcs_uri: str, local_path: str):
    assert gcs_uri.startswith("gs://")

    bucket_name, blob_path = gcs_uri[5:].split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        local_pipeline_path = os.path.join(tmpdir, "pipeline.yaml")

        # 1. Download pipeline YAML
        download_from_gcs(GCS_PIPELINE_PATH, local_pipeline_path)

        # 2. Init Kubeflow client (in-cluster)
        client = Client(host=KFP_HOST)

        # 3. Create run
        run = client.create_run_from_pipeline_package(
            pipeline_file=local_pipeline_path,
            arguments=PIPELINE_ARGS,
            experiment_name=EXPERIMENT_NAME,
            run_name=f"run-{PIPELINE_NAME}"
        )

        print(f"Pipeline triggered. Run ID: {run.run_id}")


if __name__ == "__main__":
    main()
