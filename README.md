# RecSys
An end-to-end, fully automated recommendation system covering the entire pipeline from data processing to cloud-native serving, providing standardized workflows for training, fine-tuning, deployment, and online serving, integrated with modern MLOps tools.

## System Flow Overview
The system consists of three main workflows: the data pipeline, the training pipeline, and the serving pipeline.
## Table of Contents
## Project Structure
## Implementation
### 1. Set Up Environment
Clone this repository
Set up Python Environment:
```bash
pip install uv==0.8.13
uv sync --all-groups
```
Set up Cloud(GCP):
- Create 1 Vm and 1 Cluster
- Create a Service Account or use Default Service Account
- Add Roles to Service Account: BigQuery Admin, Cloud SQL Admin, Kubernetes Engine Admin, Service Account Admin, Service Account User, Storage Admin(can be replaced with more restrictive roles for better security)
- Dowload Service Account Key (Json)

### 2. Data Pipeline
#### Prepare
Create Secret from service account:
```bash
kubectl create secret generic gcp-sa-secret \
  --from-file=gcp-key.json=path/to/key.json
```

Create namespace: `kubectl create ns serving` (all components are deployed here for simplicity, although separate namespaces can be used)

Create Data Bucket(in my project: `kltn--data`)  
- Data is updated on a scheduled basis using partitioned storage. Files follow the naming convention: `recsys_data_upto_YYYY_MM_DD.parquet`.

Create a Feature Store registry bucket (in my project: `feast-data`) and create the `registry.db` file inside this bucket.

Create a Redis instance (GCP Memorystore) and obtain the connection details.

Create a BigQuery dataset named `kltn` and three table named `traindatareviewtest` within this dataset.

# Airflow Pipeline
Install Airflow:
- `cd airflow-dags` 
- Update the repo value in dags.gitSync to point to your repository in the Airflow values file.
- helm install airlow ./airflow

Dags:
- Update the `project_id` value in the dbt profiles.
- Update the `project_id` and `connection_string` (Redis IP) in `feature_store.yaml`.
- Build the Docker image from `airflow_all_in_one.Dockerfile` and push it to Docker Hub.
- Update the IMAGE name and Kubernetes IP in `dags/data_pipeline.py`, then push the changes to GitHub.

Access Airflow:
- `kubectl port-forward svc/airflow-web 8080:8080` -n serving
- access `localhost:8080'
- Go to Admin -> Connection -> Add connection: name is `google_cloud_default`, type Google Cloud, Copy file Service Account Key Json and Paste into 

You can now run the data pipeline by triggering the DAGs.  
However, during the first run, the pipeline is expected to fail at the Feature Store step because it has not been initialized yet.

Apply feast:

GCP Memorystore (Redis) is only accessible within the internal VPC, so an SSH tunnel is used for access.
- SSH into the VM created above using an SSH key.
- ssh to redis: `gcloud compute ssh vm-name -- -L 6379:<PRIVATE_IP_OF_REDIS>:6379`.

Apply feast:
- `cd feature pipeline/feature_store/feature_repo
- cd $ROOT_DIR && MATERIALIZE_CHECKPOINT_TIME=$(uv run scripts/check_oltp_max_timestamp.py 2>&1 | awk -F'<ts>|</ts>' '{print $2}')
