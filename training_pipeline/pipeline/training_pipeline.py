from kfp import dsl

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage"]
)
def submit_pytorch_job_skipgram(bucket_name: str, yaml_skipgram_gcs_path: str):
    """
    Download YAML từ GCS và apply vào k8s cluster
    """
    import subprocess
    from google.cloud import storage
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "kubectl"])
    # Download YAML from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(yaml_skipgram_gcs_path)
    
    local_yaml = "/tmp/pytorchjob.yaml"
    blob.download_to_filename(local_yaml)
    
    print(f"Downloaded YAML from gs://{bucket_name}/{yaml_skipgram_gcs_path}")
    
    # Read and print YAML content for debugging
    with open(local_yaml, 'r') as f:
        print("YAML content:")
        print(f.read())
    
    # Apply to k8s
    result = subprocess.run(
        ['kubectl', 'apply', '-f', '/tmp/pytorchjob.yaml', '-n', 'kubeflow'],
        capture_output=True,
        text=True,
        check=False  
    )

    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        raise Exception(f"kubectl apply failed: {result.stderr}")
    

@dsl.component(
    base_image="quangtran1011/training_pipeline:v19",
)
def prep_feature():
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.pre_feature'],
        capture_output=True,
        text=True,
        check=False  
    )

    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        raise Exception(f"ERROR: {result.stderr}")
    

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage"]
)
def submit_pytorch_job_ranking(bucket_name: str, yaml_ranker_gcs_path: str):
    """
    Download YAML từ GCS và apply vào k8s cluster
    """
    import subprocess
    from google.cloud import storage
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "kubectl"])
    # Download YAML from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(yaml_ranker_gcs_path)
    
    local_yaml = "/tmp/pytorchjob.yaml"
    blob.download_to_filename(local_yaml)
    
    print(f"Downloaded YAML from gs://{bucket_name}/{yaml_ranker_gcs_path}")
    
    # Read and print YAML content for debugging
    with open(local_yaml, 'r') as f:
        print("YAML content:")
        print(f.read())
    
    # Apply to k8s
    result = subprocess.run(
        ['kubectl', 'apply', '-f', '/tmp/pytorchjob.yaml', '-n', 'kubeflow'],
        capture_output=True,
        text=True,
        check=False  
    )

    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        raise Exception(f"kubectl apply failed: {result.stderr}")

@dsl.component(
    base_image="quangtran1011/training_pipeline:v19",
)
def precompute():
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.precompute'],
        capture_output=True,
        text=True,
        check=False  
    )

    print(f"Return code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        raise Exception(f"ERROR: {result.stderr}")

# --- PIPELINE ---
@dsl.pipeline(
    name="skipgram-training-pipeline",
    description="Pipeline to check GCS flag and submit PyTorchJobs"
)
def training_pipeline(
    bucket_name: str = "kltn--data",
    yaml_skipgram_gcs_path: str = "config/skipgramptjob.yaml",
    yaml_ranker_gcs_path: str = "config/rankingptjob.yaml",
    mlflow_url: str = "http://<your_vm_ip>:8080",
    qdrant_url: str = "qdrant.serving.svc.cluster.local:6333",
    redis_host: str = "redis_ip",
    redis_port: str = '6379',
):
    # --- A. TRAIN SKIPGRAM ---
    skipgram_task = submit_pytorch_job_skipgram(
        bucket_name=bucket_name,
        yaml_skipgram_gcs_path=yaml_skipgram_gcs_path
    )
    skipgram_task.set_caching_options(False)

    # --- B. PREP FEATURE ---
    prep_feature_task = prep_feature()
    prep_feature_task.set_caching_options(False)
    prep_feature_task.after(skipgram_task)  

    # --- C. TRAIN RANKING ---
    ranking_task = submit_pytorch_job_ranking(
        bucket_name=bucket_name,
        yaml_ranker_gcs_path=yaml_ranker_gcs_path
    )
    ranking_task.set_caching_options(False)
    ranking_task.after(prep_feature_task)

    # --- D. PRECOMPUTE ---
    precompute_task = precompute()
    precompute_task.add_env_variable(
        dsl.EnvVar(
            name="mlflow_url",
            value=mlflow_url
        )
    )

    precompute_task.add_env_variable(
        dsl.EnvVar(
            name="qdrant_url",
            value=qdrant_url
        )
    )

    precompute_task.add_env_variable(
        dsl.EnvVar(
            name="redis_host",
            value=redis_host
        )
    )

    precompute_task.add_env_variable(
        dsl.EnvVar(
            name="redis_port",
            value=redis_port  
        )
    )
    precompute_task.set_caching_options(False)
    precompute_task.after(ranking_task)  


# --- Compile ---
if __name__ == "__main__":
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.yaml"
    )
    
    print("✅ Pipeline compiled to: training_pipeline.yaml")