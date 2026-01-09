import re
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from kfp import Client

# ----------------------------
# Config
# ----------------------------
BUCKET = "kltn--data"
PREFIX = "partitiondata/"
FILE_PATTERN = r"recsys_data_upto_(\d{4})_(\d{2})_(\d{2})\.parquet"
IMAGE = "quangtran1011/airflow_all_in_one:v18"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# Secret mount for GCP service account
gcp_sa_secret = Secret(
    deploy_type='volume',            # mount as volume
    deploy_target='/var/secrets/google',  # mount path in container (was 'mount_point')
    secret='gcp-sa-secret',             # name of K8s secret
    key='gcp-key.json'               # key file name inside secret
)
env_sa = {"GOOGLE_APPLICATION_CREDENTIALS": "/var/secrets/google/gcp-key.json"}

# ----------------------------
# DAG
# ----------------------------
with DAG(
    "gcs_trigger_pipeline",
    default_args=default_args,
    schedule_interval="0 1 * * *",
    start_date=datetime(2025, 11, 26),
    catchup=False,
) as dag:

    # 1. List file parquet trên GCS
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket=BUCKET,
        prefix=PREFIX,
        gcp_conn_id='google_cloud_default',  # sẽ dùng GOOGLE_APPLICATION_CREDENTIALS
        do_xcom_push=True,
    )

    # 2. Check file mới nhất xem có phải ngày hôm nay không
    def check_file_date(**context):
        files = context["ti"].xcom_pull(task_ids="list_files")
        if not files:
            print("Không có file parquet.")
            return False
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            print("Không tìm thấy file parquet.")
            return False
        latest_file = sorted(parquet_files)[-1]
        print("File detect:", latest_file)
        match = re.search(FILE_PATTERN, latest_file)
        if not match:
            print("Không match tên file.")
            return False
        yyyy, mm, dd = match.groups()
        file_date = datetime(int(yyyy), int(mm), int(dd)).date()
        today = datetime.utcnow().date()
        print("File date:", file_date)
        print("Today:", today)
        return file_date == today

    check_file = ShortCircuitOperator(
        task_id="check_file",
        python_callable=check_file_date,
    )

    # ----------------------------
    # 3. Spark job 1
    # ----------------------------
    spark_job_1 = KubernetesPodOperator(
        task_id="spark_job_1",
        name="spark-job-1",
        namespace="serving",
        image=IMAGE,
        cmds=["/opt/spark/bin/spark-submit"],
        arguments=[
            "--master", "k8s://https://136.115.43.206:443",
            "--deploy-mode", "cluster",
        
            "--conf", "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "--conf", "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            "--conf", "spark.kubernetes.namespace=serving",
            "--conf", f"spark.kubernetes.container.image={IMAGE}", 
            "--conf", "spark.kubernetes.authenticate.driver.serviceAccountName=default",
            "--conf", "spark.kubernetes.driver.secrets.gcp-sa-secret=/var/secrets/google",  
            "--conf", "spark.kubernetes.executor.secrets.gcp-sa-secret=/var/secrets/google",
            "--conf", "spark.kubernetes.driverEnv.PYTHONPATH=/app:/app/modules",
            "--conf", "spark.kubernetes.executorEnv.PYTHONPATH=/app:/app/modules",
            
            # Set env var cho driver và executor
            "--conf", "spark.kubernetes.driverEnv.GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/gcp-key.json",
            "--conf", "spark.kubernetes.executorEnv.GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/gcp-key.json",
            # Giảm CPU
            "--conf", "spark.driver.cores=1",
            "--conf", "spark.executor.cores=2",
            "--conf", "spark.driver.memory=1g",
            "--conf", "spark.executor.memory=14g",
            "local:///app/spark_job/sampling.py",
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        secrets=[gcp_sa_secret],
        env_vars=env_sa,
    )

    # Spark job 2
    spark_job_2 = KubernetesPodOperator(
        task_id="spark_job_2",
        name="spark-job-2",
        namespace="serving",
        image=IMAGE,
        cmds=["/opt/spark/bin/spark-submit"],
        arguments=[
            "--master", "k8s://https://136.115.43.206:443",
            "--deploy-mode", "cluster",
        
            "--conf", "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "--conf", "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            "--conf", "spark.kubernetes.namespace=serving",
            "--conf", f"spark.kubernetes.container.image={IMAGE}",  
            "--conf", "spark.kubernetes.authenticate.driver.serviceAccountName=default",
            "--conf", "spark.kubernetes.driver.secrets.gcp-sa-secret=/var/secrets/google", 
            "--conf", "spark.kubernetes.executor.secrets.gcp-sa-secret=/var/secrets/google",
            "--conf", "spark.kubernetes.driverEnv.PYTHONPATH=/app:/app/modules",
            "--conf", "spark.kubernetes.executorEnv.PYTHONPATH=/app:/app/modules",
            
            # Set env var cho driver và executor
            "--conf", "spark.kubernetes.driverEnv.GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/gcp-key.json",
            "--conf", "spark.kubernetes.executorEnv.GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/gcp-key.json",
            # Giảm CPU
            "--conf", "spark.driver.cores=1",
            "--conf", "spark.executor.cores=2",
            "--conf", "spark.driver.memory=1g",
            "--conf", "spark.executor.memory=14g",

            "local:///app/spark_job/sampling_item.py",
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        secrets=[gcp_sa_secret],
        env_vars=env_sa,
    )


    upload_to_dbms = KubernetesPodOperator(
        task_id="upload_to_dbms",
        name="upload-to-dbms",
        namespace="serving",
        image=IMAGE,
        cmds=["python3"],
        arguments=["/app/spark_job/upload_to_dbms.py"],
        get_logs=True,
        is_delete_operator_pod=True,
        secrets=[gcp_sa_secret],
        env_vars=env_sa,
        service_account_name="default",
    )

    # ----------------------------
    # 5. dbt build
    # ----------------------------
    dbt_build = KubernetesPodOperator(
        task_id="dbt_build",
        name="dbt-build",
        namespace="serving",
        image=IMAGE,
        cmds=["dbt"],
        arguments=[
        "build",
        "--project-dir", "/app/dbt_project/kltn",
        "--profiles-dir", "/app/dbt_project/kltn" 
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        secrets=[gcp_sa_secret],
        env_vars=env_sa,
        service_account_name="default",
    )

    # ----------------------------
    # 6. Feast materialize incremental
    # ----------------------------
    feast_run = KubernetesPodOperator(
        task_id="feast_materialize",
        name="feast-materialize",
        namespace="serving",
        image=IMAGE,
        cmds=["sh", "-c"],
        arguments=[
            """
            cd /app/feature_repo && 
            CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S") &&
            feast materialize-incremental $CURRENT_TIME
            """
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        secrets=[gcp_sa_secret],
        env_vars=env_sa,
        service_account_name="default",
    )

    def trigger_kfp_training(**context):
        # 1. Download pipeline yaml từ GCS
        hook = GCSHook(gcp_conn_id="google_cloud_default")
        local_pipeline = "/tmp/training.yaml"

        hook.download(
            bucket_name="kltn--data",
            object_name="config/training_pipeline.yaml",
            filename=local_pipeline,
        )

        # 2. Trigger Kubeflow Pipeline
        client = Client(
            host="http://ml-pipeline.kubeflow.svc.cluster.local:8080"
        )

        today = datetime.utcnow().strftime("%Y-%m-%d")

        client.create_run_from_pipeline_package(
            pipeline_file=local_pipeline,
            run_name=f"recsys-train-{today}",
            arguments={
                "bucket_name": "kltn--data",
                "yaml_skipgram_gcs_path": "config/skipgramptjob.yaml",
                "yaml_ranker_gcs_path": "config/rankingptjob.yaml",
            },
        )
    
    trigger_training = PythonOperator(
        task_id="trigger_training_pipeline",
        python_callable=trigger_kfp_training,
        retries=2,
        retry_delay=timedelta(minutes=5),
    )

    # ----------------------------
    # Pipeline
    # ----------------------------
    list_files >> check_file >> spark_job_1 >> upload_to_dbms >> dbt_build >> feast_run >> spark_job_2 >> trigger_training