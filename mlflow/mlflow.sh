#!/bin/bash

docker build -t mlflow:latest .
docker tag mlflow:latest us-docker.pkg.dev/cool-ascent-428211-r9/mlflow-repo/mlflow:latest
gcloud run deploy mlflow-server   \
    --image us-docker.pkg.dev/cool-ascent-428211-r9/mlflow-repo/mlflow:latest   \
    --region us-central1   \
    --platform managed   \
    --no-allow-unauthenticated   \
    --set-env-vars BACKEND_STORE_URI=$BACKEND_STORE_URI,ARTIFACTS_DESTINATION=$ARTIFACTS_DESTINATION    \
    --memory 1Gi