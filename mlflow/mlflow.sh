docker build -t mlflow-server:latest .
docker run -d \
  -p 8080:8080 \
  -e BACKEND_STORE_URI="$BACKEND_STORE_URI" \
  -e ARTIFACTS_DESTINATION="$ARTIFACTS_DESTINATION" \
  -e MLFLOW_SERVER_ALLOWED_HOSTS="*" \
  --name mlflow \
  mlflow-server