FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8080 

CMD mlflow server \
     --backend-store-uri ${BACKEND_STORE_URI} \
     --artifacts-destination ${ARTIFACTS_DESTINATION} \
     --host 0.0.0.0 \
     --port 8080



