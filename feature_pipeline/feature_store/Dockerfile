FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./feature_repo ./feature_repo

ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json
ENV PYTHONUNBUFFERED=1

WORKDIR /app/feature_repo

CMD ["feast", "serve", "--host", "0.0.0.0", "--port", "6566"]
