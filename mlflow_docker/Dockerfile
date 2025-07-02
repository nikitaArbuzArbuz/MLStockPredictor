FROM ghcr.io/mlflow/mlflow:latest

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && pip install --no-cache-dir psycopg2-binary \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*