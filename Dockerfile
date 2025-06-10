# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV MLFLOW_HOME=/mlflow
ENV MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory and artifact directory with proper permissions
RUN mkdir -p $MLFLOW_HOME/artifacts && \
    mkdir -p $MLFLOW_HOME/logs && \
    mkdir -p $MLFLOW_HOME/db && \
    chmod -R 777 $MLFLOW_HOME

WORKDIR $MLFLOW_HOME

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow[extras] \
    psutil \
    sqlalchemy

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Ensure artifact directory exists and has proper permissions\n\
mkdir -p $MLFLOW_ARTIFACT_ROOT\n\
chmod -R 777 $MLFLOW_ARTIFACT_ROOT\n\
\n\
# Start MLflow server with explicit artifact path\n\
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///$MLFLOW_HOME/db/mlflow.db \
    --artifacts-destination file://$MLFLOW_ARTIFACT_ROOT \
    --serve-artifacts \
    --workers 4' > /start.sh

RUN chmod +x /start.sh

# Expose the MLflow server port
EXPOSE 5000

# Start MLflow server
CMD ["/start.sh"]
