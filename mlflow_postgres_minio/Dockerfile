FROM ghcr.io/mlflow/mlflow

USER root

# Install dependencies
RUN pip install boto3 psycopg2-binary

# Create and populate the credentials file using environment variables
ENV AWS_ACCESS_KEY_ID=mlflow
ENV AWS_SECRET_ACCESS_KEY=NewStrong2025!
ENV MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:Strong2025!@mlflow-postgres:5432/mlflow
ENV MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio:9000

# scenario 4
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow

CMD [ \
    "mlflow", "server", \
    "--no-serve-artifacts", \
    "--host", "0.0.0.0", \
    "--workers", "20" \
]
