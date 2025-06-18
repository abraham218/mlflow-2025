MLflow is an open-source platform designed to manage the life-cycle of machine learning models. It addresses tasks such as experiment tracking, model deployment, and includes a central model registry. It’s like having a virtual assistant that ensures nothing is lost and keeps experiments organized.

we need to create Docker containers with PostgreSQL, Minio S3, and MLflow.
Let’s create a new network for our containers:

docker network create mlflow-network  #  No need for docker-compose 

docker-compose up -d --build

# Create a bucket called mlfow in Minio 
to verify if the bucket can be listed, run the below command from jupyter

import boto3
s3 = boto3.client(
    's3',
    endpoint_url='http://172.19.175.179:9000',  # ✅ S3 API port
    aws_access_key_id='mlflow',
    aws_secret_access_key='NewStrong2025!',
    #region_name='us-east-1'  # optional but common default
)

# List all buckets
print(s3.list_buckets())

# Once the Bucket is listed we can configure the env from jupyter 
import os
#http://172.19.175.179:5000/
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://172.19.175.179:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "NewStrong2025!"

# Finally run the train scripts to load the experiments in MLflow server and the artifacts are loaded in Minio
