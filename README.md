MLflow is an open-source platform designed to manage the life-cycle of machine learning models. It addresses tasks such as experiment tracking, model deployment, and includes a central model registry. It’s like having a virtual assistant that ensures nothing is lost and keeps experiments organized.

we need to create Docker containers with PostgreSQL, Minio S3, and MLflow.
Let’s create a new network for our containers:

docker network create mlflow-network  #  No need for docker-compose 

docker-compose up -d --build

# Create a bucket called mlfow in Minio 
to verify if the bucket can be listed, run the below command from jupyter
# !pip install boto3
import boto3
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # ✅ S3 API port
    aws_access_key_id='mlflow',
    aws_secret_access_key='NewStrong2025!',
    #region_name='us-east-1'  # optional but common default
)

# List all buckets
print(s3.list_buckets())

# Once the Bucket is listed we can configure the env from jupyter 
import os
#http://localhost:5000/
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "NewStrong2025!"

# Finally run the train scripts to load the experiments in MLflow server and the artifacts are loaded in Minio

import warnings
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# Set the remote MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Optional: Set experiment name (will be created if it doesn't exist)
mlflow.set_experiment("wine-quality-elasticnet")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

warnings.filterwarnings("ignore")
np.random.seed(40)

data = pd.read_csv("red-wine-quality.csv")

train, test = train_test_split(data, test_size=0.25, random_state=42)

train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = 0.6
l1_ratio = 0.5

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y.values)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y.values, predicted_qualities)

    print("ElasticNet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log the model
    mlflow.sklearn.log_model(lr, "model")
###############################################################################################
# check the artifacts in the Mlflow server and Minio
