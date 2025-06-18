#!/usr/


import warnings
import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--l1_ratio", type=float, default=0.7)
args = parser.parse_args()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://172.19.175.179:5000")

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load data
    try:
        data = pd.read_csv("red-wine-quality.csv")
    except FileNotFoundError:
        logger.error("File 'red-wine-quality.csv' not found.")
        exit(1)

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split data
    train, test = train_test_split(data, test_size=0.25)
    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Create or get experiment
    experiment = mlflow.set_experiment("experiment_1")

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predict
        predicted_qualities = lr.predict(test_x)

        # Evaluate
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Output
        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R2: {r2:.3f}")

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(lr, "model")

