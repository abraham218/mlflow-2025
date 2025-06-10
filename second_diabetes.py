import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Set MLflow server URI
mlflow.set_tracking_uri("http://172.19.175.179:5000")  # 🔁 Replace with your MLflow IP

# Create or set an experiment
mlflow.set_experiment("ridge-diabetes-model")

# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loop through different alpha values
for alpha in [0.1, 0.5, 1.0, 5.0]:
    with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, "ridge_model")

        # Save and log plot artifact
        plt.figure(figsize=(6,4))
        plt.scatter(y_test, preds, alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Alpha={alpha}")
        plot_file = f"pred_vs_actual_alpha_{alpha}.png"
        plt.savefig(plot_file)
        mlflow.log_artifact(plot_file)

        print(f"Logged run for alpha={alpha} | RMSE={rmse:.2f} | R²={r2:.2f}")

