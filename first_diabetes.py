import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd

# Explicitly set the MLflow tracking server URI
mlflow.set_tracking_uri("http://192.168.1.100:5000")  # 🔁 Replace with your actual IP address

# Optional: Set experiment name
mlflow.set_experiment("ridge-diabetes-model")

# Load data
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter
alpha = 0.5

with mlflow.start_run():
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # Log to remote MLflow server
    mlflow.log_param("alpha", alpha)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "ridge_model")

    print(f"Logged to MLflow Tracking Server at http://192.168.1.100:5000")
    print(f"RMSE: {rmse}")

