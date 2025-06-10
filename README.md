docker build -t mlflow-server . # Build the Docker image
docker run -itd -p 5000:5000 -v $(pwd)/artifacts:/mlflow/artifacts mlflow-server # Run the container
http://localhost:5000 # Access the MLflow UI

FOR Tracking the MLFlow Server
run the file one by first_diabetes.py and second_diabetes.py

Docker compose
docker-compose up --build -d  or docker-compose up -d
