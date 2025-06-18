MLflow is an open-source platform designed to manage the life-cycle of machine learning models. It addresses tasks such as experiment tracking, model deployment, and includes a central model registry. It’s like having a virtual assistant that ensures nothing is lost and keeps experiments organized.

we need to create Docker containers with PostgreSQL, Minio S3, and MLflow.
Let’s create a new network for our containers:

docker network create mlflow-network  #  No need for docker-compose 

docker-compose up -d --build


