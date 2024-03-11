export MLFLOW_TRACKING_URI=http://100.65.79.57
export MLFLOW_S3_ENDPOINT_URL=http://100.65.79.57:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
mlflow run . -P pipeline_steps='download_data' --experiment-name dl_model_chapter06
