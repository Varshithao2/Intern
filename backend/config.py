import os

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "your-access-key")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "your-secret-key")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "credit-card-fraud-bucket")
MODEL_KEY = os.getenv("MODEL_KEY", "model/model.joblib")

# Local model cache path
MODEL_LOCAL_PATH = "model.joblib"
