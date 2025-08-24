import boto3
import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")

def download_model_from_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    local_path = "./model.joblib"
    s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, local_path)
    return local_path

# Call this function once before loading your model
model_path = download_model_from_s3()
