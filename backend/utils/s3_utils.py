import boto3
from botocore.exceptions import NoCredentialsError
from backend.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, MODEL_KEY, MODEL_LOCAL_PATH

def download_model_from_s3():
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3.download_file(S3_BUCKET_NAME, MODEL_KEY, MODEL_LOCAL_PATH)
        print("✅ Model downloaded from S3.")
    except NoCredentialsError:
        print("❌ AWS credentials not found.")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
