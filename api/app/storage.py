from minio import Minio
from .config import settings
from datetime import timedelta
import io

_client = None

def client() -> Minio:
    global _client
    if _client is None:
        endpoint = settings.minio_endpoint  # e.g., "minio:9000"
        _client = Minio(
            endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_secure,
        )
        # Ensure bucket exists
        if not _client.bucket_exists(settings.minio_bucket):
            _client.make_bucket(settings.minio_bucket)
    return _client

def put_object_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    c = client()
    data_stream = io.BytesIO(data)
    c.put_object(
        settings.minio_bucket,
        object_name,
        data_stream,
        length=len(data),
        content_type=content_type,
    )
    return f"{settings.minio_bucket}/{object_name}"

def get_object_bytes(object_name: str) -> bytes:
    c = client()
    res = c.get_object(settings.minio_bucket, object_name)
    try:
        return res.read()
    finally:
        res.close()
        res.release_conn()

def presigned_get(object_name: str, expires_seconds: int = 3600) -> str:
    c = client()
    return c.presigned_get_object(
        settings.minio_bucket,
        object_name,
        expires=timedelta(seconds=expires_seconds),
    )
