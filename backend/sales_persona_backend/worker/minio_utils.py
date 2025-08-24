from minio import Minio
from .config import settings
import io

_client = None
def client():
    global _client
    if _client is None:
        host, port = settings.minio_endpoint.split(":")
        _client = Minio(host + ":" + port, access_key=settings.minio_root_user, secret_key=settings.minio_root_password, secure=settings.minio_secure)
        if not _client.bucket_exists(settings.minio_bucket):
            _client.make_bucket(settings.minio_bucket)
    return _client

def put_text(key: str, text: str, content_type="text/plain"):
    c = client()
    data = text.encode("utf-8")
    c.put_object(settings.minio_bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)
    return f"{settings.minio_bucket}/{key}"
