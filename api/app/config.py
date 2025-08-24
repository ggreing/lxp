from pydantic import BaseModel
import os

class Settings(BaseModel):
    rabbitmq_host: str = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbitmq_user: str = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    rabbitmq_vhost: str = os.getenv("RABBITMQ_VHOST", "/")

    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    mongo_db_og: str = os.getenv("MONGO_DB_OG", "og_keys")
    mongo_db_institution_prefix: str = os.getenv("MONGO_DB_INSTITUTION_PREFIX", "institution_")

    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_root_user: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_root_password: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "lxp-artifacts")
    minio_secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_dim: int = 768 # Default to new model's dimension

    sales_backend_url: str = os.getenv("SALES_BACKEND_URL", "http://sales:8001")

    app_org_id: str = os.getenv("APP_ORG_ID", "demo-org")
    app_secret_key: str = os.getenv("APP_SECRET_KEY", "devsecret")
    log_level: str = os.getenv("APP_LOG_LEVEL", "INFO")

settings = Settings()
