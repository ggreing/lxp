import os
class Settings:
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
    rabbitmq_vhost = os.getenv("RABBITMQ_VHOST", "/")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    mongo_db_institution_prefix = os.getenv("MONGO_DB_INSTITUTION_PREFIX", "institution_")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_root_user = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_root_password = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    minio_bucket = os.getenv("MINIO_BUCKET", "lxp-artifacts")
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_dim = 768
    org_id = os.getenv("APP_ORG_ID", "demo-org")
    prefetch = int(os.getenv("WORKER_PREFETCH", "8"))
settings = Settings()
