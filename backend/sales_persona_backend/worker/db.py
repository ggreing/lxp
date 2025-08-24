from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings

_client = None
def client():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client
def inst_db():
    return client()[f"{settings.mongo_db_institution_prefix}{settings.org_id}"]
