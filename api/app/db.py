from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from .config import settings

_mongo_client = None

def client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.mongo_uri)
    return _mongo_client

def og_db():
    return client()[settings.mongo_db_og]

def inst_db(org_id: str):
    name = f"{settings.mongo_db_institution_prefix}{org_id}"
    return client()[name]

async def ensure_indexes(org_id: str):
    # --- OG DB 인덱스 ---
    og = og_db()
    await og["og_keys"].create_index("id", unique=True)
    await og["og_keys"].create_index("key", unique=True)

    # --- 기관 DB 인덱스 ---
    db = inst_db(org_id)
    await db["user_thread"].create_index("user_id", unique=True)
    await db["threads"].create_index([("user_id", 1), ("last_timestamp", -1)])
    await db["threads"].create_index("function_name")
    await db["ai_log"].create_index([("thread_id", 1), ("timestamp", 1)])
    await db["ai_log"].create_index("job_id")
    await db["vectorstore"].create_index("created_at")

    # ✅ vectorstore.files.file_hash 중복 자동 정리(빈 값 제외)
    collection = db["vectorstore"]
    pipeline = [
        {"$unwind": {"path": "$files", "preserveNullAndEmptyArrays": False}},
        {"$match": {"files.file_hash": {"$exists": True, "$type": "string", "$ne": ""}}},
        {"$group": {"_id": "$files.file_hash", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
    ]
    async for doc in collection.aggregate(pipeline):
        duplicate_ids = doc["ids"][1:]
        if duplicate_ids:
            result = await collection.delete_many({"_id": {"$in": duplicate_ids}})
            print(f"[CLEANUP] Removed {result.deleted_count} duplicate(s) for hash {doc['_id']}")

    # ✅ 부분 인덱스: 오래된 Mongo에서 허용되는 형태로 ($exists 만 사용)
    try:
        await collection.create_index(
            [("files.file_hash", 1)],
            name="uniq_files_file_hash",
            unique=True,
            partialFilterExpression={"files.file_hash": {"$exists": True}},
        )
        print("[INFO] Unique index (partial) ensured: files.file_hash")
    except DuplicateKeyError as e:
        print(f"[WARNING] Duplicate file_hash remains during index creation: {e.details}")
    except Exception as e:
        print(f"[ERROR] Failed to ensure index for vectorstore: {e}")
