# api/app/routes/files.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from ..config import settings
from ..db import inst_db, ensure_indexes
from .. import storage, vector
from ..rag_utils import DIM
from datetime import datetime
import time, hashlib, os

router = APIRouter()

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@router.post("/upload")
async def upload_file(
    user_id: str,
    vectorstore_id: str | None = None,
    file: UploadFile = File(...)
):
    await ensure_indexes(settings.app_org_id)
    db = inst_db(settings.app_org_id)

    # 1) 파일 로드 & 저장
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    safe = os.path.basename(file.filename or "upload.bin")
    object_name = f"uploads/{settings.app_org_id}/{int(time.time() * 1000)}_{safe}"

    # MinIO 등에 실제 바이트 저장
    storage.put_object_bytes(
        object_name,
        content,
        content_type=file.content_type or "application/octet-stream",
    )

    file_hash = sha256(content)
    meta = {
        "user_id": user_id,
        "filename": safe,
        "file_hash": file_hash,
        "file_size": len(content),
        "uploaded_at": datetime.utcnow(),
        "object_name": object_name,
        "content_type": file.content_type or "application/octet-stream",
    }

    # 2) 대상 벡터스토어 검증 / 생성
    if vectorstore_id:
        try:
            oid = ObjectId(vectorstore_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid vectorstore id")
        target_id = vectorstore_id
    else:
        # 벡터스토어가 없으면 새 문서 생성하면서 파일 포함
        res = await db["vectorstore"].insert_one({"files": [meta], "created_at": datetime.utcnow()})
        return {"vectorstore_id": str(res.inserted_id), "meta": meta}

    # 3) 전역 중복 정리 (요구사항: 기존 데이터 지우고 새로 업로드)
    #    - 같은 file_hash를 가진 다른 모든 벡터스토어 문서들에서 파일 메타 제거
    #    - 각 문서의 Qdrant 컬렉션(= vs_<문서ID>)에서 해당 해시 포인트도 제거
    cursor = db["vectorstore"].find({"files.file_hash": file_hash}, {"_id": 1})
    async for doc in cursor:
        other_id = str(doc["_id"])
        if other_id == target_id:
            continue
        # files 배열에서 제거
        await db["vectorstore"].update_one(
            {"_id": doc["_id"]},
            {"$pull": {"files": {"file_hash": file_hash}}}
        )
        # Qdrant 포인트 제거
        try:
            await vector.delete_points_by_payload(other_id, {"file_hash": file_hash}, dim=DIM)
        except Exception:
            # 삭제 실패해도 업로드는 계속 진행
            pass

    # 4) 대상 벡터스토어 내에서도 같은 해시가 있으면 교체(기존 엔트리 제거 후 새 메타 삽입)
    await db["vectorstore"].update_one(
        {"_id": oid},
        {"$pull": {"files": {"file_hash": file_hash}}}
    )

    try:
        res = await db["vectorstore"].update_one(
            {"_id": oid},
            {"$push": {"files": meta}}
        )
        if res.matched_count == 0:
            raise HTTPException(status_code=404, detail="Vectorstore not found")
    except DuplicateKeyError:
        # 혹시 동시성으로 인해 유니크 인덱스 충돌이 나면 한 번 더 정리 후 재시도
        await db["vectorstore"].update_many(
            {"files.file_hash": file_hash},
            {"$pull": {"files": {"file_hash": file_hash}}}
        )
        await db["vectorstore"].update_one(
            {"_id": oid},
            {"$push": {"files": meta}}
        )

    return {"vectorstore_id": target_id, "meta": meta}
