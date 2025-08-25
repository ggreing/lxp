from fastapi import APIRouter, HTTPException
from bson import ObjectId
from datetime import datetime
from qdrant_client.http.models import PointStruct
from typing import List, Dict, Any
import csv, io, math

from ..config import settings
from ..db import inst_db, ensure_indexes
from .. import storage, vector, rabbitmq
from ..rag_utils import chunk_text, embed_texts, DIM
from ..schemas import JobRequest, JobResponse

router = APIRouter(
    tags=["coach"],
)

# RAG document storage endpoints (from vectorstores.py)

@router.post("/documents")
async def create_vectorstore():
    """Creates a new document collection (vector store)."""
    await ensure_indexes(settings.app_org_id)
    db = inst_db(settings.app_org_id)
    doc = {"files": [], "created_at": datetime.utcnow()}
    res = await db["vectorstore"].insert_one(doc)
    return {"id": str(res.inserted_id), "created_at": doc["created_at"].isoformat() + "Z"}

@router.get("/documents/{vectorstore_id}")
async def get_vectorstore(vectorstore_id: str):
    """Gets metadata for a document collection."""
    try:
        oid = ObjectId(vectorstore_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid vectorstore id")
    db = inst_db(settings.app_org_id)
    vs = await db["vectorstore"].find_one({"_id": oid}, {"files": {"$slice": 0}})
    if not vs:
        raise HTTPException(status_code=404, detail="Vectorstore not found")
    return {"id": vectorstore_id, "created_at": vs.get("created_at")}

def _safe_payload(v: Any):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

def build_course_text(row: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "course_title","topic","learning_objectives","keywords",
        "prerequisites","instructor_name","language","difficulty",
        "target_audience","interactivity_level","accessibility_features"
    ]:
        val = row.get(key)
        if val is not None and str(val).strip():
            parts.append(f"{key.replace('_',' ')}: {val}")
    return "\n".join(parts)

def build_user_text(row: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "occupation","education_level","preferred_language",
        "preferred_learning_style","learning_goals","performance_trend",
        "experience_years","country","age","gender"
    ]:
        val = row.get(key)
        if val is not None and str(val).strip():
            parts.append(f"{key.replace('_',' ')}: {val}")
    return "\n".join(parts)

def parse_csv_bytes_to_rows(data: bytes) -> List[Dict[str, Any]]:
    text = data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = [{k: r[k] for k in r.keys()} for r in reader]
    return rows

def detect_csv_kind(headers: List[str]) -> str:
    hset = set(h.lower() for h in headers)
    if {"course_id","course_title"}.issubset(hset):
        return "course"
    if {"user_id","preferred_language"}.issubset(hset):
        return "user"
    return "generic"

async def _index_csv_file(vs_id: str, meta: Dict[str, Any], max_points: int = 20000) -> int:
    data = storage.get_object_bytes(meta["object_name"])
    rows = parse_csv_bytes_to_rows(data)
    if not rows:
        return 0

    kind = detect_csv_kind(list(rows[0].keys()))
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    MAX_PAYLOAD_TEXT = 16000

    for i, row in enumerate(rows):
        if kind == "course":
            text = build_course_text(row)
            base_payload = {
                "type": "course",
                "course_id": row.get("course_id"),
                "course_title": row.get("course_title"),
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "keywords": row.get("keywords"),
                "learning_objectives": row.get("learning_objectives"),
                "prerequisites": row.get("prerequisites"),
                "interactivity_level": row.get("interactivity_level"),
                "target_audience": row.get("target_audience"),
                "accessibility_features": row.get("accessibility_features"),
            }
        elif kind == "user":
            text = build_user_text(row)
            base_payload = {
                "type": "user",
                "user_id": row.get("user_id"),
                "age": row.get("age"),
                "gender": row.get("gender"),
                "country": row.get("country"),
                "occupation": row.get("occupation"),
                "experience_years": row.get("experience_years"),
                "education_level": row.get("education_level"),
                "preferred_language": row.get("preferred_language"),
                "preferred_learning_style": row.get("preferred_learning_style"),
                "learning_goals": row.get("learning_goals"),
                "performance_trend": row.get("performance_trend"),
                "average_feedback_score": row.get("average_feedback_score"),
            }
        else:
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
            base_payload = {"type": "generic"}

        if not text or not text.strip():
            text = str(row)

        clean = text.strip()
        if len(clean) > MAX_PAYLOAD_TEXT:
            base_payload["text"] = clean[:MAX_PAYLOAD_TEXT]
            base_payload["text_truncated"] = True
        else:
            base_payload["text"] = clean
            base_payload["text_truncated"] = False
        base_payload["text_len"] = len(clean)

        base_payload["file_hash"] = meta.get("file_hash")
        base_payload["filename"] = meta.get("filename")
        base_payload["row_index"] = i

        payloads.append({k: _safe_payload(v) for k, v in base_payload.items()})
        texts.append(text)

        if len(texts) >= max_points:
            break

    if not texts:
        return 0

    BATCH = 128
    points: List[PointStruct] = []
    pid = 0
    for start in range(0, len(texts), BATCH):
        vecs = await embed_texts(texts[start:start+BATCH])
        for vec, pl in zip(vecs, payloads[start:start+BATCH]):
            points.append(PointStruct(id=pid, vector=vec, payload=pl))
            pid += 1

    if not points:
        return 0

    await vector.upsert_points(vs_id, points, dim=DIM, dedup=True)
    return len(points)

async def _index_plain_text_file(vs_id: str, meta: Dict[str, Any], max_points: int = 20000) -> int:
    data = storage.get_object_bytes(meta["object_name"])
    text = data.decode("utf-8", errors="ignore")
    chunks = chunk_text(text, chunk_size=600, overlap=120)
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks][:max_points]
    vecs = await embed_texts(texts)
    points = []
    for pid, (c, v) in enumerate(zip(chunks, vecs)):
        points.append(
            PointStruct(
                id=pid,
                vector=v,
                payload={
                    "file_hash": meta.get("file_hash"),
                    "filename": meta.get("filename"),
                    "chunk_start": c["start"],
                    "chunk_end": c["end"],
                    "text": c["text"],
                },
            )
        )
        if len(points) >= max_points:
            break
    if not points:
        return 0
    await vector.upsert_points(vs_id, points, dim=DIM, dedup=True)
    return len(points)

@router.post("/documents/{vectorstore_id}/index")
async def index_vectorstore(vectorstore_id: str, max_points: int = 20000):
    """Indexes all files in a document collection."""
    await ensure_indexes(settings.app_org_id)

    try:
        oid = ObjectId(vectorstore_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid vectorstore id")

    db = inst_db(settings.app_org_id)
    vs = await db["vectorstore"].find_one({"_id": oid})
    if not vs:
        raise HTTPException(status_code=404, detail="Vectorstore not found")

    total_indexed = 0
    for meta in vs.get("files", []) or []:
        filename = (meta.get("filename") or "").lower()
        ctype = (meta.get("content_type") or "").lower()

        try:
            if filename.endswith(".csv") or "text/csv" in ctype:
                total_indexed += await _index_csv_file(vectorstore_id, meta, max_points=max_points)
            elif filename.endswith(".md") or "text/markdown" in ctype or filename.endswith(".txt") or "text/plain" in ctype:
                total_indexed += await _index_plain_text_file(vectorstore_id, meta, max_points=max_points)
            else:
                continue
        except Exception:
            continue

    return {"indexed": total_indexed, "collection": f"vs_{vectorstore_id}"}


# Course recommendation endpoint (from assist.py)

@router.post("/recommendations", response_model=JobResponse)
async def recommend_courses(req: JobRequest):
    """Triggers a course recommendation task."""
    FUNCTION_NAME = "coach"
    ROUTING_KEY = "coach.request"

    await ensure_indexes(settings.app_org_id)
    db = inst_db(settings.app_org_id)
    doc = {
        "user_id": req.user_id,
        "title": (req.prompt or FUNCTION_NAME)[:64],
        "function_name": FUNCTION_NAME,
        "details": [],
        "create_timestamp": datetime.utcnow(),
        "last_timestamp": datetime.utcnow(),
    }
    res = await db["threads"].insert_one(doc)
    thread_id = str(res.inserted_id)

    payload = {
        "job_id": JobResponse().job_id,
        "org_id": settings.app_org_id,
        "user_id": req.user_id,
        "thread_id": thread_id,
        "function_name": FUNCTION_NAME,
        "sub_function": req.sub_function or "recommendation",
        "vectorstore_id": req.vectorstore_id,
        "params": req.params,
        "files": req.files or [],
        "prompt": req.prompt,
        "created_at": datetime.utcnow().isoformat(),
    }
    await rabbitmq.publish_task(ROUTING_KEY, payload)
    return JobResponse(job_id=payload["job_id"], thread_id=thread_id, status_url=f"/events/jobs/{payload['job_id']}")
