import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from .config import settings

_client: Optional[QdrantClient] = None

def client() -> QdrantClient:
    global _client
    if _client is None:
        # ✅ HTTP 고정(6333), gRPC 비활성화
        _client = QdrantClient(
            url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
            prefer_grpc=False,
            timeout=30,
        )
    return _client

def ping_qdrant() -> None:
    try:
        client().get_collections()
    except Exception as e:
        raise RuntimeError(f"[QDRANT] Connection failed to {settings.qdrant_host}:{settings.qdrant_port} -> {e}")

def _ensure_collection_sync(vs_id: str, dim: int | None = None):
    if dim is None:
        dim = settings.qdrant_dim
    name = f"vs_{vs_id}"
    c = client()
    ping_qdrant()
    cols = [col.name for col in c.get_collections().collections]
    if name in cols:
        try:
            info = c.get_collection(name)
            cfg = getattr(info, "config", None)
            params = getattr(cfg, "params", None)
            vectors = getattr(params, "vectors", None)
            size = getattr(vectors, "size", None)
        except Exception:
            size = None
        if size is not None and size != dim:
            c.delete_collection(name)
            c.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    else:
        c.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    return name

async def ensure_collection(vs_id: str, dim: int | None = None):
    return await asyncio.to_thread(_ensure_collection_sync, vs_id, dim)

def _delete_points_by_payload_sync(collection_name: str, where: Dict[str, Any]):
    must = []
    for k, v in (where or {}).items():
        if v is None:
            continue
        must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    if not must:
        return None
    f = Filter(must=must)
    return client().delete(collection_name=collection_name, points_selector=f)

async def delete_points_by_payload(vs_id: str, where: Dict[str, Any], dim: int | None = None):
    name = await ensure_collection(vs_id, dim)
    return await asyncio.to_thread(_delete_points_by_payload_sync, name, where)

def _upsert_points_sync(collection_name: str, points: List[PointStruct]):
    return client().upsert(collection_name=collection_name, points=points)

async def upsert_points(vs_id: str, points: List[PointStruct], dim: int | None = None, dedup: bool = True):
    name = await ensure_collection(vs_id, dim)
    if dedup and points:
        payload = getattr(points[0], "payload", None) or {}
        key = None
        for k in ("file_hash", "file_id", "filename"):
            if k in payload:
                key = k
                break
        if key:
            await delete_points_by_payload(vs_id, {key: payload.get(key)}, dim=dim)
    await asyncio.to_thread(_upsert_points_sync, name, points)
    return name
