from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from .config import settings

_client = None

def client():
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

def ensure_collection(vs_id: str, dim: int | None = None):
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

def delete_points_by_payload(vs_id: str, where: dict, dim: int | None = None):
    name = ensure_collection(vs_id, dim)
    must = []
    for k, v in (where or {}).items():
        if v is None:
            continue
        must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    if not must:
        return 0
    f = Filter(must=must)
    res = client().delete(collection_name=name, points_selector=f)
    return getattr(res, "status", "ok")

def upsert_points(vs_id: str, points: list[PointStruct], dim: int | None = None):
    name = ensure_collection(vs_id, dim)
    client().upsert(collection_name=name, points=points)
    return name
