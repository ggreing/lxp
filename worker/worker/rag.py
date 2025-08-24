import hashlib, re, math, csv, io
from typing import List, Dict, Any
from qdrant_client.http import models as qmodels
from .vector_utils import client as qclient, ensure_collection
from .config import settings

DIM = settings.qdrant_dim

def tokenize(t: str) -> List[str]:
    return re.findall(r"[\w\-]+", (t or "").lower())

def embed(text: str, dim: int = DIM) -> List[float]:
    # Simple bag-of-words hashing to a fixed-dimension vector for search compatibility.
    vec = [0.0] * dim
    for tok in tokenize(text):
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    n = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/n for v in vec]

def parse_text_from_bytes(data: bytes, filename: str = "", content_type: str | None = None) -> str:
    '''
    Extract plain text from csv, txt, md. Fallback: utf-8 decode.
    '''
    name = (filename or '').lower()
    ct = (content_type or '').lower()
    if name.endswith('.csv') or 'text/csv' in ct:
        try:
            text = data.decode('utf-8', errors='ignore')
            reader = csv.reader(io.StringIO(text))
            rows = []
            for row in reader:
                rows.append(' '.join(col.strip() for col in row if col is not None))
            return '\n'.join(rows)
        except Exception:
            return data.decode('utf-8', errors='ignore')
    if name.endswith('.md') or 'text/markdown' in ct:
        return data.decode('utf-8', errors='ignore')
    if name.endswith('.txt') or 'text/plain' in ct:
        return data.decode('utf-8', errors='ignore')
    # default
    return data.decode('utf-8', errors='ignore')

def search(vs_id: str, query: str, top_k: int = 5):
    ensure_collection(vs_id, DIM)
    qc = qclient()
    col = f"vs_{vs_id}"
    hits = qc.search(collection_name=col, query_vector=embed(query), limit=top_k, with_payload=True)
    out = []
    for h in hits:
        payload = h.payload or {}
        out.append({
            "score": float(getattr(h, "score", 0.0)),
            "text": payload.get("text") or payload.get("chunk_text") or "",
            "filename": payload.get("filename"),
            "file_id": payload.get("file_id"),
            "file_hash": payload.get("file_hash"),
            "start": payload.get("chunk_start"),
            "end": payload.get("chunk_end"),
        })
    return out
