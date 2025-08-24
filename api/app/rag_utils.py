import re, math, os
from typing import List, Dict
import httpx

# Qdrant 벡터 차원(설정값 우선)
DIM = int(os.getenv("QDRANT_DIM", "768"))

# 원격 임베딩 엔드포인트 (있으면 우선 시도)
# 예: SALES_BACKEND_URL=http://sales:8001  → http://sales:8001/embeddings 로 POST
EMBED_API = (os.getenv("EMBEDDINGS_URL")
             or os.getenv("SALES_BACKEND_URL")
             or "").strip()

async def _remote_embed(texts: List[str]) -> List[List[float]]:
    if not EMBED_API:
        raise RuntimeError("No remote embeddings endpoint configured")
    url = EMBED_API.rstrip("/") + "/embeddings"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json={"texts": texts})
        r.raise_for_status()
        data = r.json()
        vecs = data.get("embeddings", [])
        if not isinstance(vecs, list):
            raise RuntimeError("Invalid embeddings payload")
        return vecs

def _hash_embed_one(text: str, dim: int = DIM) -> List[float]:
    """
    외부 임베딩이 없을 때를 위한 가벼운 폴백:
    토큰을 해시해 고정 차원 벡터에 누적하고 L2 정규화.
    (의미 임베딩은 아니지만 파이프라인/검색은 정상 동작)
    """
    if not text:
        return [0.0] * dim
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return [0.0] * dim

    vec = [0.0] * dim
    for t in toks:
        # 파이썬 내장 hash는 실행마다 바뀔 수 있으므로, 안정성을 위해 간단한 수작업 해시
        h = 0
        for ch in t:
            h = (h * 131 + ord(ch)) & 0x7fffffff
        idx = h % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    1) EMBED_API 설정되어 있으면 원격 임베딩 시도
    2) 실패/미설정 시 로컬 해시 임베딩 폴백
    """
    if not texts:
        return []
    # 1) 원격 시도
    if EMBED_API:
        try:
            return await _remote_embed(texts)
        except Exception:
            # 폴백으로 내려감
            pass
    # 2) 로컬 폴백
    return [_hash_embed_one(t) for t in texts]

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[Dict]:
    text = text or ""
    out = []; i = 0; n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append({"text": text[i:j], "start": i, "end": j})
        if j == n:
            break
        i = max(0, j - overlap)
    return out
