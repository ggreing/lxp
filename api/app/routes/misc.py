from fastapi import APIRouter
from ..config import settings
from datetime import datetime
router = APIRouter()

@router.get("/healthz")
async def health():
    return {"ok": True, "org_id": settings.app_org_id, "ts": datetime.utcnow().isoformat()}
