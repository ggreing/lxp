from fastapi import APIRouter
from ..schemas import JobRequest, JobResponse
from ..config import settings
from .. import rabbitmq
from ..db import inst_db, ensure_indexes
from datetime import datetime

router = APIRouter()
FUNCTION_NAME = "coach"
ROUTING_KEY = "coach.request"

@router.post("", response_model=JobResponse)
async def enqueue(req: JobRequest):
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
        "sub_function": req.sub_function or "qa",
        "vectorstore_id": req.vectorstore_id,
        "params": req.params,
        "files": req.files or [],
        "prompt": req.prompt,
        "created_at": datetime.utcnow().isoformat(),
    }
    await rabbitmq.publish_task(ROUTING_KEY, payload)
    return JobResponse(job_id=payload["job_id"], thread_id=thread_id, status_url=f"/events/jobs/{payload['job_id']}")
