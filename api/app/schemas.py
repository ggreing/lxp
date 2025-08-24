from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid

def new_job_id() -> str:
    return uuid.uuid4().hex

class JobRequest(BaseModel):
    user_id: str
    prompt: Optional[str] = None
    params: Dict[str, Any] = {}
    vectorstore_id: Optional[str] = None
    files: Optional[List[str]] = None
    sub_function: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str = Field(default_factory=new_job_id)
    thread_id: Optional[str] = None
    status_url: Optional[str] = None
