from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from sales_persona_backend.personas import random_persona as get_random_persona, PRESET_PERSONAS
from datetime import datetime


router = APIRouter()

custom_persona_store = {}  # session_id: persona 형식으로 저장 (임시)

class PersonaInput(BaseModel):
    personality: str
    age_group: str
    gender: str
    tech: str
    goal: str
    type: str
    usage: str

@router.post("/persona/custom")
async def set_custom_persona(data: PersonaInput, request: Request):
    client_id = request.client.host  # 간단한 대안 (세션 관리 미구현 시)
    custom_persona_store[client_id] = data.dict()
    return {"message": "custom persona saved"}


@router.get("/persona")
async def get_persona(request: Request):
    # 전체 페르소나 목록 반환
    return PRESET_PERSONAS

# --- Navigation for frontend menu ---
@router.get("/")
async def home():
    return {"message": "홈"}

@router.get("/routes")
async def routes():
    return {
        "home": "/",
        "create_persona": "/create-persona",
        "chat": "/chat",
        "analyze": "/analyze/[sessionId]",
        "admin": "/admin/personas",
        "stats": "/stats"
    }




class SaveSessionRequest(BaseModel):
    user_id: str
    score: int
    persona: PersonaInput

@router.post("/chat/save_session")
async def save_session(request: SaveSessionRequest):
    print("[BACKEND] 세션 저장 요청 수신:", request.dict())

    # 실제로는 DB 저장 로직이 들어가야 하지만, 여기선 단순하게 처리
    session_id = f"{request.user_id}-{datetime.utcnow().isoformat()}"
    return {"session_id": session_id}

from sales_persona_backend.log_manager import PersonalLogManager
import uuid

@router.post("/api/save_session")
async def save_session(request: SaveSessionRequest):
    """
    random 페르소나, user_id, score를 DB에 저장
    """
    log_mgr = PersonalLogManager()
    # 세션 ID 생성
    session_id = f"{request.user_id}-{uuid.uuid4()}"
    persona_type = request.persona.type
    scenario = request.persona.goal  # goal을 시나리오로 사용
    # 세션 시작 로그
    log_mgr.log_session_start(request.user_id, session_id, persona_type, scenario)
    # 세션 종료(점수/피드백) 저장 (feedback은 score만 저장, 필요시 확장)
    log_mgr.log_session_end(session_id, request.score, feedback="")
    return {"ok": True, "user_id": request.user_id, "session_id": session_id}