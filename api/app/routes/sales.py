from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
from ..config import settings
from ..db import inst_db, ensure_indexes
import httpx, uuid, asyncio

router = APIRouter()
BASE = settings.sales_backend_url.rstrip("/")

class StartSessionRequest(BaseModel):
    user_id: str
    persona: dict | None = None
    session_id: str | None = None  # 없으면 생성

@router.post("/session")
async def start_session(req: StartSessionRequest):
    await ensure_indexes(settings.app_org_id)
    session_id = req.session_id or uuid.uuid4().hex

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{BASE}/chat/initiate", json={
            "session_id": session_id,
            "persona": req.persona
        })
        r.raise_for_status()
        greeting = r.json().get("message", "")

    db = inst_db(settings.app_org_id)
    now = datetime.utcnow()
    doc = {
        "user_id": req.user_id,
        "function_name": "sales",
        "sub_function": "sim",
        "created_at": now,
        "last_timestamp": now,
        "session_id": session_id,
    }
    res = await db["threads"].insert_one(doc)
    thread_id = str(res.inserted_id)
    await db["user_thread"].update_one(
        {"user_id": req.user_id},
        {"$set": {"thread_id": thread_id}},
        upsert=True
    )
    return {"session_id": session_id, "thread_id": thread_id, "greeting": greeting}

class ChatRequest(BaseModel):
    session_id: str
    seller_msg: str
    user_id: str | None = None
    thread_id: str | None = None

from .. import rabbitmq
from fastapi.responses import StreamingResponse
import aio_pika
import json

@router.get("/stream/{session_id}")
async def stream_chat_responses(session_id: str):
    """
    Creates a temporary queue, binds it to the chat fanout exchange,
    and streams messages to the client via SSE.
    """
    async def event_generator():
        conn = await rabbitmq.get_rabbitmq_connection()
        async with conn:
            ch = await conn.channel()
            # Declare an exclusive queue, it will be deleted when connection is closed
            queue = await ch.declare_queue(exclusive=True)
            exchange = await ch.declare_exchange(
                rabbitmq.CHAT_RESPONSES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
            )
            await queue.bind(exchange)

            try:
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            payload = json.loads(message.body.decode())
                            # Forward message to the client, filtering by session_id if needed
                            # Although fanout sends to all, we can double-check here
                            if payload.get("session_id") == session_id:
                                yield f"event: {payload.get('event', 'message')}\ndata: {json.dumps(payload.get('data'))}\n\n"
            except asyncio.CancelledError:
                # Handle client disconnection
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat")
async def post_chat_message(req: ChatRequest):
    """
    Publishes a user's chat message to the RabbitMQ queue for backend processing.
    """
    payload = {
        "session_id": req.session_id,
        "seller_msg": req.seller_msg,
        "user_id": req.user_id,
        "thread_id": req.thread_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    await rabbitmq.publish_chat_message(payload)
    return {"status": "message published"}

@router.get("/tts/live/{session_id}")
async def tts_live(session_id: str, language_code: str = "ko-KR", voice_name: str | None = None, speaking_rate: float = 1.18):
    """
    세일즈 백엔드 /tts/live/{session_id} 오디오 스트림 프록시 (audio/ogg; codecs=opus)
    """
    params = {"language_code": language_code, "speaking_rate": speaking_rate}
    if voice_name:
        params["voice_name"] = voice_name

    async def gen():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{BASE}/tts/live/{session_id}", params=params) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield chunk

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Disposition": "inline; filename=tts.ogg",
    }
    return StreamingResponse(gen(), media_type="audio/ogg", headers=headers)

@router.post("/stt")
async def stt(request: Request):
    """
    JSON(base64 오디오) 또는 multipart(file) 모두 지원 – 원 백엔드와 동일 형식으로 프록시
    """
    content_type = request.headers.get("content-type", "")
    async with httpx.AsyncClient(timeout=None) as client:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(400, detail="file required")
            files = {"file": (file.filename, await file.read(), file.content_type or "application/octet-stream")}
            data = {"language_code": form.get("language_code") or "ko-KR"}
            r = await client.post(f"{BASE}/stt", data=data, files=files)
        else:
            body = await request.json()
            r = await client.post(f"{BASE}/stt", json=body)
    try:
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e:
        raise HTTPException(status_code=r.status_code if "r" in locals() else 502, detail=str(e))

@router.get("/persona/random")
async def persona_random():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{BASE}/persona/random")
        r.raise_for_status()
        return JSONResponse(r.json())

@router.get("/scenarios")
async def scenarios():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{BASE}/scenarios")
        r.raise_for_status()
        return JSONResponse(r.json())

@router.post("/tts")
async def tts_proxy(request: Request):
    """
    Proxies the simple, non-streaming TTS request to the sales backend.
    """
    data = await request.json()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BASE}/tts", json=data)
        r.raise_for_status()
        return Response(content=r.content, media_type="audio/mpeg")

class RAGQueryRequest(BaseModel):
    prompt: str
    vectorstore_id: str
    top_k: int = 3

@router.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    """
    Proxies the one-shot RAG query to the sales backend.
    """
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{BASE}/rag/query", json=req.model_dump())
        r.raise_for_status()
        return JSONResponse(r.json())

class RAGEmbedRequest(BaseModel):
    texts: list[str]

@router.post("/rag/embed")
async def rag_embed(req: RAGEmbedRequest):
    """
    Proxies the document embedding request to the sales backend.
    """
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{BASE}/rag/embed", json=req.model_dump())
        r.raise_for_status()
        return JSONResponse(r.json())
