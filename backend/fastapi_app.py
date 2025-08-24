from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse, Response
import uuid
import os
import base64
import httpx
import json
import re
from typing import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import json
import redis.asyncio as redis
from datetime import datetime, timedelta, timezone

from google.cloud import speech
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest
from google.auth import default
from sentence_transformers import SentenceTransformer

import aio_pika
from sales_persona_backend.ai import SalesPersonaAI, SimpleChatbotAI, answer_with_rag, embed_documents
from sales_persona_backend.personas import random_persona, SCENARIOS, PERSONAS, PRESET_PERSONAS
from sales_persona_backend.router import router as main_router
from sales_persona_backend.tts import router as tts_router, clients as tts_clients, warmup_tts
from sales_persona_backend import rabbitmq
from sales_persona_backend.worker.main import run_task_consumer
from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
load_dotenv()

# -------------------------
# 전역 싱글톤 & 토큰 캐시
# -------------------------
# This will be populated by the lifespan manager
# FastAPI will automatically call the lifespan manager on startup and shutdown
# and the resources will be available in the app's state.
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager. Handles startup and shutdown events.
    - Initializes all singleton clients (Redis, HTTPX, Google, etc.)
    - Loads the embedding model once.
    - Starts and gracefully shuts down the RabbitMQ consumer.
    """
    # Startup
    print("Application starting up...")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    app_state["redis_client"] = await redis.from_url(f"redis://{redis_host}", encoding="utf-8", decode_responses=False)

    print("Loading embedding model...")
    app_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    app_state["google_credentials"] = get_google_credentials()

    # Initialize shared clients for TTS
    tts_clients.redis_client = app_state["redis_client"]
    tts_clients.access_token_func = get_access_token
    try:
        from google.cloud import texttospeech_v1 as texttospeech
        tts_clients.tts_async_client = texttospeech.TextToSpeechAsyncClient(credentials=app_state["google_credentials"])
    except Exception as e:
        print(f"Could not initialize TTS client: {e}")
        tts_clients.tts_async_client = None

    try:
        app_state["speech_async_client"] = speech.SpeechAsyncClient(credentials=app_state["google_credentials"])
    except Exception as e:
        print(f"Could not initialize Speech client: {e}")
        app_state["speech_async_client"] = None

    app_state["httpx_client"] = httpx.AsyncClient(timeout=30.0, http2=True)
    tts_clients.httpx_client = app_state["httpx_client"]

    # Non-blocking warmup
    asyncio.create_task(warmup_tts())

    # Start RabbitMQ consumers with graceful shutdown
    loop = asyncio.get_running_loop()
    app_state["shutdown_event"] = asyncio.Event()
    chat_consumer_task = loop.create_task(consume_chat_messages(app_state["shutdown_event"]))
    worker_consumer_task = loop.create_task(run_task_consumer(app_state["shutdown_event"]))
    app_state["consumer_tasks"] = [chat_consumer_task, worker_consumer_task]

    print("Application startup complete.")
    yield

    # Shutdown
    print("Application shutting down...")
    app_state["shutdown_event"].set()
    await asyncio.sleep(1) # Give consumers a moment to stop accepting new messages

    tasks = app_state.get("consumer_tasks", [])
    for task in tasks:
        task.cancel()

    # Wait for all tasks to be cancelled
    cancelled_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(cancelled_tasks):
        if isinstance(result, asyncio.CancelledError):
            print(f"Consumer task {i} was cancelled successfully.")
        elif isinstance(result, Exception):
            print(f"Consumer task {i} resulted in an error during shutdown: {result}")

    if app_state.get("redis_client"):
        await app_state["redis_client"].close()
    if app_state.get("httpx_client"):
        await app_state["httpx_client"].aclose()
    if tts_clients.tts_async_client and hasattr(tts_clients.tts_async_client, "transport"):
        await tts_clients.tts_async_client.transport.close()
    if app_state.get("speech_async_client") and hasattr(app_state["speech_async_client"], "transport"):
        await app_state["speech_async_client"].transport.close()
    print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(main_router)
app.include_router(tts_router)


# -------------------------
# 이전 전역 변수 참조를 app_state로 변경
# -------------------------
def get_redis_client():
    return app_state["redis_client"]

_GOOGLE_CREDENTIALS = property(lambda: app_state["google_credentials"])
_SPEECH_ASYNC_CLIENT = property(lambda: app_state["speech_async_client"])
_HTTPX_CLIENT = property(lambda: app_state["httpx_client"])
_EMBEDDING_MODEL = property(lambda: app_state["embedding_model"])

_TOKEN: str | None = None
_TOKEN_EXPIRY: datetime | None = None

def _to_aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

# Redis Keys
def RKEY_SESSION(session_id: str): return f"session:{session_id}"
def RKEY_SIMPLE_SESSION(session_id: str): return f"simple_session:{session_id}"
def RKEY_SESSION_CLOSED(session_id: str): return f"session_closed:{session_id}"
def RKEY_TTS_QUEUE(session_id: str): return f"tts_queue:{session_id}" # This will be a LIST
def RKEY_TTS_DONE_FLAG(session_id: str): return f"tts_done_flag:{session_id}" # This will be a Pub/Sub channel

async def get_session(session_id: str) -> SalesPersonaAI | None:
    data = await get_redis_client().get(RKEY_SESSION(session_id))
    if not data:
        return None
    state = json.loads(data)
    return SalesPersonaAI.from_dict(state, embedding_model=_EMBEDDING_MODEL)

async def set_session(session_id: str, session: SalesPersonaAI, ttl: int = 3600):
    state = session.to_dict()
    await get_redis_client().set(RKEY_SESSION(session_id), json.dumps(state), ex=ttl)

async def get_simple_session(session_id: str) -> SimpleChatbotAI | None:
    history_json = await get_redis_client().get(RKEY_SIMPLE_SESSION(session_id))
    if not history_json:
        return SimpleChatbotAI()
    history = json.loads(history_json)
    return SimpleChatbotAI(history=history)

async def set_simple_session(session_id: str, session: SimpleChatbotAI, ttl: int = 3600):
    await get_redis_client().set(RKEY_SIMPLE_SESSION(session_id), json.dumps(session.history), ex=ttl)

async def handle_chat_message(
    message: aio_pika.abc.AbstractIncomingMessage,
) -> None:
    """Callback function to process messages from the chat queue."""
    async with message.process():
        try:
            payload = json.loads(message.body.decode())
            session_id = payload.get("session_id")
            seller_msg = payload.get("seller_msg")

            if not session_id or not seller_msg:
                return

            engine = await get_session(session_id)
            if not engine:
                engine = SalesPersonaAI(
                    session_id=session_id,
                    user_id=payload.get("user_id"),
                    embedding_model=_EMBEDDING_MODEL
                )

        conn = await rabbitmq.get_rabbitmq_connection()
            async with conn, conn.channel() as channel:
                full_raw = ""
                # The engine's stream_response is a generator, not an async generator
                for chunk in engine.stream_response(seller_msg):
                    full_raw += chunk
                await rabbitmq.publish_chat_response(channel, session_id, chunk, event="message")
                    await get_redis_client().lpush(f"tts_queue:{session_id}", chunk)

                await set_session(session_id, engine)
                if "<대화 종료>" in full_raw:
                     await get_redis_client().set(RKEY_SESSION_CLOSED(session_id), "true")
                 await rabbitmq.publish_chat_response(channel, session_id, "Conversation ended by AI.", event="end")

                await get_redis_client().publish(f"tts_done_flag:{session_id}", "done")

        except Exception as e:
            print(f"Error processing chat message: {e}")

async def consume_chat_messages(shutdown_event: asyncio.Event):
    """Connects to RabbitMQ and consumes messages until shutdown_event is set."""
    connection = await rabbitmq.get_rabbitmq_connection()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)

        # Topology is now declared in the worker consumer startup,
        # but we need the queue object to iterate.
        # Let's ensure it's declared and get it.
        all_queues = await rabbitmq.declare_topology(channel)
        queue = all_queues.get("chat")

        if not queue:
            print("FATAL: Chat queue not found after topology declaration.")
            return

        print(" [x] Waiting for chat messages. To exit press CTRL+C")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if shutdown_event.is_set():
                    print("Shutdown event received, stopping consumer.")
                    await message.nack()
                    break
                # Fire and forget message handling
                asyncio.create_task(handle_chat_message(message))

        print("RabbitMQ consumer loop finished.")


@app.get("/")
def root():
    return {"message": "Sales Persona API active"}

def get_google_credentials():
    """환경변수/파일/ADC 순서로 자격증명 로드."""
    try:
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            return service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if service_account_file and os.path.exists(service_account_file):
            return service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        default_paths = [
            os.path.join(os.path.dirname(__file__), "..", "tts.json"),
            os.path.join(os.path.dirname(__file__), "tts.json"),
        ]
        for path in default_paths:
            if os.path.exists(path):
                return service_account.Credentials.from_service_account_file(
                    path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return credentials
    except Exception:
        return None

def get_access_token():
    """토큰 캐시: 만료 5분 전까진 재사용, 임박시에만 refresh (naive/aware 안전)."""
    global _GOOGLE_CREDENTIALS, _TOKEN, _TOKEN_EXPIRY
    cred = _GOOGLE_CREDENTIALS or get_google_credentials()
    if not cred:
        return None

    now = datetime.now(timezone.utc)
    exp = _to_aware_utc(_TOKEN_EXPIRY)

    if (not _TOKEN) or (not exp) or (exp - now <= timedelta(minutes=5)):
        try:
            cred.refresh(GoogleRequest())
            _TOKEN = cred.token
            exp_new = _to_aware_utc(getattr(cred, "expiry", None))
            _TOKEN_EXPIRY = exp_new or (now + timedelta(hours=1))
        except Exception:
            return None
    return _TOKEN


# ---------- STT (JSON base64 & multipart 통합) ----------
@app.post("/stt")
async def stt(req: Request):
    """
    두 가지 입력을 모두 지원:
    - JSON: {"audio": "<base64>", "language_code": "ko-KR"}
    - multipart/form-data: file=<업로드 파일>, language_code=ko-KR
    """
    content_type = req.headers.get("content-type", "")
    language_code = "ko-KR"

    if content_type.startswith("multipart/form-data"):
        form = await req.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(400, detail="audio file required")
        audio_bytes = await file.read()
        language_code = form.get("language_code") or language_code
        encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        sample_rate_hz = 48000
    else:
        data = await req.json()
        audio_base64 = data.get("audio")
        if not audio_base64:
            raise HTTPException(400, detail="audio required")
        audio_bytes = base64.b64decode(audio_base64)
        language_code = data.get("language_code", language_code)
        encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
        sample_rate_hz = 0

    try:
        client = _SPEECH_ASYNC_CLIENT or speech.SpeechAsyncClient(credentials=_GOOGLE_CREDENTIALS)
    except Exception as e:
        raise HTTPException(503, detail=f"Google Cloud STT 인증 오류: {str(e)}")

    audio = speech.RecognitionAudio(content=audio_bytes)
    config_kwargs = dict(language_code=language_code, encoding=encoding)
    if sample_rate_hz:
        config_kwargs["sample_rate_hertz"] = sample_rate_hz
    config = speech.RecognitionConfig(**config_kwargs)

    response = await client.recognize(config=config, audio=audio)
    text = "".join(result.alternatives[0].transcript for result in response.results)
    return {"text": text}

# ---------- TTS 상태 체크 ----------
@app.get("/tts/status")
async def tts_status():
    access_token = get_access_token()
    if not access_token:
        return {"enabled": False, "reason": "Google 인증 실패"}
    test_text = "테스트 문장입니다"
    client = _HTTPX_CLIENT or httpx.AsyncClient(timeout=10.0, http2=True)
    response = await client.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        json={
            "input": {"text": test_text},
            "voice": {"languageCode": "ko-KR", "name": "ko-KR-Chirp3-HD-Kore"},
            "audioConfig": {"audioEncoding": "MP3"}
        },
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    )
    if response.status_code == 200:
        return {"enabled": True, "method": "Google Cloud TTS"}
    return {"enabled": False, "status_code": response.status_code, "error": response.text}

# =====================
# AI 챗/세션 기능
# =====================

@app.post("/chat/initiate")
async def initiate_chat(req: Request):
    data = await req.json()
    session_id = data.get("session_id")
    persona = data.get("persona")
    if not session_id:
        raise HTTPException(400, detail="session_id required")

    ai = await get_session(session_id)
    if not ai:
        if not persona:
            persona = random_persona()
        ai = SalesPersonaAI(persona=persona)

    await get_redis_client().set(RKEY_SESSION_CLOSED(session_id), "false")

    greeting = ai.generate_first_greeting()
    ai._append_history("AI", greeting)

    await set_session(session_id, ai)

    return JSONResponse({"message": greeting})

# =====================
# 페르소나 관련
# =====================
@app.get("/persona/random")
def get_random_persona():
    return random_persona()

@app.get("/scenarios")
def get_scenarios():
    return SCENARIOS

@app.get("/persona")
def list_personas():
    return PRESET_PERSONAS + PERSONAS

@app.post("/persona")
async def add_persona(req: Request):
    data = await req.json()
    data['id'] = str(uuid.uuid4())
    PERSONAS.append(data)
    return data

@app.delete("/persona/{persona_id}")
def delete_persona(persona_id: str):
    idx = next((i for i, p in enumerate(PERSONAS) if p.get('id') == persona_id), None)
    if idx is not None:
        PERSONAS.pop(idx)
        return {"success": True}
    raise HTTPException(404, detail="Persona not found")

# =====================
# 퍼포먼스 분석 및 종료 여부
# =====================
@app.post("/analyze")
async def analyze(req: Request):
    data = await req.json()
    session_id = data.get("session_id")
    if not session_id:
        raise HTTPException(400, detail="session_id required")
    engine = await get_session(session_id)
    if not engine:
        raise HTTPException(404, detail="session not found")
    text, _score = engine.analyze_conversation()
    return PlainTextResponse(text)

@app.post("/autoclose")
async def autoclose(req: Request):
    data = await req.json()
    session_id = data.get("session_id")
    if not session_id:
        raise HTTPException(400, detail="session_id required")
    engine = await get_session(session_id)
    if not engine:
        raise HTTPException(404, detail="session not found")
    should_close, reason = engine.maybe_autoclose()
    return JSONResponse({"should_close": should_close, "reason": reason})

# =====================
# 일상 챗봇 엔드포인트
# =====================
@app.post("/chatbot")
async def chatbot(req: Request):
    data = await req.json()
    user_msg = data.get("message")
    session_id = data.get("session_id")
    if not user_msg:
        raise HTTPException(400, detail="message required")
    if not session_id:
        raise HTTPException(400, detail="session_id required")

    engine = await get_simple_session(session_id)
    if not engine:
        engine = SimpleChatbotAI()

    generator = engine.stream_response(user_msg)

    async def event_stream():
        try:
            for chunk in generator:
                yield f"{chunk}\n"
            # Save session at the end of the stream
            await set_simple_session(session_id, engine)
        except Exception as e:
            yield f"error: {str(e)}\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class RAGQueryRequest(BaseModel):
    prompt: str
    vectorstore_id: str
    top_k: int = 3

@app.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    """
    Performs a one-shot RAG query using the new embedding model.
    """
    result = answer_with_rag(req.prompt, req.vectorstore_id, req.top_k)
    return JSONResponse(result)

class RAGEmbedRequest(BaseModel):
    texts: List[str]

@app.post("/rag/embed")
async def rag_embed(req: RAGEmbedRequest):
    """
    Embeds a list of documents for indexing.
    """
    embeddings = embed_documents(req.texts)
    return JSONResponse({"embeddings": embeddings})
