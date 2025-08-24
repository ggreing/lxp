import re
import os
import json
import base64
import asyncio
import httpx
from typing import Dict, Tuple, AsyncIterator

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response

from google.cloud import texttospeech_v1 as texttospeech
import redis.asyncio as redis

# A simple dependency injection system for our clients
class TTSClients:
    def __init__(self):
        self.redis_client: redis.Redis = None
        self.tts_async_client: texttospeech.TextToSpeechAsyncClient = None
        self.httpx_client: httpx.AsyncClient = None
        self.access_token_func = None

clients = TTSClients()

def get_redis() -> redis.Redis:
    return clients.redis_client

def get_tts_async_client() -> texttospeech.TextToSpeechAsyncClient:
    return clients.tts_async_client

def get_httpx_client() -> httpx.AsyncClient:
    return clients.httpx_client

def get_access_token() -> str:
    return clients.access_token_func()

async def warmup_tts():
    """앱 기동 직후 짧은 스트리밍 합성으로 커넥션/모델 워밍업."""
    client = get_tts_async_client()
    if client is None:
        return
    try:
        if not hasattr(texttospeech.TextToSpeechAsyncClient, "streaming_synthesize"):
            return

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Chirp3-HD-Kore"),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.OGG_OPUS,
                sample_rate_hertz=16000,
                speaking_rate=1.18,
            ),
        )
        async def _reqs():
            yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
            yield texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text="워밍업입니다."))

        resp_stream = await client.streaming_synthesize(requests=_reqs())
        async for _ in resp_stream:
            break
    except Exception:
        pass

router = APIRouter()

# Helper functions (moved from fastapi_app.py)

def process_text_for_tts(text: str) -> tuple[str, dict]:
    emotions = {}
    def replace_brackets(match):
        bracket_content = match.group(1)
        emotion_keywords = {
            '웃음': 'laugh', '웃으며': 'laugh', '웃고': 'laugh', '미소': 'smile', '미소지으며': 'smile',
            '따뜻하게': 'warm', '따뜻한': 'warm', '차분하게': 'calm', '차분한': 'calm',
            '밝게': 'bright', '밝은': 'bright', '신중하게': 'careful', '신중한': 'careful',
            '열정적으로': 'passionate', '열정적인': 'passionate', '부드럽게': 'soft', '부드러운': 'soft',
            '강하게': 'strong', '강한': 'strong', '조용히': 'quiet', '조용한': 'quiet',
            '활발하게': 'energetic', '활발한': 'energetic', '진지하게': 'serious', '진지한': 'serious',
            '친근하게': 'friendly', '친근한': 'friendly', '신뢰감 있게': 'trustworthy', '신뢰감 있는': 'trustworthy'
        }
        for keyword, emotion in emotion_keywords.items():
            if keyword in bracket_content:
                emotions[emotion] = True
                break
        else: emotions['neutral'] = True
        return ""
    processed_text = re.sub(r'[\(（]([^\)）]*)[\)）]', replace_brackets, text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text, emotions

def build_audio_config_by_emotion(voice_name: str, emotions: dict | None) -> dict:
    emotions = emotions or {}
    cfg = {"audio_encoding": "MP3", "speaking_rate": 1.18, "volume_gain_db": 0.4}
    def set_pitch(val: float):
        if "Chirp3-HD" not in voice_name: cfg["pitch"] = val
    if 'laugh' in emotions or 'smile' in emotions: set_pitch(2.0)
    elif 'warm' in emotions or 'friendly' in emotions: set_pitch(1.0)
    elif 'calm' in emotions or 'quiet' in emotions: set_pitch(-1.0)
    elif 'bright' in emotions or 'energetic' in emotions: set_pitch(3.0)
    elif 'serious' in emotions or 'strong' in emotions: set_pitch(-2.0); cfg["speaking_rate"] = 0.95
    elif 'passionate' in emotions: set_pitch(2.0); cfg["speaking_rate"] = 1.25; cfg["volume_gain_db"] = 2.0
    return cfg

CHIRP3_HD_VOICES = {
    "ko": {"male": "ko-KR-Chirp3-HD-Fenrir", "female": "ko-KR-Chirp3-HD-Kore"},
    "en": {"male": "en-US-Chirp3-HD-Fenrir", "female": "en-US-Chirp3-HD-Kore"},
}

def get_voice_by_persona(persona: dict | None = None) -> tuple[str, str]:
    lang_code = "ko-KR"
    voice_gender = "female"
    if persona:
        persona_lang = persona.get("lang", "").lower()
        if persona_lang.startswith("en"): lang_code = "en-US"
        g = persona.get("gender", "").lower()
        if "남성" in g or "male" in g: voice_gender = "male"
    key = lang_code.split("-")[0]
    return lang_code, CHIRP3_HD_VOICES.get(key, CHIRP3_HD_VOICES["ko"])[voice_gender]

# Endpoints (moved from fastapi_app.py)

@router.post("/tts")
async def simple_tts(req: Request,
                     tts_client: texttospeech.TextToSpeechAsyncClient = Depends(get_tts_async_client)):
    if not tts_client:
        raise HTTPException(status_code=503, detail="Google Cloud TTS client not available")

    data = await req.json()
    text = data.get("text")
    if not text: raise HTTPException(400, detail="text required")

    processed_text, emotions = process_text_for_tts(text)
    language_code, voice_name = get_voice_by_persona(data.get("persona"))
    audio_config_dict = build_audio_config_by_emotion(voice_name, emotions)

    synthesis_input = texttospeech.SynthesisInput(text=processed_text)
    voice_params = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = texttospeech.AudioConfig(**audio_config_dict)

    try:
        response = await tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Cloud TTS client error: {str(e)}")


SENTENCE_RE = re.compile(r'(?<=[\.\?\!。！？])\s+|\n')

def _chunk_sentences(text: str, max_len: int = 160) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in parts:
        if cur_len + len(p) > max_len and cur:
            chunks.append(' '.join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p) + 1
    if cur: chunks.append(' '.join(cur))
    return chunks if chunks else [text]

@router.post("/tts/stream")
async def tts_stream(req: Request, client: texttospeech.TextToSpeechAsyncClient = Depends(get_tts_async_client)):
    data = await req.json()
    text = (data.get("text") or "").strip()
    if not text: raise HTTPException(400, detail="text는 필수입니다.")

    language_code = data.get("language_code", "ko-KR")
    voice_name = data.get("voice_name") or f"{language_code}-Chirp3-HD-Kore"
    speaking_rate = float(data.get("speaking_rate", 1.18))
    chunk_max_len = int(data.get("chunk_max_len", 160))
    processed_text = process_text_for_tts(text)[0]
    chunks = _chunk_sentences(processed_text, max_len=chunk_max_len)

    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name),
        streaming_audio_config=texttospeech.StreamingAudioConfig(
            audio_encoding=texttospeech.AudioEncoding.OGG_OPUS, sample_rate_hertz=16000, speaking_rate=speaking_rate
        ),
    )
    async def _request_iter():
        yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
        for c in chunks:
            yield texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text=c))

    resp_stream = await client.streaming_synthesize(requests=_request_iter())
    async def _audio_iter():
        try:
            async for resp in resp_stream:
                if getattr(resp, "audio_content", None): yield resp.audio_content
        except Exception as e: yield f"[stream error] {str(e)}".encode("utf-8", errors="ignore")
    return StreamingResponse(_audio_iter(), media_type="audio/ogg")

def RKEY_TTS_QUEUE(session_id: str): return f"tts_queue:{session_id}"
def RKEY_TTS_DONE_FLAG(session_id: str): return f"tts_done_flag:{session_id}"

@router.get("/tts/live/{session_id}")
async def tts_live(session_id: str, language_code: str = "ko-KR", voice_name: str | None = None, speaking_rate: float = 1.18,
                   client: texttospeech.TextToSpeechAsyncClient = Depends(get_tts_async_client),
                   redis: redis.Redis = Depends(get_redis)):
    if not voice_name: voice_name = f"{language_code}-Chirp3-HD-Kore"
    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name),
        streaming_audio_config=texttospeech.StreamingAudioConfig(
            audio_encoding=texttospeech.AudioEncoding.OGG_OPUS, sample_rate_hertz=16000, speaking_rate=float(speaking_rate)
        ),
    )
    async def _request_iter():
        yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
        pubsub = redis.pubsub()
        await pubsub.subscribe(RKEY_TTS_DONE_FLAG(session_id))
        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.01)
                if message and message['data'] == b'done': break
                item = await redis.brpop(RKEY_TTS_QUEUE(session_id), timeout=0.1)
                if item:
                    _, sentence_bytes = item
                    yield texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text=sentence_bytes.decode('utf-8')))
        finally:
            await pubsub.unsubscribe(RKEY_TTS_DONE_FLAG(session_id))

    resp_stream = await client.streaming_synthesize(requests=_request_iter())
    async def _audio_iter():
        try:
            async for resp in resp_stream:
                if getattr(resp, "audio_content", None): yield resp.audio_content
        except Exception as e: yield f"[stream error] {str(e)}".encode("utf-8", errors="ignore")
        finally: await redis.delete(RKEY_TTS_QUEUE(session_id), RKEY_TTS_DONE_FLAG(session_id))
    return StreamingResponse(_audio_iter(), media_type="audio/ogg")
