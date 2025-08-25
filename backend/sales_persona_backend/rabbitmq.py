# MERGED rabbitmq.py
import os
import json
import asyncio
from typing import Dict, Tuple, Optional, Any

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection

# ====== 환경설정 (Backend & Worker 통합) ======
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
# Worker의 URL 방식도 fallback으로 지원
RABBITMQ_URL = os.getenv("RABBITMQ_URL")

WORKER_PREFETCH = int(os.getenv("WORKER_PREFETCH", "10")) # Backend 기본값과 맞춤

# --- Backend Exchanges/Queues ---
CHAT_MESSAGES_EXCHANGE = "chat.messages"
CHAT_RESPONSES_EXCHANGE = "chat.responses"
CHAT_QUEUE_NAME = "q.chat.messages"

# --- Worker Exchanges/Queues ---
TASKS_EXCHANGE   = os.getenv("TASKS_EXCHANGE", "ai.tasks")
RESULTS_EXCHANGE = os.getenv("RESULTS_EXCHANGE", "ai.results")
DLX_EXCHANGE     = os.getenv("DLX_EXCHANGE", "ai.dlq") # FIX: Mismatch with existing queue property

Q_ASSIST     = os.getenv("Q_ASSIST",     "q.assist")
Q_GALAXY     = os.getenv("Q_GALAXY",     "q.galaxy")
Q_PICKS      = os.getenv("Q_PICKS",      "q.picks")
Q_TRANSLATE  = os.getenv("Q_TRANSLATE",  "q.translate")
Q_SIM        = os.getenv("Q_SIM",        "q.sim.control")

RK_ASSIST     = os.getenv("RK_ASSIST",     "assist.*")
RK_GALAXY     = os.getenv("RK_GALAXY",     "galaxy.*")
RK_PICKS      = os.getenv("RK_PICKS",      "picks.*")
RK_TRANSLATE  = os.getenv("RK_TRANSLATE",  "translate.*")
RK_SIM        = os.getenv("RK_SIM",        "sim.*")

DLQ_SUFFIX = ".dlq"


# ---------- 호환 레이어 (from Worker) ----------
async def _compat_declare_exchange(ch: Any, name: str, ex_type: ExchangeType, durable: bool = True):
    if hasattr(ch, "declare_exchange"):
        return await ch.declare_exchange(name, ex_type, durable=durable)
    if hasattr(ch, "exchange_declare"):
        ex_type_str = ex_type.value.lower() if hasattr(ex_type, "value") else str(ex_type).lower()
        return await ch.exchange_declare(exchange=name, exchange_type=ex_type_str, durable=durable)
    raise AttributeError("Channel has neither declare_exchange nor exchange_declare")

async def _compat_get_exchange(ch: Any, name: str):
    if hasattr(ch, "get_exchange"):
        try:
            return await ch.get_exchange(name)
        except Exception:
            return None
    return None

async def _compat_publish(exchange: Any, message: Message, routing_key: str):
    if hasattr(exchange, "publish"):
        return await exchange.publish(message, routing_key=routing_key)
    raise AttributeError("Exchange object has no publish()")


async def declare_topology(channel: Any) -> Dict[str, aio_pika.Queue]:
    """
    Backend와 Worker의 모든 교환기/큐/바인딩을 선언.
    """
    # === Backend Topology ===
    await _compat_declare_exchange(channel, CHAT_MESSAGES_EXCHANGE, ExchangeType.DIRECT, durable=True)
    await _compat_declare_exchange(channel, CHAT_RESPONSES_EXCHANGE, ExchangeType.FANOUT, durable=True)
    chat_queue = await channel.declare_queue(CHAT_QUEUE_NAME, durable=True)
    await chat_queue.bind(CHAT_MESSAGES_EXCHANGE, routing_key="request")

    # === Worker Topology ===
    await _compat_declare_exchange(channel, TASKS_EXCHANGE, ExchangeType.TOPIC, durable=True)
    await _compat_declare_exchange(channel, RESULTS_EXCHANGE, ExchangeType.TOPIC, durable=True)
    await _compat_declare_exchange(channel, DLX_EXCHANGE, ExchangeType.FANOUT, durable=True)

    args_with_dlx = {"x-dead-letter-exchange": DLX_EXCHANGE}

    # 큐
    q_assist = await channel.declare_queue(Q_ASSIST, durable=True, arguments=args_with_dlx)
    q_galaxy = await channel.declare_queue(Q_GALAXY, durable=True, arguments=args_with_dlx)
    q_picks  = await channel.declare_queue(Q_PICKS,  durable=True, arguments=args_with_dlx)
    q_trans  = await channel.declare_queue(Q_TRANSLATE, durable=True, arguments=args_with_dlx)
    q_sim    = await channel.declare_queue(Q_SIM, durable=True, arguments=args_with_dlx)

    # 바인딩
    await q_assist.bind(TASKS_EXCHANGE, RK_ASSIST)
    await q_galaxy.bind(TASKS_EXCHANGE, RK_GALAXY)
    await q_picks.bind(TASKS_EXCHANGE, RK_PICKS)
    await q_trans.bind(TASKS_EXCHANGE, RK_TRANSLATE)
    await q_sim.bind(TASKS_EXCHANGE, RK_SIM)

    # DLQ 큐들
    await channel.declare_queue(f"{Q_ASSIST}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_GALAXY}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_PICKS}{DLQ_SUFFIX}",  durable=True)
    await channel.declare_queue(f"{Q_TRANSLATE}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_SIM}{DLQ_SUFFIX}", durable=True)

    return {
        "chat": chat_queue,
        "assist": q_assist,
        "galaxy": q_galaxy,
        "picks": q_picks,
        "translate": q_trans,
        "sim": q_sim,
    }


async def get_rabbitmq_connection() -> aio_pika.Connection:
    """기존 Backend의 연결 방식을 우선 사용하고, Worker의 URL 방식을 fallback으로 지원."""
    if RABBITMQ_URL:
        return await aio_pika.connect_robust(RABBITMQ_URL)
    return await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        virtualhost=RABBITMQ_VHOST,
    )

# --- Publishing Functions ---

async def publish_chat_response(channel: aio_pika.abc.AbstractChannel, session_id: str, chunk: str, event: str = "message"):
    """Backend의 원본 함수. Fanout exchange에 메시지 발행."""
    exchange = await channel.declare_exchange(
        CHAT_RESPONSES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True
    )
    message_body = json.dumps({
        "session_id": session_id,
        "event": event,
        "data": chunk
    }).encode('utf-8')
    message = aio_pika.Message(
        body=message_body,
        content_type='application/json',
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
    )
    await exchange.publish(message, routing_key=session_id)


async def publish_result(channel: Any, routing_key: str, payload: dict):
    """Worker의 원본 함수. Topic exchange에 결과 발행."""
    ex = await _compat_get_exchange(channel, RESULTS_EXCHANGE)
    if ex is None:
        ex = await _compat_declare_exchange(channel, RESULTS_EXCHANGE, ExchangeType.TOPIC, durable=True)
        tmp = await _compat_get_exchange(channel, RESULTS_EXCHANGE)
        if tmp is not None:
            ex = tmp

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    msg = Message(
        body=body,
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
    )
    await _compat_publish(ex, msg, routing_key=routing_key)
