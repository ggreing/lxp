# worker/rabbitmq.py
import os
import json
import asyncio
from typing import Dict, Tuple, Optional, Any

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection

# ====== 환경설정 ======
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
WORKER_PREFETCH = int(os.getenv("WORKER_PREFETCH", "8"))

TASKS_EXCHANGE   = os.getenv("TASKS_EXCHANGE", "ai.tasks")
RESULTS_EXCHANGE = os.getenv("RESULTS_EXCHANGE", "ai.results")
DLX_EXCHANGE     = os.getenv("DLX_EXCHANGE", "ai.dlq")

Q_ASSIST     = os.getenv("Q_ASSIST",     "q.assist")
Q_GALAXY     = os.getenv("Q_GALAXY",     "q.galaxy")
Q_COACH      = os.getenv("Q_COACH",      "q.coach")
Q_TRANSLATE  = os.getenv("Q_TRANSLATE",  "q.translate")
Q_SIM        = os.getenv("Q_SIM",        "q.sim.control")

RK_ASSIST     = os.getenv("RK_ASSIST",     "assist.*")
RK_GALAXY     = os.getenv("RK_GALAXY",     "galaxy.*")
RK_COACH      = os.getenv("RK_COACH",      "coach.*")
RK_TRANSLATE  = os.getenv("RK_TRANSLATE",  "translate.*")
RK_SIM        = os.getenv("RK_SIM",        "sim.*")

DLQ_SUFFIX = ".dlq"


# ---------- 호환 레이어 ----------
async def _compat_declare_exchange(ch: Any, name: str, ex_type: ExchangeType, durable: bool = True):
    """
    aio-pika v9~11: channel.declare_exchange(...)
    aio-pika v12+ or aiormq-style: channel.exchange_declare(...)
    """
    if hasattr(ch, "declare_exchange"):  # aio-pika <=11
        return await ch.declare_exchange(name, ex_type, durable=durable)
    # fallback (aio-pika 12 / aiormq 스타일)
    ex_type_str = ex_type.value.lower() if hasattr(ex_type, "value") else str(ex_type).lower()
    # 일부 구현은 positional-only거나 키워드-only 이슈가 있으므로 키워드로 시도
    if hasattr(ch, "exchange_declare"):
        return await ch.exchange_declare(
            exchange=name,
            exchange_type=ex_type_str,
            durable=durable,
        )
    raise AttributeError("Channel has neither declare_exchange nor exchange_declare")

async def _compat_get_exchange(ch: Any, name: str):
    """
    aio-pika 일부 버전에만 get_exchange가 있음. 없으면 None 반환.
    """
    if hasattr(ch, "get_exchange"):
        try:
            return await ch.get_exchange(name)
        except Exception:
            return None
    return None

async def _compat_publish(exchange: Any, message: Message, routing_key: str):
    """
    exchange.publish(...)는 aio-pika 공통. 만약 exchange 객체가 없으면 예외.
    """
    if hasattr(exchange, "publish"):
        return await exchange.publish(message, routing_key=routing_key)
    raise AttributeError("Exchange object has no publish()")


async def _declare_topology(channel: Any) -> Dict[str, aio_pika.Queue]:
    """
    교환기/큐/바인딩을 시작 시 1회 선언.
    """
    # 교환기
    await _compat_declare_exchange(channel, TASKS_EXCHANGE,   ExchangeType.TOPIC, durable=True)
    await _compat_declare_exchange(channel, RESULTS_EXCHANGE, ExchangeType.TOPIC, durable=True)
    await _compat_declare_exchange(channel, DLX_EXCHANGE,     ExchangeType.FANOUT, durable=True)

    args_with_dlx = {"x-dead-letter-exchange": DLX_EXCHANGE}

    # 큐
    q_assist = await channel.declare_queue(Q_ASSIST, durable=True, arguments=args_with_dlx)
    q_galaxy = await channel.declare_queue(Q_GALAXY, durable=True, arguments=args_with_dlx)
    q_coach  = await channel.declare_queue(Q_COACH,  durable=True, arguments=args_with_dlx)
    q_trans  = await channel.declare_queue(Q_TRANSLATE, durable=True, arguments=args_with_dlx)
    q_sim    = await channel.declare_queue(Q_SIM, durable=True, arguments=args_with_dlx)

    # 바인딩
    # aio-pika Queue.bind(exchange, routing_key) – exchange는 이름/객체 모두 허용되도록 구현됨
    await q_assist.bind(TASKS_EXCHANGE,   RK_ASSIST)
    await q_galaxy.bind(TASKS_EXCHANGE,   RK_GALAXY)
    await q_coach.bind(TASKS_EXCHANGE,    RK_COACH)
    await q_trans.bind(TASKS_EXCHANGE,    RK_TRANSLATE)
    await q_sim.bind(TASKS_EXCHANGE,      RK_SIM)

    # DLQ 큐들(옵션)
    await channel.declare_queue(f"{Q_ASSIST}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_GALAXY}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_COACH}{DLQ_SUFFIX}",  durable=True)
    await channel.declare_queue(f"{Q_TRANSLATE}{DLQ_SUFFIX}", durable=True)
    await channel.declare_queue(f"{Q_SIM}{DLQ_SUFFIX}", durable=True)

    return {
        "assist": q_assist,
        "galaxy": q_galaxy,
        "coach": q_coach,
        "translate": q_trans,
        "sim": q_sim,
    }


async def connect_robust() -> Tuple[AbstractRobustConnection, Any, Dict[str, aio_pika.Queue]]:
    """
    Robust 연결 + 채널 + QoS + 토폴로지 선언.
    반환 채널은 aio-pika v9~12 / aiormq 스타일 모두 수용 가능한 Any로 둠.
    """
    conn: AbstractRobustConnection = await aio_pika.connect_robust(RABBITMQ_URL)
    ch = await conn.channel()  # RobustChannel or Channel (버전에 따라)
    if hasattr(ch, "set_qos"):
        await ch.set_qos(prefetch_count=WORKER_PREFETCH)
    queues = await _declare_topology(ch)
    return conn, ch, queues


async def publish_result(channel: Any, routing_key: str, payload: dict):
    """
    결과를 ai.results로 발행. 채널/메서드 차이를 흡수.
    """
    # exchange 객체를 받아보되, 없으면 선언
    ex = await _compat_get_exchange(channel, RESULTS_EXCHANGE)
    if ex is None:
        ex = await _compat_declare_exchange(channel, RESULTS_EXCHANGE, ExchangeType.TOPIC, durable=True)

        # 일부 구현은 declare가 exchange 객체를 반환하지 않을 수 있음 → exchange 객체를 다시 얻기
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


async def ack(message: aio_pika.IncomingMessage):
    await message.ack()


async def nack_or_dlx(message: aio_pika.IncomingMessage, requeue: bool = False):
    await message.reject(requeue=requeue)
