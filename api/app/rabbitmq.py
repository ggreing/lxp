import json, aio_pika
from .config import settings

TASKS_EXCHANGE = "ai.tasks"
RESULTS_EXCHANGE = "ai.results"
DLQ_EXCHANGE = "ai.dlq"
CHAT_MESSAGES_EXCHANGE = "chat.messages"
CHAT_RESPONSES_EXCHANGE = "chat.responses"

QUEUE_BINDINGS = {
    "q.assist": ["assist.*"],
    "q.galaxy": ["galaxy.*"],
    "q.coach": ["coach.*"],
    "q.translate": ["translate.*"],
    "q.sim.control": ["sim.control.*"],
}

async def connect():
    return await aio_pika.connect_robust(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        login=settings.rabbitmq_user,
        password=settings.rabbitmq_password,
        virtualhost=settings.rabbitmq_vhost,
    )

async def ensure_topology():
    conn = await connect()
    async with conn:
        ch = await conn.channel()
        await ch.set_qos(prefetch_count=8)

        ex_tasks = await ch.declare_exchange(TASKS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
        ex_results = await ch.declare_exchange(RESULTS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
        ex_dlq = await ch.declare_exchange(DLQ_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)
        ex_chat_msg = await ch.declare_exchange(CHAT_MESSAGES_EXCHANGE, aio_pika.ExchangeType.DIRECT, durable=True)
        ex_chat_resp = await ch.declare_exchange(CHAT_RESPONSES_EXCHANGE, aio_pika.ExchangeType.FANOUT, durable=True)

        # Queue for chat messages to be consumed by the sales service
        q_chat = await ch.declare_queue("q.chat.messages", durable=True)
        await q_chat.bind(ex_chat_msg, routing_key="request")

        for qname, routes in QUEUE_BINDINGS.items():
            q = await ch.declare_queue(qname, durable=True, arguments={"x-dead-letter-exchange": DLQ_EXCHANGE})
            for rk in routes:
                await q.bind(ex_tasks, routing_key=rk)

        # DLQ queue bind (fixed)
        q_dlq = await ch.declare_queue("q.dlq", durable=True)
        await q_dlq.bind(ex_dlq)

async def publish_task(routing_key: str, payload: dict):
    conn = await connect()
    async with conn:
        ch = await conn.channel()
        ex = await ch.declare_exchange(TASKS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
        body = json.dumps(payload).encode("utf-8")
        msg = aio_pika.Message(body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT, content_type="application/json")
        await ex.publish(msg, routing_key=routing_key)
        return True

async def publish_chat_message(payload: dict):
    conn = await connect()
    async with conn:
        ch = await conn.channel()
        ex = await ch.declare_exchange(CHAT_MESSAGES_EXCHANGE, aio_pika.ExchangeType.DIRECT, durable=True)
        body = json.dumps(payload).encode("utf-8")
        msg = aio_pika.Message(body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT, content_type="application/json")
        await ex.publish(msg, routing_key="request")
        return True
