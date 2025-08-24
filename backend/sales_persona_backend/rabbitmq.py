import aio_pika
import os
import json

CHAT_MESSAGES_EXCHANGE = "chat.messages"
CHAT_RESPONSES_EXCHANGE = "chat.responses"
CHAT_QUEUE_NAME = "q.chat.messages"

async def get_rabbitmq_connection() -> aio_pika.Connection:
    return await aio_pika.connect_robust(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        port=int(os.getenv("RABBITMQ_PORT", 5672)),
        login=os.getenv("RABBITMQ_USER", "guest"),
        password=os.getenv("RABBITMQ_PASSWORD", "guest"),
        virtualhost=os.getenv("RABBITMQ_VHOST", "/"),
    )

async def publish_chat_response(channel: aio_pika.abc.AbstractChannel, session_id: str, chunk: str, event: str = "message"):
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

    # Fanout exchange doesn't use routing keys, but we can use it for metadata if needed
    await exchange.publish(message, routing_key=session_id)
