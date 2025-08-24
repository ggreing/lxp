import json, aio_pika
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..rabbitmq import connect, RESULTS_EXCHANGE

router = APIRouter()

@router.get("/jobs/{job_id}")
async def sse_job(job_id: str):
    async def gen():
        conn = await connect()
        ch = await conn.channel()
        ex = await ch.declare_exchange(RESULTS_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True)
        q = await ch.declare_queue("", exclusive=True, durable=False, auto_delete=True)
        await q.bind(ex, routing_key="#")
        try:
            async with q.iterator() as it:
                async for msg in it:
                    async with msg.process():
                        try:
                            payload = json.loads(msg.body.decode())
                        except Exception:
                            continue
                        if payload.get("job_id") == job_id:
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            try:
                await q.unbind(ex, routing_key="#"); await q.delete(if_unused=False, if_empty=False)
            except Exception:
                pass
            await ch.close(); await conn.close()
    return StreamingResponse(gen(), media_type="text/event-stream")
