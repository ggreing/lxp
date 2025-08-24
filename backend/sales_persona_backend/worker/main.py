# worker/main.py
import os
import json
import asyncio
import signal
from typing import Any, Dict

import aio_pika
from aio_pika import IncomingMessage

from sales_persona_backend import rabbitmq

try:
    from . import ai_assist, ai_galaxy, ai_coach, ai_translate, ai_sim
except ImportError as e:
    print(f"Could not import AI modules: {e}")
    ai_assist = ai_galaxy = ai_coach = ai_translate = ai_sim = None


async def _dispatch_task(routing_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    prefix = routing_key.split(".", 1)[0] if routing_key else ""

    if prefix == "assist" and ai_assist and hasattr(ai_assist, "run"):
        return await ai_assist.run(payload)
    if prefix == "galaxy" and ai_galaxy and hasattr(ai_galaxy, "run"):
        return await ai_galaxy.run(payload)
    if prefix == "coach" and ai_coach and hasattr(ai_coach, "run"):
        return await ai_coach.run(payload)
    if prefix == "translate" and ai_translate and hasattr(ai_translate, "run"):
        return await ai_translate.run(payload)
    if prefix == "sim" and ai_sim and hasattr(ai_sim, "control"):
        return await ai_sim.control(payload)

    return {"ok": True, "echo": True, "routing_key": routing_key, "input": payload}


async def handle_message(channel: Any, message: IncomingMessage):
    async with message.process(requeue=False):
        body_text = message.body.decode("utf-8") if message.body else "{}"
        try:
            payload = json.loads(body_text or "{}")
        except Exception as e:
            await rabbitmq.publish_result(channel, "task.failed", {
                "error": f"invalid_json: {e}", "raw": body_text
            })
            return

        rk_in = message.routing_key or payload.get("task_routing_key") or "unknown"
        job_id = payload.get("job_id") or payload.get("id")

        try:
            result = await _dispatch_task(rk_in, payload)
            await rabbitmq.publish_result(channel, "task.succeeded", {
                "job_id": job_id, "routing_key": rk_in, "status": "succeeded", "result": result
            })
        except Exception as e:
            await rabbitmq.publish_result(channel, "task.failed", {
                "job_id": job_id, "routing_key": rk_in, "status": "failed", "error": str(e)
            })
            return


async def run_task_consumer(shutdown_event: asyncio.Event):
    """Connects to RabbitMQ and consumes worker tasks until shutdown_event is set."""
    connection = await rabbitmq.get_rabbitmq_connection()

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=rabbitmq.WORKER_PREFETCH)

        # The declare_topology function now returns all queues.
        # We need to get the worker-specific queues.
        all_queues = await rabbitmq.declare_topology(channel)
        worker_queues = {
            name: queue for name, queue in all_queues.items()
            if name in ["assist", "galaxy", "coach", "translate", "sim"]
        }

        print(f" [x] Waiting for worker tasks on queues: {list(worker_queues.keys())}")

        async def consume_queue(queue: aio_pika.abc.AbstractQueue):
            async for message in queue:
                if shutdown_event.is_set():
                    await message.nack(requeue=True)
                    break
                # Fire and forget message handling
                asyncio.create_task(handle_message(channel, message))

        consumer_tasks = [
            asyncio.create_task(consume_queue(q)) for q in worker_queues.values()
        ]

        # Wait for the shutdown event to be set
        await shutdown_event.wait()

        # Gracefully stop consumers
        print("Shutdown event received, stopping worker task consumers...")
        for task in consumer_tasks:
            task.cancel()

        # Wait for all consumer tasks to finish cancelling
        await asyncio.gather(*consumer_tasks, return_exceptions=True)
        print("Worker task consumers stopped.")
