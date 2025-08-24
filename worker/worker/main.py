# worker/main.py
import os
import json
import asyncio
import signal
from typing import Any, Dict

import aio_pika
from aio_pika import IncomingMessage

from worker import rabbitmq

try:
    from worker import ai_assist, ai_galaxy, ai_coach, ai_translate, ai_sim
except Exception:
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
    if prefix == "sim" and ai_sim and hasattr(ai_sim, "run"):
        return await ai_sim.run(payload)

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


async def run() -> None:
    conn, ch, qs = await rabbitmq.connect_robust()

    # 채널 타입 로그(디버깅 용) – 원하면 제거 가능
    try:
        import sys
        print(f"[worker] aio_pika={aio_pika.__version__} channel_cls={ch.__class__.__name__}", file=sys.stderr)
    except Exception:
        pass

    for _, q in qs.items():
        await q.consume(lambda m: handle_message(ch, m), no_ack=False)

    stop_event = asyncio.Event()

    def _stop(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await ch.close()
    await conn.close()


if __name__ == "__main__":
    asyncio.run(run())
