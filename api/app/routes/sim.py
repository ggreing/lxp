from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

router = APIRouter()

@router.websocket("/session")
async def ws_session(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_json({"event": "sim.started", "ts": datetime.utcnow().isoformat()})
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"echo: {data}")
    except WebSocketDisconnect:
        pass
