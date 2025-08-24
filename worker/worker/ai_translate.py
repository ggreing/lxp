async def run(payload: dict) -> dict:
    t = payload.get("prompt", "")
    return {"src": t, "dst": t[::-1]}
