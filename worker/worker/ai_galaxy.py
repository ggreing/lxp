async def run(payload: dict) -> dict:
    items = ["A", "B", "C"]
    return {"pick": items, "selected": items[0]}
