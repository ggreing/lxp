import httpx
import os

API_URL = os.getenv("API_URL", "http://api:8000")

async def run(payload: dict) -> dict:
    prompt = payload.get("prompt") or ""
    vs_id = payload.get("vectorstore_id")

    if not vs_id or not prompt:
        return {"answer": "Vector store ID and prompt are required.", "evidence": []}

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{API_URL}/sales/rag/query",
                json={
                    "prompt": prompt,
                    "vectorstore_id": vs_id,
                    "top_k": 3
                }
            )
            response.raise_for_status()
            result = response.json()
            result["sub_function"] = payload.get("sub_function") or "qa"
            return result
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        return {"answer": error_message, "evidence": []}
    except Exception as e:
        return {"answer": f"An unexpected error occurred: {str(e)}", "evidence": []}
