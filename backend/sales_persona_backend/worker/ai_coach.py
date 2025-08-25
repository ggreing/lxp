import asyncio
from .. import ai

async def run(payload: dict) -> dict:
    """
    Handles the 'coach' task by calling the main RAG function from the parent ai module.
    """
    prompt = payload.get("prompt")
    vs_id = payload.get("vectorstore_id")
    params = payload.get("params", {}) or {}

    if not vs_id or not prompt:
        return {"answer": "Vector store ID and prompt are required for a recommendation.", "evidence": []}

    top_k = params.get("top_k", 3)

    try:
        # Run the synchronous answer_with_rag function in a separate thread
        # to avoid blocking the asyncio event loop.
        result = await asyncio.to_thread(
            ai.answer_with_rag,
            prompt=prompt,
            vector_store_id=vs_id,
            top_k=top_k
        )
        return result

    except Exception as e:
        return {"answer": f"An unexpected error occurred during RAG execution: {str(e)}", "evidence": []}
