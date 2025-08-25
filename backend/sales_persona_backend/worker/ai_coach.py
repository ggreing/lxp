from . import rag

async def run(payload: dict) -> dict:
    """
    Handles the 'coach' task by calling the internal RAG function.
    """
    prompt = payload.get("prompt")
    vs_id = payload.get("vectorstore_id")
    params = payload.get("params", {}) or {}

    if not vs_id or not prompt:
        return {"answer": "Vector store ID and prompt are required to get a recommendation.", "evidence": []}

    # Construct a new payload compatible with rag.run
    rag_payload = {
        "prompt": prompt,
        "vectorstore_id": vs_id,
        "top_k": params.get("top_k", 3),
        "filters": params.get("filters"),
    }

    try:
        # Directly call the internal RAG function with the new payload
        return await rag.run(rag_payload)

    except Exception as e:
        # Catch any unexpected errors from the RAG function
        return {"answer": f"An unexpected error occurred during RAG execution: {str(e)}", "evidence": []}
