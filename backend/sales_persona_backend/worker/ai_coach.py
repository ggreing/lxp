import httpx
import os

# The API service is on the same Docker network
API_URL = os.getenv("API_URL", "http://api:8000")

async def run(payload: dict) -> dict:
    """
    Handles the 'coach' task by calling the central RAG query endpoint.
    """
    prompt = payload.get("prompt")
    vs_id = payload.get("vectorstore_id")

    if not vs_id or not prompt:
        return {"answer": "Vector store ID and prompt are required to get a recommendation.", "evidence": []}

    try:
        # Use a context manager for the client for proper resource management
        async with httpx.AsyncClient(timeout=300) as client:
            # Proxy the request to the main API's RAG endpoint
            response = await client.post(
                f"{API_URL}/sales/rag/query",
                json={
                    "prompt": prompt,
                    "vectorstore_id": vs_id,
                    "top_k": 3  # Use a reasonable default for top_k
                }
            )
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Return the JSON response from the RAG service
            result = response.json()
            return result

    except httpx.HTTPStatusError as e:
        # Provide a more informative error message if the API call fails
        error_message = f"API call failed: {e.response.status_code} - {e.response.text}"
        return {"answer": error_message, "evidence": []}
    except Exception as e:
        # Catch any other unexpected errors
        return {"answer": f"An unexpected error occurred: {str(e)}", "evidence": []}
