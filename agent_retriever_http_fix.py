# Agent retriever implementation using Weaviate
import os
import weaviate


def ask_agent(question: str) -> str:
    """
    Handle questions for the agent by querying the Weaviate instance.
    Returns the most relevant piece of text from the knowledge base or a fallback message.
    """
    weaviate_url = os.getenv("WEAVIATE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not weaviate_url or not openai_api_key:
        return "Agent is not configured with Weaviate URL or OpenAI API key."
    try:
        client = weaviate.Client(
            url=weaviate_url,
            additional_headers={"X-OpenAI-Api-Key": openai_api_key},
        )
        # Query the knowledge base using a simple near text search. Adjust class and property names as needed.
        result = client.query.get("Document", ["content"]).with_near_text({"concepts": [question]}).with_limit(1).do()
        docs = result.get("data", {}).get("Get", {}).get("Document", [])
        if docs:
            return docs[0].get("content", "")
        else:
            return "I couldn't find an answer in the knowledge base."
    except Exception as e:
        return f"Error querying Weaviate: {e}"
