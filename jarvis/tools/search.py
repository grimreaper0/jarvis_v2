"""Search tools for LangGraph agents."""
from langchain_core.tools import tool


@tool
def search_automem(query: str, limit: int = 5) -> list[dict]:
    """Search AutoMem (pgvector) for relevant past conversations and patterns.

    Args:
        query: Natural language search query
        limit: Max results to return (default 5)

    Returns:
        List of matching records with similarity scores
    """
    from jarvis.core.memory import AutoMem
    mem = AutoMem()
    return [{"result": f"AutoMem search for '{query}' — implement embedding-based search", "similarity": 0.0}]


@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for recent information about a topic.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results with title, url, snippet
    """
    return [{"title": f"Web search: {query}", "url": "#", "snippet": "Implement with DuckDuckGo/SerpAPI"}]
