"""
This module is used where is no information at all or information is not enough
to answer user questions. It is also used to find real-time info, events and facts.
"""

from ddgs import DDGS
from langchain.tools import tool
import trafilatura

def extract_full_text(url: str, max_chars: int = 1500) -> str:
    if not url:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        full_text = trafilatura.extract(downloaded)
        return full_text[:max_chars] if full_text else ""
    except Exception:
        return ""

@tool
def search_on_web(query: str) -> str:
    """Search the web for real-time events, current facts, or information not found in uploaded documents.

    Args:
        query: The search query string.
    """
    try:
        with DDGS() as ddg_search:
            results = list(ddg_search.text(query, max_results=5))
            if not results:
                return "No relevant information found on web search"
            formatted = []
            for r in results:
                url = r.get("href", "")
                text = extract_full_text(url) or r.get("body", "")
                formatted.append(
                    f"Title: {r.get("title", "")},\nURL: {r.get("href", "")}\nText: {text}"
                )
            return "\n\n---\n\n".join(formatted)
    except Exception as exc:
        return f"Web search error: {exc}"