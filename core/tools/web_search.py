"""Shared web search tool for agents."""

from __future__ import annotations

import os
import httpx


async def web_search(query: str, limit: int = 10) -> dict:
    """Search the web using Serper or Brave Search.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        Dictionary with search results
    """
    serper_key = os.getenv("SERPER_API_KEY") or ""
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY") or ""

    if serper_key:
        return await _serper_search(query, limit, serper_key)
    if brave_key:
        return await _brave_search(query, limit, brave_key)

    # Fallback: DuckDuckGo instant answer
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=15,
        )
        data = resp.json()
        return {
            "results": [
                {
                    "title": data.get("Heading", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", ""),
                }
            ]
        }


async def _serper_search(query: str, limit: int, api_key: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": limit},
            headers={"X-API-KEY": api_key},
            timeout=15,
        )
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:limit]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return {"results": results}


async def _brave_search(query: str, limit: int, api_key: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": limit},
            headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key},
            timeout=15,
        )
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:limit]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return {"results": results}


async def fetch_page(url: str) -> str:
    """Fetch and extract text content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        Extracted text content
    """
    import httpx
    from bs4 import BeautifulSoup

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, timeout=20)
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text[:15000]  # Cap at 15k chars
