"""Tavily Search adapter returning ranked snippets for retrieval and RAG.

Advanced search requests publication dates and usage, with no generated answer
or raw-page extraction. Reported credits use pay-as-you-go pricing; responses
without usage use the configured depth's published credit cost.

API reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# https://docs.tavily.com/documentation/api-credits
TAVILY_COST_PER_CREDIT = 0.008
TAVILY_SEARCH_CREDITS = {"basic": 1, "advanced": 2, "fast": 1, "ultra-fast": 1}
_MAX_QUERY_CHARS = 400
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
logger = logging.getLogger(__name__)


class TavilySearcher(Searcher):
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        search_depth: str = "advanced",
        max_attempts: int = 4,
        timeout: float = 60.0,
    ):
        if search_depth not in TAVILY_SEARCH_CREDITS:
            raise ValueError(f"unsupported Tavily search_depth: {search_depth!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY required")
        self.search_depth = search_depth
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout)

    def search_payload(self, query: str, num_results: int) -> dict[str, Any]:
        """Apply the provider's query and result limits and request snippet metadata."""
        if num_results < 0:
            raise ValueError("num_results must be non-negative")
        if len(query) > _MAX_QUERY_CHARS:
            logger.warning("Tavily query exceeds %s characters; truncating", _MAX_QUERY_CHARS)
            query = query[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        return {
            "query": query,
            "max_results": min(num_results, 20),
            "search_depth": self.search_depth,
            "include_published_date": True,
            "include_usage": True,
        }

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        return (await self.run(query, num_results)).results

    async def run(
        self,
        query: str,
        num_results: int = 10,
        *,
        objective: str | None = None,
        search_queries: list[str] | None = None,
    ) -> SearchResponse:
        """Search once and preserve ranked evidence, provider metadata and request cost."""
        del objective, search_queries
        start = time.perf_counter()
        payload = self.search_payload(query, num_results)
        data = await self._post(payload)
        results = [
            SearchResult(
                url=row.get("url") or "",
                title=row.get("title") or "",
                text=row.get("raw_content") or row.get("content") or "",
                metadata={
                    "rank": rank,
                    "score": row.get("score"),
                    "published_date": row.get("published_date"),
                    **{key: row[key] for key in ("id", "favicon", "images") if key in row},
                },
            )
            for rank, row in enumerate(data.get("results") or [])
        ]
        credits = (data.get("usage") or {}).get("credits")
        if credits is None:
            credits = TAVILY_SEARCH_CREDITS[self.search_depth]
        return SearchResponse(
            results=results,
            cost_usd=float(credits) * TAVILY_COST_PER_CREDIT,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=data.get("request_id"),
            queries=[payload["query"]],
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Retry transient provider failures, propagating permanent or exhausted errors."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(TAVILY_SEARCH_URL, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.max_attempts - 1:
                    raise
            else:
                if response.is_success:
                    return response.json()
                if (
                    response.status_code not in _RETRYABLE_STATUS
                    or attempt == self.max_attempts - 1
                ):
                    response.raise_for_status()
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()
