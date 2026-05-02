from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import SecretStr

from ai_company_insights.config import Settings
from ai_company_insights.models import SearchResult


def _secret_value(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


class SearchClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(
        self, query: str, *, count: int | None = None, provider: str | None = None
    ) -> list[SearchResult]:
        provider = provider or self._settings.search_provider
        if provider == "none":
            return []
        if provider == "brave":
            return await self.search_brave(query, count=count)
        if provider == "tavily":
            return await self.search_tavily(query, count=count)
        if provider == "auto" and self._settings.brave_api_key:
            try:
                return await self.search_brave(query, count=count)
            except httpx.HTTPStatusError:
                if self._settings.tavily_api_key:
                    return await self.search_tavily(query, count=count)
                raise
        if provider == "auto" and self._settings.tavily_api_key:
            return await self.search_tavily(query, count=count)
        return []

    async def search_brave(self, query: str, *, count: int | None = None) -> list[SearchResult]:
        api_key = _secret_value(self._settings.brave_api_key)
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = None
            for attempt in range(3):
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={
                        "q": query,
                        "count": count or self._settings.max_search_results,
                        "search_lang": "cs",
                    },
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                        "User-Agent": self._settings.user_agent,
                    },
                )
                if response.status_code != 429:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
            if response is None:
                return []
            response.raise_for_status()
        results = (response.json().get("web") or {}).get("results") or []
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item["url"],
                snippet=item.get("description"),
                provider="brave",
            )
            for item in results
            if item.get("url")
        ]

    async def search_tavily(self, query: str, *, count: int | None = None) -> list[SearchResult]:
        api_key = _secret_value(self._settings.tavily_api_key)
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": count or self._settings.max_search_results,
                    "search_depth": "advanced",
                    "include_answer": False,
                    "include_raw_content": False,
                },
                headers={"User-Agent": self._settings.user_agent},
            )
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled",
                url=item["url"],
                snippet=item.get("content"),
                provider="tavily",
            )
            for item in payload.get("results", [])
            if item.get("url")
        ]
