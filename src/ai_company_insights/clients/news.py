from __future__ import annotations

from typing import Any

import httpx
from pydantic import SecretStr

from ai_company_insights.config import Settings
from ai_company_insights.models import SearchResult


def _secret_value(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


class NewsClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, query: str, *, count: int | None = None) -> list[SearchResult]:
        results: list[SearchResult] = []
        results.extend(await self.search_newsapi(query, count=count))
        results.extend(await self.search_mediastack(query, count=count))
        return self._dedupe(results)

    async def search_newsapi(self, query: str, *, count: int | None = None) -> list[SearchResult]:
        api_key = _secret_value(self._settings.newsapi_api_key)
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": count or self._settings.max_news_results,
                    "sortBy": "relevancy",
                    "apiKey": api_key,
                },
                headers={"User-Agent": self._settings.user_agent},
            )
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled news item",
                url=item["url"],
                snippet=item.get("description"),
                provider=f"newsapi:{(item.get('source') or {}).get('name') or 'unknown'}",
            )
            for item in payload.get("articles", [])
            if item.get("url")
        ]

    async def search_mediastack(
        self, query: str, *, count: int | None = None
    ) -> list[SearchResult]:
        api_key = _secret_value(self._settings.mediastack_api_key)
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.get(
                "http://api.mediastack.com/v1/news",
                params={
                    "access_key": api_key,
                    "keywords": query,
                    "limit": count or self._settings.max_news_results,
                    "sort": "published_desc",
                },
                headers={"User-Agent": self._settings.user_agent},
            )
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [
            SearchResult(
                title=item.get("title") or item.get("url") or "Untitled news item",
                url=item["url"],
                snippet=item.get("description"),
                provider=f"mediastack:{item.get('source') or 'unknown'}",
            )
            for item in payload.get("data", [])
            if item.get("url")
        ]

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            url = str(result.url)
            title = " ".join(result.title.casefold().split())
            if url in seen or title in seen_titles:
                continue
            seen.add(url)
            seen_titles.add(title)
            unique.append(result)
            if len(unique) >= self._settings.max_news_results:
                break
        return unique
