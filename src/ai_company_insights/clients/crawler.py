from __future__ import annotations

import contextlib
from typing import Any
from urllib.parse import urldefrag, urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup

from ai_company_insights.config import Settings
from ai_company_insights.models import CrawledPage
from ai_company_insights.utils import truncate


class PageCrawler:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def crawl(self, urls: list[str]) -> list[CrawledPage]:
        pages: list[CrawledPage] = []
        for url in urls[: self._settings.max_crawl_pages]:
            with contextlib.suppress(Exception):
                page = await self._crawl_one(url)
                if page.markdown.strip() and not self._is_low_value_extraction(page.markdown):
                    pages.append(page)
        return pages

    async def _crawl_one(self, url: str) -> CrawledPage:
        with contextlib.suppress(Exception):
            http_page = await self._crawl_with_http(url)
            if http_page.markdown.strip():
                return http_page
        crawl4ai_page = await self._crawl_with_crawl4ai(url)
        if crawl4ai_page:
            return crawl4ai_page
        return CrawledPage(url=url, title=None, markdown="", source="not-extracted")

    async def _crawl_with_crawl4ai(self, url: str) -> CrawledPage | None:
        if not self._settings.enable_browser_crawler:
            return None
        try:
            from crawl4ai import AsyncWebCrawler  # type: ignore[import-not-found]
        except Exception:
            return None

        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result: Any = await crawler.arun(url=url)
            markdown = getattr(result, "markdown", None) or ""
            if not markdown.strip():
                return None
            return CrawledPage(
                url=url,
                title=getattr(result, "title", None),
                markdown=truncate(markdown, self._settings.max_page_chars),
                source="crawl4ai",
            )
        except Exception:
            return None

    async def _crawl_with_http(self, url: str) -> CrawledPage:
        async with httpx.AsyncClient(
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self._settings.user_agent},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        html = response.text
        text = trafilatura.extract(html, url=url, output_format="markdown") or ""
        title = None
        links: list[str] = []
        with contextlib.suppress(Exception):
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.title
            title = title_tag.string if title_tag else None
            links = self._links_from_soup(soup, str(response.url))
        return CrawledPage(
            url=str(response.url),
            title=title.strip() if title else None,
            markdown=truncate(text, self._settings.max_page_chars),
            source="http+trafilatura",
            links=links,
        )

    def _links_from_soup(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            absolute, _fragment = urldefrag(urljoin(base_url, str(anchor["href"])))
            if absolute.startswith(("http://", "https://")) and absolute not in seen:
                seen.add(absolute)
                links.append(absolute)
        return links

    def _is_low_value_extraction(self, markdown: str) -> bool:
        normalized = " ".join(markdown.casefold().split())
        if not normalized:
            return True
        cookie_markers = [
            "# website information",
            "# informace o webu",
            "information about cookies",
            "informace o cookies",
            "zpracování osobních údajů",
            "vážení uživatelé webových stránek",
            "nařízení gdpr",
            "zákona o elektronických komunikacích",
            "processing of personal data",
            "rules of use of this website",
            "idnes a reklama",
            "souhlas s reklamou",
            "využitím cookies",
            "sitových identifikátorů",
            "síťových identifikátorů",
            "antiyoutuber.cz",
        ]
        marker_hits = sum(1 for marker in cookie_markers if marker in normalized[:1500])
        return marker_hits >= 3
