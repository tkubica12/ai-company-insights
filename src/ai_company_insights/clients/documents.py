from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ai_company_insights.config import Settings
from ai_company_insights.models import CrawledPage
from ai_company_insights.utils import truncate


class DocumentProcessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def convert_urls(self, urls: list[str]) -> list[CrawledPage]:
        document_urls = [url for url in urls if self._looks_like_supported_document(url)]
        converted: list[CrawledPage] = []
        for url in document_urls[: self._settings.max_documents]:
            with contextlib.suppress(Exception):
                converted.append(await self._convert_url(url))
        return converted

    def _looks_like_supported_document(self, url: str) -> bool:
        lowered = url.lower().split("?", 1)[0]
        return lowered.endswith((".pdf", ".docx", ".pptx", ".xlsx"))

    async def _convert_url(self, url: str) -> CrawledPage:
        from markitdown import MarkItDown

        suffix = Path(url.lower().split("?", 1)[0]).suffix or ".bin"
        async with httpx.AsyncClient(
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self._settings.user_agent},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_path = Path(temp_file.name)
        try:
            if suffix.casefold() == ".pdf":
                return self._convert_pdf(url=str(response.url), temp_path=temp_path)
            result = MarkItDown().convert(str(temp_path))
            return CrawledPage(
                url=str(response.url),
                title=self._title_from_url(str(response.url)),
                markdown=truncate(result.text_content or "", self._settings.max_page_chars),
                source="markitdown",
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _convert_pdf(self, *, url: str, temp_path: Path) -> CrawledPage:
        import pdfplumber

        chunks: list[str] = []
        with pdfplumber.open(temp_path) as pdf:
            for page_number, page in enumerate(pdf.pages[: self._settings.max_pdf_pages], start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(f"## Strana {page_number}\n\n{text.strip()}")
        return CrawledPage(
            url=url,
            title=self._title_from_url(url),
            markdown=truncate("\n\n".join(chunks), self._settings.max_page_chars),
            source="pdfplumber",
        )

    def _title_from_url(self, url: str) -> str:
        path = unquote(urlparse(url).path)
        return Path(path).name or url
