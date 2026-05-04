from __future__ import annotations

import contextlib
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ai_company_insights.config import Settings
from ai_company_insights.models import CrawledPage
from ai_company_insights.utils import safe_filename_stem, truncate


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
        artifact_path = self._store_document(str(response.url), response.content, suffix)
        if len(response.content) > self._settings.max_document_conversion_bytes:
            return CrawledPage(
                url=str(response.url),
                title=self._title_from_url(str(response.url)),
                markdown=(
                    "Dokument byl stažen a uložen jako lokální artefakt, ale nebyl celý "
                    "převeden do Markdownu, protože překročil limit velikosti pro bezpečnou "
                    "konverzi."
                ),
                source="downloaded-document",
                artifact_path=artifact_path,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_path = Path(temp_file.name)
        try:
            if suffix.casefold() == ".pdf":
                page = self._convert_pdf(url=str(response.url), temp_path=temp_path)
                page.artifact_path = artifact_path
                return page
            result = MarkItDown().convert(str(temp_path))
            return CrawledPage(
                url=str(response.url),
                title=self._title_from_url(str(response.url)),
                markdown=truncate(result.text_content or "", self._settings.max_page_chars),
                source="markitdown",
                artifact_path=artifact_path,
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

    def _store_document(self, url: str, content: bytes, suffix: str) -> str | None:
        artifact_dir = self._settings.output_artifact_dir
        link_prefix = self._settings.output_artifact_link_prefix
        if not artifact_dir or not link_prefix:
            return None
        path = unquote(urlparse(url).path)
        stem = safe_filename_stem(Path(path).stem or "document", default="document")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        filename = f"{stem}-{digest}{suffix.lower()}"
        target = artifact_dir / "documents" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return f"{link_prefix}/documents/{filename}"
