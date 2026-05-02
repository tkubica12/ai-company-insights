from datetime import UTC, datetime

from ai_company_insights.config import Settings
from ai_company_insights.models import (
    Citation,
    CompanyIdentity,
    CrawledPage,
    SearchResult,
    StockQuote,
)
from ai_company_insights.researcher import CompanyResearcher


class FakeAres:
    async def resolve_company(self, query: str):
        raw = {
            "ico": "45274649",
            "obchodniJmeno": "ČEZ, a. s.",
            "sidlo": {"textovaAdresa": "Duhová 1444/2, Praha 4"},
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI"},
        }
        return (
            CompanyIdentity(
                query=query,
                ico="45274649",
                legal_name="ČEZ, a. s.",
                address="Duhová 1444/2, Praha 4",
                source_citation_ids=["ares-entity"],
            ),
            Citation(
                id="ares-entity",
                title="ARES",
                url="https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/45274649",
                source_type="government_registry",
            ),
            raw,
        )


class FakeSearch:
    async def search(self, query: str, *, count: int | None = None, provider: str | None = None):
        return [
            SearchResult.model_validate(
                {
                    "title": "ČEZ official",
                    "url": "https://www.cez.cz/",
                    "snippet": "Official ČEZ website",
                    "provider": provider or "fake",
                }
            ),
            SearchResult.model_validate(
                {
                    "title": "ČEZ výroční zpráva PDF",
                    "url": "https://www.cez.cz/webpublic/file/edee/ospol/fileexport/investors/annual-reports/cez-2024.pdf",
                    "snippet": "Výroční finanční zpráva Skupina ČEZ 2024.",
                    "provider": provider or "fake",
                }
            ),
        ]


class FakeCrawler:
    async def crawl(self, urls: list[str]):
        return [
            CrawledPage(
                url=urls[0],
                title="ČEZ",
                markdown=(
                    "ČEZ je energetická společnost. Jsou zde popsány investor relations a produkty."
                ),
                source="fake",
                links=[
                    "https://www.cez.cz/cs/pro-investory/aktualne/dulezite-oznameni",
                    "https://www.cez.cz/webpublic/file/edee/ospol/fileexport/investors/annual-reports/cez-2024.pdf",
                ],
            )
        ]


class FakeDocuments:
    async def convert_urls(self, urls: list[str]):
        return [
            CrawledPage(
                url=urls[0],
                title="Výroční finanční zpráva Skupina ČEZ 2024",
                markdown="# Výroční finanční zpráva\n\nStrategie a výsledky.",
                source="markitdown",
            )
        ]


class FakeNews:
    async def search(self, query: str, *, count: int | None = None):
        return [
            SearchResult.model_validate(
                {
                    "title": "ČEZ news",
                    "url": "https://example.com/cez-news",
                    "snippet": "ČEZ appeared in the news.",
                    "provider": "fake-news",
                }
            )
        ]


class FakeStock:
    async def get_quote(self, company: CompanyIdentity):
        return StockQuote(
            symbol="CEZ.PR",
            provider="fake-stock",
            source_url="https://example.com/quote/CEZ.PR",
            currency="CZK",
            exchange_name="Prague",
            regular_market_price=1200.0,
            previous_close=1190.0,
            market_time=datetime(2026, 4, 30, tzinfo=UTC),
        )


class FakeFoundryWeb:
    def registrations_needed(self) -> list[str]:
        return ["Volitelné: nakonfigurujte Foundry Web Search."]


async def test_company_research_report_has_structure_and_citations() -> None:
    researcher = CompanyResearcher(Settings(search_provider="none"))
    researcher._ares = FakeAres()  # type: ignore[method-assign]
    researcher._search = FakeSearch()  # type: ignore[method-assign]
    researcher._news = FakeNews()  # type: ignore[method-assign]
    researcher._stocks = FakeStock()  # type: ignore[method-assign]
    researcher._crawler = FakeCrawler()  # type: ignore[method-assign]
    researcher._documents = FakeDocuments()  # type: ignore[method-assign]
    researcher._foundry_web = FakeFoundryWeb()  # type: ignore[method-assign]

    from ai_company_insights.models import ResearchMode

    report = await researcher.research("ČEZ", ResearchMode(use_foundry_synthesis=False))

    assert report.company.ico == "45274649"
    assert report.citations[0].id == "ares-entity"
    assert any(citation.id.startswith("news-") for citation in report.citations)
    assert any(citation.id == "stock-quote" for citation in report.citations)
    assert any(citation.id.startswith("document-") for citation in report.citations)
    assert any(section.evidence for section in report.sections)
    assert report.raw["document_urls"]
    assert report.token_usage.input_tokens == 0
    assert report.registrations_needed == ["Volitelné: nakonfigurujte Foundry Web Search."]
