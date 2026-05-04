from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    id: str
    title: str
    url: str | None = None
    artifact_path: str | None = None
    source_type: str
    publisher: str | None = None
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    snippet: str | None = None


class Evidence(BaseModel):
    citation_id: str
    claim: str
    value: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.7)


class CompanyIdentity(BaseModel):
    query: str
    ico: str | None = None
    legal_name: str | None = None
    address: str | None = None
    legal_form: str | None = None
    tax_id: str | None = None
    established_on: str | None = None
    nace_codes: list[str] = Field(default_factory=list)
    source_citation_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str | None = None
    provider: str


class CrawledPage(BaseModel):
    url: str
    title: str | None = None
    markdown: str
    source: str
    links: list[str] = Field(default_factory=list)
    artifact_path: str | None = None


class StockQuote(BaseModel):
    symbol: str
    provider: str
    source_url: str
    currency: str | None = None
    exchange_name: str | None = None
    regular_market_price: float | None = None
    previous_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    volume: int | None = None
    shares_outstanding: int | None = None
    estimated_market_cap: float | None = None
    market_time: datetime | None = None


class ReportSection(BaseModel):
    title: str
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


class CompanyResearchReport(BaseModel):
    company: CompanyIdentity
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executive_summary: str
    sections: list[ReportSection]
    citations: list[Citation]
    registrations_needed: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


class ResearchMode(BaseModel):
    use_foundry_synthesis: bool = True
    use_foundry_web_search: bool = False
    search_provider: Literal["auto", "brave", "tavily", "foundry-web", "none"] = "auto"
