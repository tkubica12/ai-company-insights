from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    foundry_project_endpoint: str = Field(
        default="https://tomaskubica-foundry-resource.services.ai.azure.com/api/projects/tomaskubica-foundry-project"
    )
    foundry_model: str = "gpt-5.4"
    foundry_use_entra: bool = True
    foundry_azure_cli_process_timeout_seconds: int = 90

    brave_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    newsapi_api_key: SecretStr | None = None
    mediastack_api_key: SecretStr | None = None

    bing_web_search_enabled: bool = False
    foundry_web_search_mode: Literal["hosted", "bing-grounding", "custom"] = "hosted"
    bing_project_connection_name: str | None = "tomaskubica-bing-grounding"
    bing_custom_search_project_connection_id: str | None = None
    bing_custom_search_instance_name: str | None = None

    search_provider: Literal["auto", "brave", "tavily", "foundry-web", "none"] = "auto"
    user_agent: str = (
        "ai-company-insights/0.1 (public-source company research; contact: repository owner)"
    )
    request_timeout_seconds: float = 30.0
    max_parallel_source_queries: int = 5
    max_search_results: int = 12
    max_web_results: int = 150
    max_news_results: int = 150
    max_crawl_pages: int = 60
    max_followup_pages: int = 50
    max_documents: int = 20
    max_pdf_pages: int = 30
    max_page_chars: int = 40_000
    max_document_conversion_bytes: int = 8_000_000
    enable_stock_lookup: bool = True
    stock_symbol_overrides: str = "45274649=CEZ.PR"
    stock_shares_outstanding_overrides: str = "45274649=537989759"
    report_template_path: Path = Path("templates/company_report.md")
    enable_browser_crawler: bool = False
    output_dir: Path = Path("outputs")
    output_artifact_dir: Path | None = None
    output_artifact_link_prefix: str | None = None
    skills_dir: Path = Path("skills")


@lru_cache
def get_settings() -> Settings:
    return Settings()
