from datetime import UTC, datetime

from ai_company_insights.config import Settings
from ai_company_insights.models import (
    Citation,
    CompanyIdentity,
    CompanyResearchReport,
    Evidence,
    ReportSection,
)
from ai_company_insights.report_renderer import MarkdownReportRenderer


def test_markdown_report_renderer_uses_template_sections(tmp_path) -> None:
    template = tmp_path / "report.md"
    template.write_text(
        "# {{company_name}}\n\n"
        "{{company_at_glance}}\n\n"
        "{{section:Identita v registrech}}\n\n"
        "{{token_usage}}\n\n"
        "{{resources_table}}\n",
        encoding="utf-8",
    )
    report = CompanyResearchReport(
        company=CompanyIdentity(
            query="ČEZ",
            ico="45274649",
            legal_name="ČEZ, a. s.",
            source_citation_ids=["ares-entity"],
        ),
        generated_at=datetime(2026, 4, 30, tzinfo=UTC),
        executive_summary="Shrnutí",
        sections=[
            ReportSection(
                title="Identita v registrech",
                summary="ARES dohledal firmu.",
                evidence=[
                    Evidence(
                        citation_id="ares-entity",
                        claim="Obchodní firma",
                        value="ČEZ, a. s.",
                    )
                ],
            )
        ],
        citations=[
            Citation(
                id="ares-entity",
                title="ARES",
                url="https://ares.gov.cz/",
                source_type="government_registry",
            )
        ],
    )

    markdown = MarkdownReportRenderer(Settings(report_template_path=template)).render(report)

    assert "# ČEZ, a. s." in markdown
    assert "ARES dohledal firmu." in markdown
    assert "| Obchodní firma | ČEZ, a. s. | [ares-entity] |" in markdown
    assert "| Vstupní tokeny | 0 |" in markdown
    assert "[ARES](https://ares.gov.cz/)" in markdown


def test_extracted_content_renders_as_markdown_block(tmp_path) -> None:
    template = tmp_path / "report.md"
    template.write_text("{{section:Extrahovaný obsah zdrojů}}\n", encoding="utf-8")
    report = CompanyResearchReport(
        company=CompanyIdentity(query="ČEZ", ico="45274649", legal_name="ČEZ, a. s."),
        generated_at=datetime(2026, 4, 30, tzinfo=UTC),
        executive_summary="Shrnutí",
        sections=[
            ReportSection(
                title="Extrahovaný obsah zdrojů",
                summary="Výňatky.",
                evidence=[
                    Evidence(
                        citation_id="page-1",
                        claim="Výroční finanční zprávy",
                        value="# Nadpis\n\n| A | B |\n|---|---|\n| 1 | 2 |",
                    )
                ],
            )
        ],
        citations=[],
    )

    markdown = MarkdownReportRenderer(Settings(report_template_path=template)).render(report)

    assert "| Zdroj | Užitečný výňatek |" not in markdown
    assert "### Výroční finanční zprávy" in markdown
    assert "Citace: [page-1]" in markdown
    assert "> # Nadpis" in markdown


def test_public_documents_render_as_markdown_blocks(tmp_path) -> None:
    template = tmp_path / "report.md"
    template.write_text("{{section:Veřejné dokumenty}}\n", encoding="utf-8")
    report = CompanyResearchReport(
        company=CompanyIdentity(query="ČEZ", ico="45274649", legal_name="ČEZ, a. s."),
        generated_at=datetime(2026, 4, 30, tzinfo=UTC),
        executive_summary="Shrnutí",
        sections=[
            ReportSection(
                title="Veřejné dokumenty",
                summary="Dokumenty.",
                evidence=[
                    Evidence(
                        citation_id="document-1",
                        claim="vyrocni-zprava.pdf",
                        value="## Strana 1\n\nText výroční zprávy.",
                    )
                ],
            )
        ],
        citations=[],
    )

    markdown = MarkdownReportRenderer(Settings(report_template_path=template)).render(report)

    assert "| Dokument | Výňatek |" not in markdown
    assert "### vyrocni-zprava.pdf" in markdown
    assert "Citace: [document-1]" in markdown
    assert "> ## Strana 1" in markdown
