from __future__ import annotations

import re
from collections.abc import Sequence

from ai_company_insights.config import Settings
from ai_company_insights.models import Citation, CompanyResearchReport, Evidence, ReportSection


class MarkdownReportRenderer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def render(self, report: CompanyResearchReport) -> str:
        template = self._settings.report_template_path.read_text(encoding="utf-8")
        template = re.sub(r"<!--.*?-->\s*", "", template, flags=re.DOTALL)
        rendered = re.sub(
            r"\{\{\s*section:(?P<title>[^}]+?)\s*\}\}",
            lambda match: self._render_section(report, match.group("title").strip()),
            template,
        )
        replacements = {
            "company_name": report.company.legal_name or report.company.query,
            "query": report.company.query,
            "ico": report.company.ico or "Not found",
            "generated_at": report.generated_at.isoformat(),
            "executive_summary": report.executive_summary,
            "company_at_glance": self._company_at_glance(report),
            "resources_table": self._resources_table(report.citations),
            "registrations_needed": self._registrations_needed(report),
            "token_usage": self._token_usage(report),
        }
        for key, value in replacements.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
            rendered = rendered.replace(f"{{{{ {key} }}}}", value)
        return rendered.strip() + "\n"

    def _company_at_glance(self, report: CompanyResearchReport) -> str:
        company = report.company
        rows = [
            ("Obchodní firma", company.legal_name),
            ("Dotaz", company.query),
            ("IČO", company.ico),
            ("DIČ", company.tax_id),
            ("Sídlo", company.address),
            ("Právní forma", company.legal_form),
            ("Datum vzniku", company.established_on),
            ("Kódy NACE", ", ".join(company.nace_codes[:12]) if company.nace_codes else None),
        ]
        return self._markdown_table(
            ("Pole", "Hodnota"), [(label, value or "Nenalezeno") for label, value in rows]
        )

    def _render_section(self, report: CompanyResearchReport, title: str) -> str:
        section = next(
            (
                candidate
                for candidate in report.sections
                if candidate.title.lower() == title.lower()
            ),
            None,
        )
        if not section:
            return ""
        if section.title.casefold() in {"extrahovaný obsah zdrojů", "veřejné dokumenty"}:
            return self._render_evidence_blocks(section, report.citations)
        lines = [section.summary]
        if section.evidence:
            finding_header, evidence_header = self._headers_for_section(section.title)
            citations_by_id = {citation.id: citation for citation in report.citations}
            lines.extend(
                [
                    "",
                    f"| {finding_header} | {evidence_header} | Citace | Spolehlivost |",
                    "| --- | --- | --- | ---: |",
                ]
            )
            for evidence in section.evidence:
                lines.append(self._evidence_row(evidence, citations_by_id))
        return "\n".join(lines)

    def _render_evidence_blocks(self, section: ReportSection, citations: list[Citation]) -> str:
        citations_by_id = {citation.id: citation for citation in citations}
        lines = [section.summary]
        for evidence in section.evidence:
            lines.extend(
                [
                    "",
                    f"### {evidence.claim}",
                    "",
                    "Citace: "
                    f"{self._citation_link(evidence.citation_id, citations_by_id)} · "
                    f"Spolehlivost: {evidence.confidence:.2f}",
                    "",
                    self._blockquote(evidence.value or ""),
                ]
            )
        return "\n".join(lines)

    def _headers_for_section(self, title: str) -> tuple[str, str]:
        section_title = title.casefold()
        if section_title == "extrahovaný obsah zdrojů":
            return ("Zdroj", "Užitečný výňatek")
        if section_title in {
            "webové zdroje",
            "zprávy a média",
            "strategie a priority",
            "produkty, služby a inovace",
            "významná oznámení a obchody",
            "oborové trendy",
            "mediální sentiment",
            "webové ověření foundry",
        }:
            return ("Zjištění", "Proč je to důležité")
        if section_title == "byznysový kontext":
            return ("Oblast", "Zjištění")
        if section_title == "vlastnická a skupinová struktura":
            return ("Oblast", "Zjištění")
        if section_title == "finanční informace":
            return ("Oblast", "Zjištění")
        if section_title == "reputační rizika":
            return ("Rizikový okruh", "Signál")
        if section_title == "potenciální příležitosti":
            return ("Příležitost", "Proč je relevantní")
        if section_title == "otázky pro první obchodní schůzku":
            return ("Otázka", "Důvod")
        if section_title == "hlubší syntéza a triangulace":
            return ("Syntetizované zjištění", "Interpretace")
        if section_title == "kontrola kvality a mezery v důkazech":
            return ("Kontrolní bod", "Dopad na spolehlivost")
        if section_title == "obchodní hypotézy pro další jednání":
            return ("Hypotéza", "Jak ji ověřit")
        if section_title == "akciové informace":
            return ("Metrika", "Hodnota")
        if section_title == "identita v registrech":
            return ("Pole registru", "Hodnota")
        if section_title == "veřejné dokumenty":
            return ("Dokument", "Výňatek")
        return ("Zjištění", "Důkaz")

    def _evidence_row(self, evidence: Evidence, citations_by_id: dict[str, Citation]) -> str:
        return (
            f"| {self._cell(evidence.claim)} | {self._cell(evidence.value or '')} | "
            f"{self._citation_link(evidence.citation_id, citations_by_id)} | "
            f"{evidence.confidence:.2f} |"
        )

    def _resources_table(self, citations: list[Citation]) -> str:
        rows = [
            (
                citation.id,
                citation.source_type,
                citation.publisher or "",
                self._resource_link(citation),
            )
            for citation in citations
        ]
        return self._markdown_table(("ID", "Typ", "Vydavatel/domain", "Zdroj"), rows)

    def _registrations_needed(self, report: CompanyResearchReport) -> str:
        if not report.registrations_needed:
            return "_Nebyly hlášeny žádné chybějící registrace ani konfigurace._"
        return "\n".join(f"- {item}" for item in report.registrations_needed)

    def _token_usage(self, report: CompanyResearchReport) -> str:
        return self._markdown_table(
            ("Metrika", "Počet tokenů"),
            [
                ("Vstupní tokeny", str(report.token_usage.input_tokens)),
                ("Vstupní tokeny z cache", str(report.token_usage.cached_input_tokens)),
                ("Výstupní tokeny", str(report.token_usage.output_tokens)),
            ],
        )

    def _markdown_table(
        self, headers: tuple[str, ...], rows: Sequence[Sequence[str | None]]
    ) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(self._cell(value or "") for value in row) + " |")
        return "\n".join(lines)

    def _cell(self, value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    def _blockquote(self, value: str) -> str:
        lines = str(value).strip().splitlines()
        return "\n".join(f"> {line}" if line else ">" for line in lines)

    def _citation_link(self, citation_id: str, citations_by_id: dict[str, Citation]) -> str:
        citation = citations_by_id.get(citation_id)
        if not citation:
            return f"[{self._cell(citation_id)}]"
        target = citation.artifact_path or citation.url
        if not target:
            return f"[{self._cell(citation_id)}]"
        return f"[{self._cell(citation_id)}]({self._markdown_url(target)})"

    def _resource_link(self, citation: Citation) -> str:
        if citation.artifact_path and citation.url:
            return (
                f"[{self._cell(citation.title)}]({self._markdown_url(citation.artifact_path)})"
                f" · [web]({self._markdown_url(citation.url)})"
            )
        target = citation.artifact_path or citation.url
        if target:
            return f"[{self._cell(citation.title)}]({self._markdown_url(target)})"
        return citation.title

    def _markdown_url(self, value: str) -> str:
        return str(value).replace(" ", "%20").replace(")", "%29")
