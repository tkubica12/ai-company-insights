from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from azure.identity.aio import AzureCliCredential, DefaultAzureCredential

from ai_company_insights.config import Settings
from ai_company_insights.models import CompanyResearchReport, Evidence, ReportSection, TokenUsage
from ai_company_insights.token_usage import extract_token_usage, merge_token_usage
from ai_company_insights.utils import truncate


class FoundrySynthesizer:
    """Optional Microsoft Agent Framework synthesis over collected evidence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def synthesize(
        self, report: CompanyResearchReport
    ) -> tuple[str, list[ReportSection], TokenUsage]:
        from agent_framework import Agent, SkillsProvider
        from agent_framework.foundry import FoundryChatClient

        credential = (
            AzureCliCredential(
                process_timeout=self._settings.foundry_azure_cli_process_timeout_seconds
            )
            if self._settings.foundry_use_entra
            else DefaultAzureCredential()
        )
        async with credential:
            client = FoundryChatClient(
                project_endpoint=self._settings.foundry_project_endpoint,
                model=self._settings.foundry_model,
                credential=credential,
            )
            skills_provider = SkillsProvider(skill_paths=Path(self._settings.skills_dir))
            agent = Agent(
                client=client,
                name="CzechCompanyResearchSynthesizer",
                instructions=(
                    "Jsi seniorní tým pro hloubkovou due-diligence rešerši české firmy. "
                    "Pracuješ výhradně z dodaných důkazů a citačních ID. Piš česky. "
                    "Každé podstatné tvrzení musí být opřené o citační ID ze vstupu. "
                    "Nevymýšlej si fakta a jasně rozlišuj doložené poznatky od hypotéz."
                ),
                context_providers=[skills_provider],
            )
            payload = json.dumps(self._analysis_payload(report), ensure_ascii=False)
            analyst = await agent.run(
                "ROLE 1 - hlavní analytik. Proveď hlubší triangulaci napříč registry, "
                "financemi, veřejnými dokumenty, médii, strategickými signály a tržními daty. "
                "Vrať pracovní memorandum v češtině se sekcemi: jistá fakta, sporné/nejisté "
                "body, finanční a strategické implikace, reputační rizika, obchodní příležitosti. "
                "Každý bod cituj ID ve formátu [web-1].\n\n"
                f"{payload}"
            )
            critic = await agent.run(
                "ROLE 2 - skeptický reviewer a quality-control. Zkontroluj následující "
                "memorandum proti původnímu payloadu. Najdi přestřelená tvrzení, chybějící "
                "citace, duplicitní nebo slabé důkazy a doporuč, co má finální report "
                "zpřesnit. Neuváděj nová fakta bez citačních ID.\n\n"
                f"PAYLOAD:\n{payload}\n\nMEMO:\n{analyst.text}"
            )
            final = await agent.run(
                "ROLE 3 - finální syntéza pro obchodní tým. Vrať pouze validní JSON bez "
                "Markdownu a bez komentáře mimo JSON. Schema: "
                '{"executive_summary":"2-5 odstavců s citacemi",'
                '"sections":[{"title":"Hlubší syntéza a triangulace",'
                '"summary":"stručný souhrn",'
                '"evidence":[{"claim":"krátké tvrzení",'
                '"value":"obchodní význam, nejistota nebo doporučení s citacemi",'
                '"citation_id":"existující citační ID", "confidence":0.0}]}]}. '
                "Vytvoř právě tyto tři sekce: Hlubší syntéza a triangulace; "
                "Kontrola kvality a mezery v důkazech; Obchodní hypotézy pro další jednání. "
                "Do každé sekce dej 3-6 důkazních řádků. Používej pouze citační ID z payloadu. "
                "Zapracuj kritiku a neopakuj celé zdrojové výňatky.\n\n"
                f"PAYLOAD:\n{payload}\n\nMEMO:\n{analyst.text}\n\nKRITIKA:\n{critic.text}"
            )
        usage = merge_token_usage(extract_token_usage(analyst), extract_token_usage(critic))
        usage = merge_token_usage(usage, extract_token_usage(final))
        summary, sections = self._parse_synthesis_json(final.text, report)
        return summary, sections, usage

    def _analysis_payload(self, report: CompanyResearchReport) -> dict[str, Any]:
        citation_ids = {citation.id for citation in report.citations}
        sections = []
        for section in report.sections:
            evidence = []
            for item in section.evidence[:12]:
                if item.citation_id not in citation_ids:
                    continue
                evidence.append(
                    {
                        "claim": item.claim,
                        "value": truncate(item.value or "", 1200),
                        "citation_id": item.citation_id,
                        "confidence": item.confidence,
                    }
                )
            sections.append(
                {"title": section.title, "summary": section.summary, "evidence": evidence}
            )
        return {
            "company": report.company.model_dump(mode="json"),
            "executive_summary": report.executive_summary,
            "sections": sections,
            "citations": [
                {
                    "id": citation.id,
                    "title": citation.title,
                    "url": citation.url,
                    "artifact_path": citation.artifact_path,
                    "source_type": citation.source_type,
                    "publisher": citation.publisher,
                    "snippet": truncate(citation.snippet or "", 800),
                }
                for citation in report.citations
            ],
            "raw": {
                "queries": report.raw.get("queries", []),
                "news_queries": report.raw.get("news_queries", []),
                "source_errors": report.raw.get("source_errors", []),
            },
        }

    def _parse_synthesis_json(
        self, text: str, report: CompanyResearchReport
    ) -> tuple[str, list[ReportSection]]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Foundry synthesis did not return a JSON object.")
        payload = json.loads(match.group(0))
        citation_ids = {citation.id for citation in report.citations}
        sections = []
        for raw_section in payload.get("sections", []):
            evidence = []
            for raw_item in raw_section.get("evidence", []):
                citation_id = str(raw_item.get("citation_id") or "")
                if citation_id not in citation_ids:
                    continue
                confidence = float(raw_item.get("confidence") or 0.6)
                evidence.append(
                    Evidence(
                        citation_id=citation_id,
                        claim=str(raw_item.get("claim") or "Syntetizované zjištění"),
                        value=str(raw_item.get("value") or ""),
                        confidence=max(0.0, min(confidence, 1.0)),
                    )
                )
            if evidence:
                sections.append(
                    ReportSection(
                        title=str(raw_section.get("title") or "Hlubší syntéza"),
                        summary=str(raw_section.get("summary") or ""),
                        evidence=evidence[:8],
                    )
                )
        summary = str(payload.get("executive_summary") or report.executive_summary)
        return summary, sections
