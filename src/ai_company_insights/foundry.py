from __future__ import annotations

import json
from pathlib import Path

from azure.identity.aio import AzureCliCredential, DefaultAzureCredential

from ai_company_insights.config import Settings
from ai_company_insights.models import CompanyResearchReport, TokenUsage
from ai_company_insights.token_usage import extract_token_usage


class FoundrySynthesizer:
    """Optional Microsoft Agent Framework synthesis over collected evidence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def synthesize(self, report: CompanyResearchReport) -> tuple[str, TokenUsage]:
        from agent_framework import Agent, SkillsProvider
        from agent_framework.foundry import FoundryChatClient

        credential = (
            AzureCliCredential() if self._settings.foundry_use_entra else DefaultAzureCredential()
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
                    "Syntetizuješ rešerši české firmy výhradně z dodaných důkazů. "
                    "Piš česky. Každé podstatné tvrzení musí obsahovat citační ID ze vstupu. "
                    "Nevymýšlej si fakta. Vrať pouze exekutivní shrnutí: 2-4 stručné "
                    "odstavce, bez nadpisu, bez odrážek, bez tabulek a bez nabídek další pomoci."
                ),
                context_providers=[skills_provider],
            )
            payload = report.model_dump(mode="json")
            result = await agent.run(
                "Napiš pouze exekutivní shrnutí této rešerše firmy podložené důkazy. "
                "Neopakuj detailní sekce. Používej citační ID přímo v textu, například "
                "[ares-entity]. Odpověz česky.\n\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
        return result.text, extract_token_usage(result)
