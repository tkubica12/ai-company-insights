from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    BingGroundingSearchConfiguration,
    BingGroundingSearchToolParameters,
    BingGroundingTool,
    PromptAgentDefinition,
    WebSearchApproximateLocation,
    WebSearchConfiguration,
    WebSearchTool,
)
from azure.identity import DefaultAzureCredential

from ai_company_insights.config import Settings
from ai_company_insights.models import TokenUsage
from ai_company_insights.token_usage import extract_token_usage


class FoundryWebSearch:
    """Optional Foundry Web Search / Grounding with Bing helper."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def registrations_needed(self) -> list[str]:
        if not self._settings.bing_web_search_enabled:
            return ["Volitelné: zapněte hostované Foundry Web Search nebo Grounding with Bing."]
        if self._settings.foundry_web_search_mode == "hosted":
            return []
        if self._settings.foundry_web_search_mode == "custom":
            if bool(self._settings.bing_custom_search_project_connection_id) != bool(
                self._settings.bing_custom_search_instance_name
            ):
                return [
                    "Nastavte současně BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID a "
                    "BING_CUSTOM_SEARCH_INSTANCE_NAME pro doménově omezené Bing Custom Search."
                ]
            return []
        if not self._settings.bing_project_connection_name and not (
            self._settings.bing_custom_search_project_connection_id
            and self._settings.bing_custom_search_instance_name
        ):
            return [
                "Nastavte BING_PROJECT_CONNECTION_NAME pro Grounding with Bing Search, nebo "
                "současně BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID a "
                "BING_CUSTOM_SEARCH_INSTANCE_NAME pro doménově omezené Bing Custom Search."
            ]
        return []

    def ask(self, prompt: str) -> tuple[str, list[str], TokenUsage]:
        project = AIProjectClient(
            endpoint=self._settings.foundry_project_endpoint,
            credential=DefaultAzureCredential(),
        )
        openai = project.get_openai_client()

        if self._settings.foundry_web_search_mode == "hosted":
            tool = WebSearchTool(
                user_location=WebSearchApproximateLocation(
                    country="CZ", city="Prague", region="Prague"
                )
            )
        elif self._settings.foundry_web_search_mode == "custom":
            connection_id = self._settings.bing_custom_search_project_connection_id
            if not connection_id:
                raise ValueError("BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID is required.")
            tool = WebSearchTool(
                custom_search_configuration=WebSearchConfiguration(
                    project_connection_id=connection_id,
                    instance_name=self._settings.bing_custom_search_instance_name or "",
                )
            )
        else:
            connection = project.connections.get(self._settings.bing_project_connection_name or "")
            tool = BingGroundingTool(
                bing_grounding=BingGroundingSearchToolParameters(
                    search_configurations=[
                        BingGroundingSearchConfiguration(project_connection_id=connection.id)
                    ]
                )
            )

        agent = project.agents.create_version(
            agent_name="company-research-web-search",
            definition=PromptAgentDefinition(
                model=self._settings.foundry_model,
                instructions=(
                    "Vyhledáváš veřejné webové informace o českých firmách a uvádíš URL citace. "
                    "Vracej pouze stručná rešeršní zjištění v češtině. Nenabízej další pomoc."
                ),
                tools=[tool],
            ),
            description="Temporary company research web search agent.",
        )
        try:
            response = openai.responses.create(
                tool_choice="required",
                input=prompt,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )
            citations: list[str] = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "message":
                    continue
                for content in getattr(item, "content", []) or []:
                    for annotation in getattr(content, "annotations", []) or []:
                        if getattr(annotation, "type", None) == "url_citation":
                            citations.append(str(annotation.url))
            return response.output_text, sorted(set(citations)), extract_token_usage(response)
        finally:
            project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
