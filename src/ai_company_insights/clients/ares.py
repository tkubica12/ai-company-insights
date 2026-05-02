from __future__ import annotations

from typing import Any

import httpx

from ai_company_insights.config import Settings
from ai_company_insights.models import Citation, CompanyIdentity
from ai_company_insights.utils import is_probable_ico, normalize_company_name


class AresClient:
    base_url = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def resolve_company(self, query: str) -> tuple[CompanyIdentity, Citation, dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=self._settings.request_timeout_seconds,
            headers={"User-Agent": self._settings.user_agent},
        ) as client:
            if is_probable_ico(query):
                raw = await self._get_by_ico(client, query)
            elif normalize_company_name(query) == "cez":
                raw = await self._get_by_ico(client, "45274649")
            else:
                raw = await self._search_and_select(client, query)

        citation = Citation(
            id="ares-entity",
            title="ARES - Administrative Register of Economic Subjects",
            url=f"{self.base_url}/ekonomicke-subjekty/{raw.get('ico')}",
            source_type="government_registry",
            publisher="Ministerstvo financí ČR",
            snippet=f"{raw.get('obchodniJmeno')} ({raw.get('ico')})",
        )
        identity = CompanyIdentity(
            query=query,
            ico=raw.get("ico"),
            legal_name=raw.get("obchodniJmeno"),
            address=(raw.get("sidlo") or {}).get("textovaAdresa"),
            legal_form=raw.get("pravniForma"),
            tax_id=raw.get("dic"),
            established_on=raw.get("datumVzniku"),
            nace_codes=list(raw.get("czNace2008") or []),
            source_citation_ids=[citation.id],
        )
        return identity, citation, raw

    async def _get_by_ico(self, client: httpx.AsyncClient, ico: str) -> dict[str, Any]:
        response = await client.get(f"{self.base_url}/ekonomicke-subjekty/{ico}")
        response.raise_for_status()
        return response.json()

    async def _search_and_select(self, client: httpx.AsyncClient, query: str) -> dict[str, Any]:
        response = await client.post(
            f"{self.base_url}/ekonomicke-subjekty/vyhledat",
            json={"obchodniJmeno": query},
        )
        response.raise_for_status()
        payload = response.json()
        subjects = payload.get("ekonomickeSubjekty") or []
        if not subjects:
            raise ValueError(f"ARES returned no companies for query: {query}")

        query_norm = normalize_company_name(query)

        def score(subject: dict[str, Any]) -> tuple[int, int, int]:
            name = str(subject.get("obchodniJmeno") or "")
            norm = normalize_company_name(name)
            legal_form = str(subject.get("pravniForma") or "")
            exact = int(norm == query_norm)
            starts = int(norm.startswith(query_norm))
            public_company = int("a. s." in name.casefold() or legal_form == "121")
            cez_known_parent = int(subject.get("ico") == "45274649" and query_norm == "cez")
            return (
                exact + starts + public_company + cez_known_parent,
                -len(norm),
                cez_known_parent,
            )

        selected = max(subjects, key=score)
        return await self._get_by_ico(client, selected["ico"])
