import respx
from httpx import Response

from ai_company_insights.clients.ares import AresClient
from ai_company_insights.config import Settings


@respx.mock
async def test_ares_resolves_cez_to_parent_company() -> None:
    settings = Settings(search_provider="none")
    client = AresClient(settings)

    respx.post(
        "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/vyhledat"
    ).mock(
        return_value=Response(
            200,
            json={
                "ekonomickeSubjekty": [
                    {
                        "ico": "03479919",
                        "obchodniJmeno": "ČEZ Recyklace, s.r.o.",
                        "pravniForma": "112",
                    },
                    {"ico": "45274649", "obchodniJmeno": "ČEZ, a. s.", "pravniForma": "121"},
                ]
            },
        )
    )
    respx.get(
        "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/45274649"
    ).mock(
        return_value=Response(
            200,
            json={
                "ico": "45274649",
                "obchodniJmeno": "ČEZ, a. s.",
                "pravniForma": "121",
                "dic": "CZ45274649",
                "datumVzniku": "1992-05-06",
                "sidlo": {"textovaAdresa": "Duhová 1444/2, Praha 4"},
                "czNace2008": ["35110"],
            },
        )
    )

    identity, citation, raw = await client.resolve_company("ČEZ")

    assert identity.ico == "45274649"
    assert identity.legal_name == "ČEZ, a. s."
    assert citation.id == "ares-entity"
    assert raw["pravniForma"] == "121"
