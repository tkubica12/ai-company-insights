import respx
from httpx import Response

from ai_company_insights.clients.stocks import StockClient
from ai_company_insights.config import Settings
from ai_company_insights.models import CompanyIdentity


@respx.mock
async def test_stock_client_reads_yahoo_chart_quote() -> None:
    respx.get("https://query1.finance.yahoo.com/v8/finance/chart/CEZ.PR").mock(
        return_value=Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "currency": "CZK",
                                "symbol": "CEZ.PR",
                                "fullExchangeName": "Prague",
                                "regularMarketPrice": 1200.0,
                                "previousClose": 1195.0,
                                "regularMarketVolume": 12345,
                                "regularMarketTime": 1777472261,
                            }
                        }
                    ]
                }
            },
        )
    )
    client = StockClient(Settings(stock_symbol_overrides="45274649=CEZ.PR"))

    quote = await client.get_quote(CompanyIdentity(query="ČEZ", ico="45274649"))

    assert quote is not None
    assert quote.symbol == "CEZ.PR"
    assert quote.currency == "CZK"
    assert quote.regular_market_price == 1200.0
