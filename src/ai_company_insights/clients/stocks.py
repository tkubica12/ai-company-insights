from __future__ import annotations

from datetime import UTC, datetime

import httpx

from ai_company_insights.config import Settings
from ai_company_insights.models import CompanyIdentity, StockQuote
from ai_company_insights.utils import normalize_company_name


class StockClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def get_quote(self, company: CompanyIdentity) -> StockQuote | None:
        if not self._settings.enable_stock_lookup:
            return None
        symbol = self._symbol_for(company)
        if not symbol:
            return None
        shares_outstanding = self._shares_outstanding_for(company)

        source_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.get(
                source_url,
                params={"range": "1d", "interval": "1d"},
                headers={"User-Agent": self._settings.user_agent},
            )
            response.raise_for_status()

        result = ((response.json().get("chart") or {}).get("result") or [None])[0]
        if not result:
            return None
        meta = result.get("meta") or {}
        market_time = meta.get("regularMarketTime")
        price = meta.get("regularMarketPrice")
        return StockQuote(
            symbol=symbol,
            provider="yahoo_finance_chart",
            source_url=source_url,
            currency=meta.get("currency"),
            exchange_name=meta.get("fullExchangeName") or meta.get("exchangeName"),
            regular_market_price=price,
            previous_close=meta.get("previousClose") or meta.get("chartPreviousClose"),
            day_high=meta.get("regularMarketDayHigh"),
            day_low=meta.get("regularMarketDayLow"),
            fifty_two_week_high=meta.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=meta.get("fiftyTwoWeekLow"),
            volume=meta.get("regularMarketVolume"),
            shares_outstanding=shares_outstanding,
            estimated_market_cap=(
                float(price) * shares_outstanding
                if isinstance(price, int | float) and shares_outstanding
                else None
            ),
            market_time=(
                datetime.fromtimestamp(market_time, tz=UTC)
                if isinstance(market_time, int)
                else None
            ),
        )

    def _symbol_for(self, company: CompanyIdentity) -> str | None:
        overrides = self._parse_overrides()
        keys = [
            company.ico,
            normalize_company_name(company.legal_name or ""),
            normalize_company_name(company.query),
        ]
        for key in keys:
            if key and key in overrides:
                return overrides[key]
        return None

    def _shares_outstanding_for(self, company: CompanyIdentity) -> int | None:
        overrides = self._parse_shares_outstanding_overrides()
        keys = [
            company.ico,
            normalize_company_name(company.legal_name or ""),
            normalize_company_name(company.query),
        ]
        for key in keys:
            if key and key in overrides:
                return overrides[key]
        return None

    def _parse_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for item in self._settings.stock_symbol_overrides.split(","):
            if "=" not in item:
                continue
            key, symbol = item.split("=", 1)
            key = key.strip()
            symbol = symbol.strip()
            if not key or not symbol:
                continue
            overrides[key] = symbol
            overrides[normalize_company_name(key)] = symbol
        return overrides

    def _parse_shares_outstanding_overrides(self) -> dict[str, int]:
        overrides: dict[str, int] = {}
        for item in self._settings.stock_shares_outstanding_overrides.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            shares = int(value)
            overrides[key] = shares
            overrides[normalize_company_name(key)] = shares
        return overrides
