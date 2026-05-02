from __future__ import annotations

from collections import OrderedDict

import httpx

from ai_company_insights.clients.ares import AresClient
from ai_company_insights.clients.crawler import PageCrawler
from ai_company_insights.clients.documents import DocumentProcessor
from ai_company_insights.clients.news import NewsClient
from ai_company_insights.clients.search import SearchClient
from ai_company_insights.clients.stocks import StockClient
from ai_company_insights.config import Settings
from ai_company_insights.foundry_web_search import FoundryWebSearch
from ai_company_insights.models import (
    Citation,
    CompanyResearchReport,
    CrawledPage,
    Evidence,
    ReportSection,
    ResearchMode,
    SearchResult,
    StockQuote,
    TokenUsage,
)
from ai_company_insights.token_usage import merge_token_usage
from ai_company_insights.utils import host_from_url, truncate


class CompanyResearcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ares = AresClient(settings)
        self._search = SearchClient(settings)
        self._news = NewsClient(settings)
        self._stocks = StockClient(settings)
        self._crawler = PageCrawler(settings)
        self._documents = DocumentProcessor(settings)
        self._foundry_web = FoundryWebSearch(settings)

    async def research(
        self, company: str, mode: ResearchMode | None = None
    ) -> CompanyResearchReport:
        mode = mode or ResearchMode(search_provider=self._settings.search_provider)
        identity, ares_citation, ares_raw = await self._ares.resolve_company(company)
        citations: OrderedDict[str, Citation] = OrderedDict([(ares_citation.id, ares_citation)])

        queries = self._queries(identity.legal_name or company, identity.ico)
        search_results: list[SearchResult] = []
        source_errors: list[str] = []
        if mode.search_provider != "foundry-web":
            for query in queries:
                try:
                    search_results.extend(
                        await self._search.search(
                            query,
                            count=self._settings.max_search_results,
                            provider=mode.search_provider,
                        )
                    )
                except httpx.HTTPError as exc:
                    source_errors.append(f"search:{mode.search_provider}:{query}: {exc}")
        news_results: list[SearchResult] = []
        for query in self._news_queries(identity.legal_name or company):
            try:
                news_results.extend(
                    await self._news.search(query, count=self._settings.max_news_results)
                )
            except httpx.HTTPError as exc:
                source_errors.append(f"news:{query}: {exc}")
        stock_quote = await self._stocks.get_quote(identity)

        unique_results = self._dedupe_results(search_results, limit=self._settings.max_web_results)
        unique_news_results = self._dedupe_results(
            news_results, limit=self._settings.max_news_results
        )
        crawl_results = self._dedupe_results(
            [*unique_results, *unique_news_results], limit=self._settings.max_crawl_pages
        )
        all_result_urls = [str(result.url) for result in [*unique_results, *unique_news_results]]
        result_urls = [str(result.url) for result in crawl_results]
        pages = await self._crawler.crawl(result_urls)
        followup_urls = self._discover_followup_urls(pages, set(all_result_urls))
        if followup_urls:
            followup_pages = await self._crawler.crawl(
                [url for url in followup_urls if not self._looks_like_document_url(url)]
            )
            known_followup_urls = {*all_result_urls, *followup_urls}
            second_followup_urls = self._discover_followup_urls(followup_pages, known_followup_urls)
            if second_followup_urls:
                second_followup_pages = await self._crawler.crawl(
                    [url for url in second_followup_urls if not self._looks_like_document_url(url)]
                )
                followup_urls = [*followup_urls, *second_followup_urls]
                pages = self._dedupe_pages([*pages, *followup_pages, *second_followup_pages])
            else:
                pages = self._dedupe_pages([*pages, *followup_pages])
        document_urls = self._document_candidate_urls(all_result_urls, pages, followup_urls)
        documents = await self._documents.convert_urls(document_urls)

        for idx, result in enumerate(unique_results, start=1):
            citation_id = f"web-{idx}"
            citations[citation_id] = Citation(
                id=citation_id,
                title=result.title,
                url=str(result.url),
                source_type=f"web_search:{result.provider}",
                publisher=host_from_url(str(result.url)),
                snippet=result.snippet,
            )

        for idx, result in enumerate(unique_news_results, start=1):
            citation_id = f"news-{idx}"
            citations[citation_id] = Citation(
                id=citation_id,
                title=result.title,
                url=str(result.url),
                source_type=f"news_search:{result.provider}",
                publisher=host_from_url(str(result.url)),
                snippet=result.snippet,
            )

        for idx, page in enumerate(pages, start=1):
            citation_id = f"page-{idx}"
            citations[citation_id] = Citation(
                id=citation_id,
                title=page.title or page.url,
                url=page.url,
                source_type=f"crawled_page:{page.source}",
                publisher=host_from_url(page.url),
                snippet=truncate(page.markdown, 500),
            )

        for idx, document in enumerate(documents, start=1):
            citation_id = f"document-{idx}"
            citations[citation_id] = Citation(
                id=citation_id,
                title=document.title or document.url,
                url=document.url,
                source_type=f"document:{document.source}",
                publisher=host_from_url(document.url),
                snippet=truncate(document.markdown, 500),
            )

        if stock_quote:
            citations["stock-quote"] = Citation(
                id="stock-quote",
                title=f"Tržní kotace {stock_quote.symbol}",
                url=stock_quote.source_url,
                source_type=f"market_data:{stock_quote.provider}",
                publisher=stock_quote.exchange_name,
                snippet=(
                    f"Cena {stock_quote.regular_market_price} {stock_quote.currency}; "
                    f"předchozí zavírací cena {stock_quote.previous_close}."
                ),
            )

        web_grounding_summary = None
        web_grounding_citations: list[str] = []
        token_usage = TokenUsage()
        web_grounding_error = None
        registrations_needed = self._foundry_web.registrations_needed()
        if mode.use_foundry_web_search and self._settings.bing_web_search_enabled:
            try:
                web_grounding_summary, web_grounding_citations, web_token_usage = (
                    self._foundry_web.ask(
                        f"Prozkoumej českou firmu {identity.legal_name or company} "
                        f"({identity.ico or 'neznámé IČO'}) pomocí aktuálních veřejných webových "
                        "zdrojů. Vrať stručná zjištění v češtině s citacemi."
                    )
                )
                token_usage = merge_token_usage(token_usage, web_token_usage)
                for idx, url in enumerate(web_grounding_citations, start=1):
                    citations[f"foundry-web-{idx}"] = Citation(
                        id=f"foundry-web-{idx}",
                        title=url,
                        url=url,
                        source_type="foundry_web_search",
                        publisher=host_from_url(url),
                    )
            except Exception as exc:
                web_grounding_error = str(exc)
                registrations_needed.append(
                    "Foundry Grounding with Bing je připojené, ale živé volání nástroje selhalo; "
                    "zkontrolujte stav Bing Grounding prostředku a klíč/připojení v Azure."
                )

        analysis_sources = self._analysis_sources(
            unique_results, unique_news_results, pages, documents
        )
        core_sections = [
            self._registry_section(ares_citation.id, ares_raw),
            self._ownership_structure_section(ares_citation.id, ares_raw, analysis_sources),
            self._business_context_section(identity, analysis_sources),
            self._financial_information_section(analysis_sources, stock_quote),
            self._reputation_risk_section(analysis_sources),
            *self._insight_sections(analysis_sources, unique_news_results),
        ]
        opportunity_section = self._opportunities_section(analysis_sources)
        sections = [
            *core_sections,
            opportunity_section,
            self._meeting_questions_section([*core_sections, opportunity_section]),
            self._stock_section(stock_quote),
            self._news_section(unique_news_results),
            self._web_presence_section(unique_results),
            self._crawled_content_section(pages),
            self._document_section(documents),
        ]
        if web_grounding_summary:
            sections.append(
                ReportSection(
                    title="Webové ověření Foundry",
                    summary=web_grounding_summary,
                    evidence=[
                        Evidence(
                            citation_id=cid,
                            claim="Citace z webového ověření",
                            value=citations[cid].url,
                            confidence=0.8,
                        )
                        for cid in citations
                        if cid.startswith("foundry-web-")
                    ],
                )
            )

        report = CompanyResearchReport(
            company=identity,
            executive_summary=self._executive_summary(identity, sections),
            sections=sections,
            citations=list(citations.values()),
            registrations_needed=registrations_needed,
            token_usage=token_usage,
            raw={
                "ares": ares_raw,
                "search_results": [result.model_dump(mode="json") for result in unique_results],
                "news_results": [result.model_dump(mode="json") for result in unique_news_results],
                "stock_quote": stock_quote.model_dump(mode="json") if stock_quote else None,
                "foundry_web_search_error": web_grounding_error,
                "queries": queries,
                "news_queries": self._news_queries(identity.legal_name or company),
                "source_errors": source_errors,
                "followup_urls": followup_urls,
                "document_urls": document_urls,
                "pages": [page.model_dump(mode="json") for page in pages],
                "documents": [document.model_dump(mode="json") for document in documents],
            },
        )

        if mode.use_foundry_synthesis:
            try:
                from ai_company_insights.foundry import FoundrySynthesizer

                synthesized, synthesis_usage = await FoundrySynthesizer(self._settings).synthesize(
                    report
                )
                report.executive_summary = synthesized
                report.token_usage = merge_token_usage(report.token_usage, synthesis_usage)
            except Exception as exc:
                report.raw["foundry_synthesis_error"] = str(exc)
        return report

    def _queries(self, legal_name: str, ico: str | None) -> list[str]:
        queries = [
            f'"{legal_name}" oficiální web',
            f'"{legal_name}" obchodní rejstřík statutární orgán vlastnická struktura',
            f'"{legal_name}" profil obor hlavní činnost zaměstnanci',
            f'"{legal_name}" výroční zpráva annual report 2024',
            f'"{legal_name}" účetní závěrka hospodářské výsledky tržby EBITDA',
            f'"{legal_name}" investor relations strategie',
            f'"{legal_name}" produkty služby',
            f'"{legal_name}" ESG dekarbonizace obnovitelné jaderná energetika',
            f'"{legal_name}" SMR Rolls-Royce Temelín',
            f'"{legal_name}" AI prediktivní údržba diagnostika',
            f'"{legal_name}" akvizice partnerství smlouvy kontrakty',
            f'"{legal_name}" investice expanze nové oblasti změna vedení',
            f'"{legal_name}" žaloby insolvence sankce veřejná kauza',
            f'"{legal_name}" akcie burza Praha',
        ]
        if ico:
            queries.append(f'"{legal_name}" IČO {ico}')
        queries.extend([f'"{legal_name}" zprávy', f'"{legal_name}" justice insolvence'])
        return queries

    def _news_queries(self, legal_name: str) -> list[str]:
        return [
            legal_name,
            f'"{legal_name}" ČEZ',
            f'"{legal_name}" Rolls-Royce SMR',
            f'"{legal_name}" větrné elektrárny',
            f'"{legal_name}" umělá inteligence',
            f'"{legal_name}" výsledky akcie',
            f'"{legal_name}" investice akvizice expanze vedení',
            f'"{legal_name}" kauza soud insolvence sankce',
        ]

    def _dedupe_results(
        self, results: list[SearchResult], *, limit: int | None = None
    ) -> list[SearchResult]:
        seen: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            url = str(result.url)
            title = " ".join(result.title.casefold().split())
            if self._is_low_value_result(result):
                continue
            if url in seen or title in seen_titles:
                continue
            seen.add(url)
            seen_titles.add(title)
            unique.append(result)
            if len(unique) >= (limit or self._settings.max_crawl_pages):
                break
        return unique

    def _dedupe_pages(self, pages: list[CrawledPage]) -> list[CrawledPage]:
        seen: set[str] = set()
        unique = []
        for page in pages:
            if page.url in seen:
                continue
            seen.add(page.url)
            unique.append(page)
        return unique

    def _discover_followup_urls(self, pages: list[CrawledPage], known_urls: set[str]) -> list[str]:
        urls: list[str] = []
        seen = set(known_urls)
        for page in pages:
            if not self._is_official_cez_source(page.url):
                continue
            page_context = f"{page.url} {page.title or ''}".casefold()
            if not self._is_followup_source_page(page_context):
                continue
            for link in getattr(page, "links", []) or []:
                if link in seen or not self._is_official_cez_source(link):
                    continue
                if not (
                    self._looks_like_document_url(link)
                    or self._looks_like_official_announcement_url(link)
                ):
                    continue
                seen.add(link)
                urls.append(link)
                if len(urls) >= self._settings.max_followup_pages:
                    return urls
        return urls

    def _document_candidate_urls(
        self, result_urls: list[str], pages: list[CrawledPage], followup_urls: list[str]
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for url in [*result_urls, *followup_urls]:
            if self._looks_like_document_url(url) and url not in seen:
                seen.add(url)
                candidates.append(url)
        for page in pages:
            if not self._is_official_cez_source(page.url):
                continue
            for link in getattr(page, "links", []) or []:
                if self._looks_like_document_url(link) and link not in seen:
                    seen.add(link)
                    candidates.append(link)
        return candidates

    def _looks_like_document_url(self, url: str) -> bool:
        lowered = url.casefold().split("?", 1)[0]
        return lowered.endswith((".pdf", ".docx", ".pptx", ".xlsx"))

    def _is_official_cez_source(self, url: str) -> bool:
        host = host_from_url(url).casefold()
        return host == "cez.cz" or host.endswith(".cez.cz")

    def _is_followup_source_page(self, text: str) -> bool:
        markers = [
            "vyrocni",
            "výroční",
            "annual",
            "report",
            "zpravy",
            "zprávy",
            "news",
            "aktual",
            "inside-information",
            "pro-investory",
            "investors",
        ]
        return any(marker in text for marker in markers)

    def _looks_like_official_announcement_url(self, url: str) -> bool:
        lowered = url.casefold()
        markers = [
            "/news",
            "/aktual",
            "/inside-information",
            "/tiskove-zpravy",
            "/media/",
            "/pro-investory/",
            "/investors/",
        ]
        return any(marker in lowered for marker in markers)

    def _is_low_value_result(self, result: SearchResult) -> bool:
        title = result.title.casefold()
        url = str(result.url).casefold()
        low_value_url_parts = [
            "/about-web",
            "/informace-o-webu",
            "/cookies",
            "/cookie",
            "/privacy",
            "/personal-data",
            "/terms-of-use",
            "/gdpr",
        ]
        low_value_titles = [
            "website information",
            "cookie",
            "privacy policy",
            "personal data",
            "terms of use",
        ]
        return any(part in url for part in low_value_url_parts) or any(
            marker in title for marker in low_value_titles
        )

    def _registry_section(self, citation_id: str, raw: dict) -> ReportSection:
        registrations = raw.get("seznamRegistraci") or {}
        active_sources = [
            name
            for name, state in registrations.items()
            if isinstance(state, str) and state == "AKTIVNI"
        ]
        summary = (
            f"ARES identifikuje {raw.get('obchodniJmeno')} s IČO {raw.get('ico')} "
            f"a sídlem {(raw.get('sidlo') or {}).get('textovaAdresa')}. "
            f"Aktivní registry: {', '.join(active_sources[:10]) or 'neuvedeno'}."
        )
        return ReportSection(
            title="Identita v registrech",
            summary=summary,
            evidence=[
                Evidence(
                    citation_id=citation_id,
                    claim="Obchodní firma",
                    value=raw.get("obchodniJmeno"),
                    confidence=0.98,
                ),
                Evidence(
                    citation_id=citation_id, claim="IČO", value=raw.get("ico"), confidence=0.98
                ),
                Evidence(
                    citation_id=citation_id,
                    claim="Sídlo",
                    value=(raw.get("sidlo") or {}).get("textovaAdresa"),
                    confidence=0.95,
                ),
            ],
        )

    def _insight_sections(
        self,
        sources: list[tuple[str, str, str]],
        news_results: list[SearchResult],
    ) -> list[ReportSection]:
        return [
            self._strategy_section(sources),
            self._products_and_innovation_section(sources),
            self._notable_deals_section(news_results),
            self._industry_trends_section(sources),
            self._sentiment_section(news_results),
        ]

    def _business_context_section(
        self, identity, sources: list[tuple[str, str, str]]
    ) -> ReportSection:
        evidence: list[Evidence] = []
        if identity.nace_codes:
            evidence.append(
                Evidence(
                    citation_id=(
                        identity.source_citation_ids[0]
                        if identity.source_citation_ids
                        else "ares-entity"
                    ),
                    claim="Obor působení podle NACE",
                    value=", ".join(identity.nace_codes[:12]),
                    confidence=0.9,
                )
            )
        themes = [
            (
                "Hlavní činnost podle veřejných zdrojů",
                [
                    "předmětem podnikání",
                    "hlavní činnost",
                    "výroba",
                    "distribuce",
                    "distribution",
                    "products",
                    "služb",
                    "energet",
                ],
            ),
            (
                "Velikost a tržní dosah",
                [
                    "zaměstnanc",
                    "customers",
                    "zákazník",
                    "tržní kapitaliz",
                    "market cap",
                    "instalovan",
                    "skupina",
                ],
            ),
            (
                "Skupinový profil a role ve skupině",
                ["skupina", "group", "dceřin", "subsidiar", "holding", "mateřsk"],
            ),
        ]
        evidence.extend(self._theme_evidence(sources, themes, confidence=0.68))
        if not evidence:
            return ReportSection(
                title="Byznysový kontext",
                summary=(
                    "Ze shromážděných veřejných zdrojů se nepodařilo spolehlivě odvodit "
                    "obor, velikost ani hlavní činnost firmy."
                ),
                evidence=[],
            )
        summary = (
            "Byznysový kontext shrnuje obor, hlavní činnost, velikost a skupinový profil "
            "firmy z registrů a veřejných zdrojů."
        )
        return ReportSection(title="Byznysový kontext", summary=summary, evidence=evidence[:6])

    def _ownership_structure_section(
        self, ares_citation_id: str, raw: dict, sources: list[tuple[str, str, str]]
    ) -> ReportSection:
        registrations = raw.get("seznamRegistraci") or {}
        active_sources = [
            name
            for name, state in registrations.items()
            if isinstance(state, str) and state == "AKTIVNI"
        ]
        evidence = [
            Evidence(
                citation_id=ares_citation_id,
                claim="Obchodní rejstřík a aktivní registrace",
                value=", ".join(active_sources[:10]) or "Aktivní registrace nebyly v ARES uvedeny.",
                confidence=0.9,
            )
        ]
        themes = [
            (
                "Statutární orgány a vedení",
                [
                    "statutár",
                    "představenstv",
                    "dozorčí rada",
                    "management",
                    "board",
                    "členem představenstva",
                    "vedení",
                ],
            ),
            (
                "Vlastnická struktura a akcionáři",
                [
                    "akcionář",
                    "akcionar",
                    "shareholder",
                    "vlastnick",
                    "podíl",
                    "základního kapitálu",
                    "capital",
                    "stát",
                ],
            ),
            (
                "Skupinová struktura",
                ["skupina", "group", "dceřin", "subsidiar", "holding", "mateřsk"],
            ),
        ]
        evidence.extend(self._theme_evidence(sources, themes, confidence=0.7))
        summary = (
            "Sekce soustřeďuje základní registry a dostupné signály o orgánech, "
            "vlastnictví a skupinové struktuře. Detail statutárů je vhodné v případě "
            "potřeby ověřit v úplném výpisu z obchodního rejstříku."
        )
        return ReportSection(
            title="Vlastnická a skupinová struktura",
            summary=summary,
            evidence=evidence[:6],
        )

    def _financial_information_section(
        self, sources: list[tuple[str, str, str]], stock_quote: StockQuote | None
    ) -> ReportSection:
        themes = [
            (
                "Výroční zprávy a účetní závěrky",
                ["výroční zpráva", "annual report", "účetní závěr", "financial report"],
            ),
            (
                "Vývoj hospodaření a výsledky",
                ["výsledky", "hospodaření", "revenues", "tržby", "ebitda", "čistý zisk", "profit"],
            ),
            (
                "Dividenda a kapitálový trh",
                ["dividend", "akcie", "burza", "market cap", "tržní kapitaliz"],
            ),
        ]
        evidence = self._theme_evidence(sources, themes, confidence=0.72)
        if stock_quote:
            value = (
                f"{stock_quote.symbol}: poslední cena {stock_quote.regular_market_price} "
                f"{stock_quote.currency or ''}; předchozí závěr {stock_quote.previous_close}."
            )
            evidence.append(
                Evidence(
                    citation_id="stock-quote",
                    claim="Orientační akciová metrika",
                    value=value.strip(),
                    confidence=0.7,
                )
            )
        if not evidence:
            return ReportSection(
                title="Finanční informace",
                summary=(
                    "V dostupných zdrojích nebyly v nastavených limitech nalezeny výroční "
                    "zprávy, účetní závěrky ani jiné finanční signály."
                ),
                evidence=[],
            )
        summary = (
            "Finanční část prioritizuje výroční zprávy, účetní závěrky, vývoj hospodaření "
            "a kapitálové tržní signály dostupné z veřejných zdrojů."
        )
        return ReportSection(title="Finanční informace", summary=summary, evidence=evidence[:6])

    def _reputation_risk_section(self, sources: list[tuple[str, str, str]]) -> ReportSection:
        themes = [
            (
                "Insolvenční, soudní nebo sankční signály",
                ["insolv", "žalob", "soudní spor", "pokut", "sankc", "antimonopol", "úohs"],
            ),
            (
                "Veřejné kauzy a politická citlivost",
                ["kauz", "kriti", "polit", "zestátn", "ovládnutí", "menšin", "referendum"],
            ),
            (
                "Regulatorní nebo reputační mediální riziko",
                ["regulator", "odpůr", "nadhodnoc", "předražen", "rizik", "kritika"],
            ),
        ]
        risk_sources = self._prioritized_sources(sources, ("news-", "web-", "page-", "document-"))
        evidence = self._theme_evidence(risk_sources, themes, confidence=0.62)
        if not evidence:
            return ReportSection(
                title="Reputační rizika",
                summary=(
                    "Ve shromážděném vzorku nebyly identifikovány explicitní insolvenční, "
                    "soudní, sankční nebo reputační rizikové signály."
                ),
                evidence=[],
            )
        summary = (
            "Reputační část zvýrazňuje mediální zmínky, veřejné kauzy a právní či "
            "insolvenční signály, které stojí za manuální kontrolu před obchodním jednáním."
        )
        return ReportSection(title="Reputační rizika", summary=summary, evidence=evidence[:6])

    def _opportunities_section(self, sources: list[tuple[str, str, str]]) -> ReportSection:
        themes = [
            (
                "Investiční záměry a expanze",
                [
                    "investič",
                    "investic",
                    "investment",
                    "expanz",
                    "moderniz",
                    "výstavb",
                    "větr",
                    "renewable",
                    "obnoviteln",
                    "smr",
                    "jadern",
                ],
            ),
            (
                "Akvizice, partnerství nebo vstup do nových oblastí",
                [
                    "akviz",
                    "partner",
                    "dohod",
                    "rolls-royce",
                    "framatome",
                    "nové oblasti",
                    "esco",
                ],
            ),
            (
                "Změna vedení, governance nebo vlastnické struktury",
                [
                    "novým členem",
                    "představenstv",
                    "governance",
                    "vlastnické struktury",
                    "akcionář",
                    "zestátn",
                    "ovládn",
                    "restrukturaliz",
                ],
            ),
            (
                "Digitalizace, AI a provozní efektivita",
                ["digitaliz", "artificial intelligence", "umělé intelig", "prediktiv", "neuron"],
            ),
        ]
        opportunity_sources = self._prioritized_sources(
            sources, ("news-", "page-", "document-", "web-")
        )
        evidence = self._theme_evidence(opportunity_sources, themes, confidence=0.66)
        if not evidence:
            return ReportSection(
                title="Potenciální příležitosti",
                summary=(
                    "Ze shromážděných zdrojů nebyly v nastavených limitech odvozeny "
                    "konkrétní obchodní příležitosti."
                ),
                evidence=[],
            )
        summary = (
            "Potenciální příležitosti jsou odvozené ze signálů expanze, investic, partnerství, "
            "změn vedení nebo vstupu do nových oblastí."
        )
        return ReportSection(title="Potenciální příležitosti", summary=summary, evidence=evidence)

    def _meeting_questions_section(self, sections: list[ReportSection]) -> ReportSection:
        priority_titles = [
            "Potenciální příležitosti",
            "Strategie a priority",
            "Finanční informace",
            "Reputační rizika",
            "Vlastnická a skupinová struktura",
            "Produkty, služby a inovace",
        ]
        source_evidence = []
        for title in priority_titles:
            source_evidence.extend(
                evidence
                for section in sections
                if section.title == title
                for evidence in section.evidence
            )
        evidence = [
            Evidence(
                citation_id=item.citation_id,
                claim=self._meeting_question_for_claim(item.claim),
                value=(
                    f"Navazuje na zjištění '{item.claim}'. Cílem je ověřit prioritu, časování, "
                    "rozpočet, rozhodovací proces a prostor pro relevantní nabídku."
                ),
                confidence=min(item.confidence, 0.7),
            )
            for item in source_evidence[:6]
        ]
        if not evidence:
            return ReportSection(
                title="Otázky pro první obchodní schůzku",
                summary=(
                    "Bez konkrétních signálů ve zdrojích nelze navrhnout dostatečně ukotvené "
                    "otázky pro první obchodní schůzku."
                ),
                evidence=[],
            )
        return ReportSection(
            title="Otázky pro první obchodní schůzku",
            summary=(
                "Návrh otázek převádí zjištěné veřejné signály na obchodní témata pro první "
                "schůzku."
            ),
            evidence=evidence,
        )

    def _analysis_sources(
        self,
        web_results: list[SearchResult],
        news_results: list[SearchResult],
        pages: list,
        documents: list,
    ) -> list[tuple[str, str, str]]:
        sources: list[tuple[str, str, str]] = []
        for idx, result in enumerate(web_results, start=1):
            sources.append((f"web-{idx}", result.title, self._source_note(result)))
        for idx, result in enumerate(news_results, start=1):
            sources.append((f"news-{idx}", result.title, self._source_note(result)))
        for idx, page in enumerate(pages, start=1):
            sources.append((f"page-{idx}", page.title or page.url, page.markdown))
        for idx, document in enumerate(documents, start=1):
            sources.append((f"document-{idx}", document.title or document.url, document.markdown))
        return sources

    def _strategy_section(self, sources: list[tuple[str, str, str]]) -> ReportSection:
        themes = [
            (
                "Dekarbonizace a přechod k čistší energetice",
                [
                    "decarbon",
                    "clean energy",
                    "sustainable",
                    "sustainability",
                    "obnoviteln",
                    "bezemis",
                    "energy for the future",
                    "větr",
                    "wind",
                ],
            ),
            (
                "Jaderná energetika a možnost malých modulárních reaktorů",
                ["nuclear", "jadern", "smr", "reactor", "reaktor", "temelín", "rolls-royce"],
            ),
            (
                "Rozsah a spolehlivost distribuční sítě",
                ["distribuce", "distribution", "distributor", "customers", "power lines"],
            ),
            (
                "Digitalizace a prediktivní údržba",
                ["uměl", "neuron", "prediktiv", "diagnost", "artificial intelligence", "digital"],
            ),
        ]
        evidence = self._theme_evidence(sources, themes, confidence=0.72)
        if not evidence:
            return ReportSection(
                title="Strategie a priority",
                summary=(
                    "Ze shromážděných veřejných zdrojů se nepodařilo odvodit strategická témata."
                ),
                evidence=[],
            )
        summary = (
            "Shromážděné zdroje ukazují na tato strategická témata: "
            + "; ".join(item.claim for item in evidence)
            + "."
        )
        return ReportSection(title="Strategie a priority", summary=summary, evidence=evidence)

    def _products_and_innovation_section(
        self, sources: list[tuple[str, str, str]]
    ) -> ReportSection:
        themes = [
            (
                "Distribuce elektřiny a služby sítí",
                ["distribuce", "distribution", "distributor", "electricity supply"],
            ),
            (
                "Energetické služby a dekarbonizace pro zákazníky",
                ["esco", "decarbon", "zárukami původu", "renewable", "bezemis"],
            ),
            (
                "Kompetence v obchodování s energiemi",
                ["trading", "trade on our own account", "traders", "analysts"],
            ),
            (
                "Prediktivní diagnostika využívající AI",
                ["neuron", "prediktivní diagnost", "umělé inteligenci", "artificial intelligence"],
            ),
        ]
        evidence = self._theme_evidence(sources, themes, confidence=0.7)
        if not evidence:
            return ReportSection(
                title="Produkty, služby a inovace",
                summary=(
                    "Ve shromážděných zdrojích nebyla identifikována témata produktů, služeb "
                    "nebo inovací."
                ),
                evidence=[],
            )
        summary = (
            "Veřejné zdroje naznačují aktivitu v oblastech: "
            + "; ".join(item.claim.casefold() for item in evidence)
            + "."
        )
        return ReportSection(
            title="Produkty, služby a inovace",
            summary=summary,
            evidence=evidence,
        )

    def _notable_deals_section(self, news_results: list[SearchResult]) -> ReportSection:
        evidence: list[Evidence] = []
        for idx, result in enumerate(news_results, start=1):
            label = self._news_deal_label(result)
            if not label:
                continue
            evidence.append(
                Evidence(
                    citation_id=f"news-{idx}",
                    claim=label,
                    value=self._source_note(result),
                    confidence=0.68,
                )
            )
        if not evidence:
            return ReportSection(
                title="Významná oznámení a obchody",
                summary=(
                    "V mediálních výsledcích nebyla identifikována významná nedávná oznámení "
                    "nebo obchody."
                ),
                evidence=[],
            )
        summary = (
            "Nedávné mediální výsledky zvýrazňují: "
            + "; ".join(item.claim.casefold() for item in evidence[:4])
            + "."
        )
        return ReportSection(
            title="Významná oznámení a obchody",
            summary=summary,
            evidence=evidence[:6],
        )

    def _industry_trends_section(self, sources: list[tuple[str, str, str]]) -> ReportSection:
        themes = [
            (
                "Malé modulární reaktory a plánování jaderných kapacit",
                ["smr", "modulár", "reactor", "reaktor", "rolls-royce", "temelín"],
            ),
            (
                "Rozvoj větrné energetiky a lokální povolování/přijetí",
                ["větr", "wind", "mikroregion", "obce", "svitav"],
            ),
            (
                "Dekarbonizace zákazníků a záruky původu",
                ["zárukami původu", "decarbon", "bezemis", "obnoviteln"],
            ),
            (
                "AI a prediktivní diagnostika energetických aktiv",
                ["prediktiv", "diagnost", "neuron", "umělé inteligenci"],
            ),
        ]
        evidence = self._theme_evidence(sources, themes, confidence=0.68)
        if not evidence:
            return ReportSection(
                title="Oborové trendy",
                summary="Nebyly identifikovány signály zapojení do oborových trendů.",
                evidence=[],
            )
        summary = (
            "Shromážděné zdroje firmu zasazují do širších oborových trendů kolem témat: "
            + "; ".join(item.claim.casefold() for item in evidence)
            + "."
        )
        return ReportSection(
            title="Oborové trendy",
            summary=summary,
            evidence=evidence,
        )

    def _sentiment_section(self, news_results: list[SearchResult]) -> ReportSection:
        evidence: list[Evidence] = []
        for idx, result in enumerate(news_results, start=1):
            label = self._sentiment_label(result)
            if not label:
                continue
            evidence.append(
                Evidence(
                    citation_id=f"news-{idx}",
                    claim=label,
                    value=self._source_note(result),
                    confidence=0.62,
                )
            )
        if not evidence:
            return ReportSection(
                title="Mediální sentiment",
                summary="Ze shromážděného vzorku zpráv nebyly identifikovány sentimentové signály.",
                evidence=[],
            )
        positive = sum(1 for item in evidence if item.claim.startswith("Pozitivní"))
        risk = sum(1 for item in evidence if item.claim.startswith("Rizikový"))
        neutral = len(evidence) - positive - risk
        summary = (
            "Tón zkoumaného mediálního vzorku působí "
            f"{'spíše konstruktivně' if positive >= risk else 'smíšeně'}: "
            f"detekováno {positive} pozitivních/strategických signálů, "
            f"{neutral} neutrálních signálů a {risk} rizikových signálů."
        )
        return ReportSection(title="Mediální sentiment", summary=summary, evidence=evidence[:6])

    def _theme_evidence(
        self,
        sources: list[tuple[str, str, str]],
        themes: list[tuple[str, list[str]]],
        *,
        confidence: float,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        used_citations: set[str] = set()
        for label, keywords in themes:
            match = self._first_keyword_match(sources, keywords, used_citations)
            if not match:
                continue
            citation_id, _title, text = match
            used_citations.add(citation_id)
            evidence.append(
                Evidence(
                    citation_id=citation_id,
                    claim=label,
                    value=self._focused_excerpt(text, keywords),
                    confidence=confidence,
                )
            )
        return evidence

    def _prioritized_sources(
        self, sources: list[tuple[str, str, str]], prefixes: tuple[str, ...]
    ) -> list[tuple[str, str, str]]:
        prioritized: list[tuple[str, str, str]] = []
        used: set[str] = set()
        for prefix in prefixes:
            for source in sources:
                citation_id = source[0]
                if citation_id.startswith(prefix) and citation_id not in used:
                    prioritized.append(source)
                    used.add(citation_id)
        for source in sources:
            citation_id = source[0]
            if citation_id not in used:
                prioritized.append(source)
        return prioritized

    def _first_keyword_match(
        self,
        sources: list[tuple[str, str, str]],
        keywords: list[str],
        used_citations: set[str],
    ) -> tuple[str, str, str] | None:
        for citation_id, title, text in sources:
            if citation_id in used_citations:
                continue
            haystack = f"{title} {text}".casefold()
            if any(keyword.casefold() in haystack for keyword in keywords):
                return citation_id, title, text
        return None

    def _focused_excerpt(self, text: str, keywords: list[str]) -> str:
        compact = " ".join(text.split())
        lowered = compact.casefold()
        for keyword in keywords:
            index = lowered.find(keyword.casefold())
            if index >= 0:
                start = max(index - 80, 0)
                end = min(index + 240, len(compact))
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(compact) else ""
                return prefix + compact[start:end].strip() + suffix
        return truncate(compact, 280)

    def _news_deal_label(self, result: SearchResult) -> str | None:
        haystack = f"{result.title} {result.snippet or ''}".casefold()
        if any(marker in haystack for marker in ["rolls-royce", "smr", "modulár"]):
            return "Partnerství v SMR a jaderná opce"
        if any(marker in haystack for marker in ["framatome", "palivo", "dukovany"]):
            return "Jaderné palivo a dodavatelský řetězec Dukovan"
        if any(marker in haystack for marker in ["khnp", "nových bloků", "dostavbu"]):
            return "Harmonogram velké jaderné výstavby"
        if any(marker in haystack for marker in ["větr", "wind"]):
            return "Rozvoj větrné energetiky s obcemi"
        if any(marker in haystack for marker in ["ai", "neuron", "prediktiv"]):
            return "Iniciativa prediktivní údržby s AI"
        if any(marker in haystack for marker in ["zestátn", "ovládnutí", "menšin"]):
            return "Možná restrukturalizace vlastnictví státem"
        if any(marker in haystack for marker in ["dividend", "dividendu"]):
            return "Signál dividendy a výnosu pro akcionáře"
        if any(marker in haystack for marker in ["moody", "rating"]):
            return "Signál zlepšení úvěrového ratingu"
        if any(marker in haystack for marker in ["zemního plynu", "gas", "plyn"]):
            return "Dodávky plynu a přechodové palivo"
        if any(marker in haystack for marker in ["dohod", "smlouv", "partner"]):
            return "Signál partnerství nebo dohody"
        return None

    def _meeting_question_for_claim(self, claim: str) -> str:
        normalized = claim.casefold()
        if any(marker in normalized for marker in ["invest", "expanz", "výstavb", "moderniz"]):
            return "Jaké investiční a rozvojové priority mají nejvyšší obchodní relevanci?"
        if any(marker in normalized for marker in ["akviz", "partner", "dohod", "nových oblast"]):
            return "Kde hledáte partnery nebo dodavatele pro rozvoj nových aktivit?"
        if any(
            marker in normalized
            for marker in ["vedení", "governance", "vlastnick", "akcionář", "představenstv"]
        ):
            return "Jak změny ve vedení, governance nebo vlastnictví ovlivní nákupní priority?"
        if any(marker in normalized for marker in ["finan", "výsledk", "dividend", "akci"]):
            return (
                "Jak se finanční plán a kapitálové priority promítají do investičních rozhodnutí?"
            )
        if any(marker in normalized for marker in ["rizik", "soud", "insolv", "kauz", "sankc"]):
            return (
                "Která reputační, právní nebo regulatorní rizika je potřeba při spolupráci "
                "zohlednit?"
            )
        if any(marker in normalized for marker in ["digital", "ai", "prediktiv", "diagnost"]):
            return "Kde má digitalizace nebo AI největší prostor pro rychle měřitelný přínos?"
        if any(marker in normalized for marker in ["dekarbon", "obnoviteln", "jadern", "smr"]):
            return "Které transformační projekty budou vyžadovat externí kapacity nebo technologie?"
        return (
            "Jaké priority, problémy a rozhodovací kritéria z tohoto zjištění plynou "
            "pro spolupráci?"
        )

    def _sentiment_label(self, result: SearchResult) -> str | None:
        haystack = f"{result.title} {result.snippet or ''}".casefold()
        if any(
            marker in haystack
            for marker in [
                "žalob",
                "soud",
                "insolv",
                "pokut",
                "kriti",
                "zestátn",
                "nadhodnoc",
                "předražen",
                "drahá",
                "odpůrci",
                "referendum",
                "politiku",
            ]
        ):
            return "Rizikový signál v mediálním pokrytí"
        if any(
            marker in haystack
            for marker in [
                "ai",
                "neuron",
                "smr",
                "rolls-royce",
                "větr",
                "partner",
                "dohod",
                "moody",
                "rating",
                "dividend",
            ]
        ):
            return "Pozitivní strategický nebo inovační signál"
        if any(marker in haystack for marker in ["babiš", "premiér", "vlád", "delegac"]):
            return "Neutrální veřejně-politický signál"
        return "Neutrální mediální zmínka"

    def _web_presence_section(self, results: list[SearchResult]) -> ReportSection:
        if not results:
            return ReportSection(
                title="Webové zdroje",
                summary=(
                    "Nebyly shromážděny obecné výsledky webového vyhledávání. To může nastat, "
                    "pokud běh používá pouze mediální API, Foundry grounding nebo "
                    "search_provider=none."
                ),
                evidence=[],
            )
        evidence = [
            Evidence(
                citation_id=f"web-{idx}",
                claim=self._web_finding_label(result),
                value=self._source_note(result),
                confidence=0.65,
            )
            for idx, result in enumerate(results, start=1)
        ]
        hosts = ", ".join(sorted({host_from_url(str(result.url)) for result in results})[:8])
        return ReportSection(
            title="Webové zdroje",
            summary=(
                f"Vyhledávání našlo {len(results)} kandidátních veřejných zdrojů napříč: {hosts}."
            ),
            evidence=evidence,
        )

    def _news_section(self, results: list[SearchResult]) -> ReportSection:
        if not results:
            return ReportSection(
                title="Zprávy a média",
                summary="Nebyly shromážděny žádné výsledky z novinových nebo mediálních API.",
                evidence=[],
            )
        evidence = [
            Evidence(
                citation_id=f"news-{idx}",
                claim=result.title,
                value=self._source_note(result),
                confidence=0.65,
            )
            for idx, result in enumerate(results, start=1)
        ]
        providers = ", ".join(sorted({result.provider for result in results})[:8])
        return ReportSection(
            title="Zprávy a média",
            summary=f"Shromážděno {len(results)} kandidátních mediálních zdrojů přes: {providers}.",
            evidence=evidence,
        )

    def _stock_section(self, quote: StockQuote | None) -> ReportSection:
        if not quote:
            return ReportSection(
                title="Akciové informace",
                summary=(
                    "Nebyla shromážděna akciová kotace. Pokud je firma veřejně obchodovaná, "
                    "přidejte veřejný ticker do STOCK_SYMBOL_OVERRIDES."
                ),
                evidence=[],
            )
        values = [
            ("Symbol", quote.symbol),
            ("Burza", quote.exchange_name),
            ("Měna", quote.currency),
            ("Poslední běžná tržní cena", str(quote.regular_market_price)),
            ("Předchozí závěr", str(quote.previous_close)),
            ("Denní rozpětí", f"{quote.day_low} - {quote.day_high}"),
            ("Rozpětí 52 týdnů", f"{quote.fifty_two_week_low} - {quote.fifty_two_week_high}"),
            ("Objem", str(quote.volume)),
            ("Počet akcií použitý pro odhad", str(quote.shares_outstanding)),
            (
                "Odhadovaná tržní hodnota",
                (
                    f"{quote.estimated_market_cap:,.0f} {quote.currency}"
                    if quote.estimated_market_cap and quote.currency
                    else None
                ),
            ),
        ]
        return ReportSection(
            title="Akciové informace",
            summary=(
                f"Byla shromážděna veřejná tržní kotace bez klíče pro {quote.symbol} "
                f"od {quote.provider}. Berte ji jako orientační tržní údaj; tržní hodnota "
                "se odhaduje pouze tehdy, když je nakonfigurovaný počet akcií."
            ),
            evidence=[
                Evidence(
                    citation_id="stock-quote",
                    claim=label,
                    value=value,
                    confidence=0.7,
                )
                for label, value in values
                if value and value != "None"
            ],
        )

    def _crawled_content_section(self, pages: list) -> ReportSection:
        if not pages:
            return ReportSection(
                title="Extrahovaný obsah zdrojů",
                summary=(
                    "Žádné zdrojové stránky neposkytly v nastavených limitech užitečný "
                    "extrahovaný obsah. Cookie/consent-wall stránky jsou odfiltrovány."
                ),
                evidence=[],
            )
        evidence = [
            Evidence(
                citation_id=f"page-{idx}",
                claim=page.title or page.url,
                value=truncate(page.markdown, 1200),
                confidence=0.7,
            )
            for idx, page in enumerate(pages, start=1)
        ]
        return ReportSection(
            title="Extrahovaný obsah zdrojů",
            summary=(
                f"Extrahován užitečný text z {len(pages)} veřejných stránek. Tyto řádky jsou "
                "výňatky ze zdrojů pro dohledatelnost, nikoli syntetizované obchodní závěry."
            ),
            evidence=evidence,
        )

    def _source_note(self, result: SearchResult) -> str:
        snippet = self._clean_snippet(result.snippet or "")
        if snippet:
            return truncate(snippet, 220)
        return f"Zdroj nalezen přes {result.provider}: {result.url}"

    def _clean_snippet(self, value: str) -> str:
        snippet = " ".join(value.split())
        noisy_prefixes = [
            "Skip to Content",
            "Group CEZGroup CEZ EN CZDEFR",
            "About usCEZ Group",
        ]
        for prefix in noisy_prefixes:
            snippet = snippet.replace(prefix, "").strip()
        return snippet

    def _web_finding_label(self, result: SearchResult) -> str:
        haystack = f"{result.title} {result.url}".casefold()
        if any(marker in haystack for marker in ["výroční", "annual", "zprava", "report"]):
            return "Zdroj finančního reportingu nebo veřejného podání"
        if "distribuce" in haystack:
            return "Stopa distribučního podnikání"
        if "trading" in haystack:
            return "Aktivita v obchodování s energiemi"
        if "esco" in haystack:
            return "Stopa ESCO a dekarbonizačních služeb"
        if any(marker in haystack for marker in ["investor", "news"]):
            return "Investorský nebo zpravodajský publikační kanál"
        if "cez group" in haystack or "skupina čez" in haystack:
            return "Základní profil skupiny a popis podnikání"
        return result.title

    def _document_section(self, documents: list) -> ReportSection:
        if not documents:
            return ReportSection(
                title="Veřejné dokumenty",
                summary=(
                    "V nastavených limitech nebyly převedeny žádné podporované veřejné dokumenty."
                ),
                evidence=[],
            )
        evidence = [
            Evidence(
                citation_id=f"document-{idx}",
                claim=document.title or document.url,
                value=truncate(document.markdown, 1200),
                confidence=0.75,
            )
            for idx, document in enumerate(documents, start=1)
        ]
        return ReportSection(
            title="Veřejné dokumenty",
            summary=f"Převedeno {len(documents)} veřejných dokumentů do Markdownu.",
            evidence=evidence,
        )

    def _executive_summary(self, identity, sections: list[ReportSection]) -> str:
        cited = ", ".join(f"[{cid}]" for cid in identity.source_citation_ids)
        return (
            f"{identity.legal_name or identity.query} byla dohledána v ARES s IČO "
            f"{identity.ico or 'neznámé'} {cited}. "
            f"Úvodní report obsahuje {len(sections)} sekcí podložených důkazy a je vhodné "
            "jej číst jako rešeršní návrh opřený o citované zdroje."
        )
