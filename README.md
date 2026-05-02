# ai-company-insights

Agent pro hloubkovou rešerši českých firem podloženou důkazy. Je postavený v Pythonu s `uv`, Microsoft Agent Framework, Foundry, veřejnými registry, vyhledávacími API a lokálním crawlingem.

## Nastavení

```powershell
uv sync
Copy-Item .env.example .env
az login
```

## Spuštění

```powershell
uv run ai-company-insights research "ČEZ" --output outputs\cez.json --foundry-web-search
```

Výstupní JSON obsahuje strukturované sekce, pole `citations` a `token_usage`. Tvrzení odkazují na citační ID.
CLI ve výchozím stavu zapisuje i čitelný Markdown report vedle JSON výstupu. Cestu lze změnit přes `--markdown-output`, případně Markdown vypnout přes `--no-markdown`.

## Foundry Web Search / Grounding with Bing

Foundry může používat hostované Web Search nebo Grounding with Bing pro agentní odpovědi podložené webem. Doporučený režim je `FOUNDRY_WEB_SEARCH_MODE=hosted`, protože v této subscription funguje bez samostatného Bing prostředku. Použití zapnete přes `BING_WEB_SEARCH_ENABLED=true`.

1. Ověřte, že Azure admin povoluje Foundry Web Search tool.
2. Pokud bude potřeba samostatný Bing prostředek, zaregistrujte resource provider `Microsoft.Bing`.
3. Vytvořte Grounding with Bing nebo Bing Custom Search resource.
4. Připojte resource do Foundry projektu.
5. Nastavte `BING_WEB_SEARCH_ENABLED=true` v `.env`.
6. Pro doménově omezený Bing Custom Search nastavte také `BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID` a `BING_CUSTOM_SEARCH_INSTANCE_NAME`.

`FOUNDRY_WEB_SEARCH_MODE=bing-grounding` používá nakonfigurované připojení `tomaskubica-bing-grounding`, ale tato subscription aktuálně hlásí Bing resources jako suspended. Brave a Tavily zůstávají užitečné fallbacky pro prvotní nalezení URL, které pipeline umí crawlovat, ukládat a citovat.

## Mediální zdroje

Nastavte `NEWSAPI_API_KEY` a `MEDIASTACK_API_KEY` pro doplnění novinových a mediálních výsledků. Tato API doplňují obecné webové vyhledávání; registry jako ARES zůstávají veřejné zdroje bez klíče.

## Šablona Markdown reportu

Čitelný report se vykresluje ze souboru `templates\company_report.md`, takže názvy a pořadí kapitol lze měnit bez úprav promptů nebo Python kódu. Renderer podporuje zástupné symboly jako `{{company_at_glance}}`, `{{executive_summary}}`, `{{resources_table}}`, `{{token_usage}}` a `{{section:Název sekce}}`.

## Akciové informace

Akciová kapitola používá veřejný chart endpoint Yahoo Finance bez klíče, pokud je ticker namapovaný v `STOCK_SYMBOL_OVERRIDES`. Výchozí nastavení obsahuje `45274649=CEZ.PR` pro ČEZ. Pokud `STOCK_SHARES_OUTSTANDING_OVERRIDES` obsahuje odpovídající počet akcií, report odhadne i tržní hodnotu. Data berte jako orientační tržní údaj, nikoli oficiální burzovní feed.

## Lokální crawling

Výchozí crawler používá lehké lokální HTTP + Trafilatura. Nastavením `ENABLE_BROWSER_CRAWLER=true` zapnete projektově spravovaný Crawl4AI browser fallback pro stránky vyžadující JavaScript rendering. Globální MCP server není potřeba. Pokud se později přidá Firecrawl, měl by být nakonfigurovaný jako projektová závislost nebo self-hosted endpoint, ne jako instalace pro celý počítač.
