from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ai_company_insights.config import get_settings
from ai_company_insights.models import ResearchMode
from ai_company_insights.report_renderer import MarkdownReportRenderer
from ai_company_insights.researcher import CompanyResearcher

app = typer.Typer(help="Rešerše českých firem z veřejných zdrojů s citovanými důkazy.")
console = Console()


@app.callback()
def callback() -> None:
    """Rešerše českých firem podložená důkazy."""


@app.command()
def research(
    company: Annotated[str, typer.Argument(help="Název firmy nebo osmimístné české IČO.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Cesta k JSON výstupu.")
    ] = None,
    markdown_output: Annotated[
        Path | None, typer.Option("--markdown-output", help="Cesta k Markdown reportu.")
    ] = None,
    markdown: Annotated[bool, typer.Option("--markdown/--no-markdown")] = True,
    foundry_synthesis: Annotated[
        bool, typer.Option("--foundry-synthesis/--no-foundry-synthesis")
    ] = True,
    foundry_web_search: Annotated[
        bool, typer.Option("--foundry-web-search/--no-foundry-web-search")
    ] = False,
    search_provider: Annotated[
        str, typer.Option(help="auto, brave, tavily, foundry-web nebo none.")
    ] = "auto",
) -> None:
    settings = get_settings()
    if output is None:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() else "-" for ch in company).strip("-").lower()
        output = settings.output_dir / f"{safe_name or 'company'}-research.json"
    markdown_output = markdown_output or output.with_suffix(".md")
    artifact_dir = markdown_output.with_name(f"{markdown_output.stem}-files")
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    settings = settings.model_copy(
        update={
            "output_artifact_dir": artifact_dir,
            "output_artifact_link_prefix": artifact_dir.name,
        }
    )
    mode = ResearchMode(
        use_foundry_synthesis=foundry_synthesis,
        use_foundry_web_search=foundry_web_search,
        search_provider=search_provider,  # type: ignore[arg-type]
    )
    report = asyncio.run(CompanyResearcher(settings).research(company, mode=mode))
    text = report.model_dump_json(indent=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    console.print(f"[green]Zapsán strukturovaný report s citacemi:[/green] {output}")
    if markdown:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(
            MarkdownReportRenderer(settings).render(report), encoding="utf-8"
        )
        console.print(f"[green]Zapsán čitelný Markdown report:[/green] {markdown_output}")
    console.print(
        json.dumps(report.model_dump(mode="json")["company"], ensure_ascii=False, indent=2)
    )
    if report.registrations_needed:
        console.print("[yellow]Stále chybí registrace nebo konfigurace:[/yellow]")
        for item in report.registrations_needed:
            console.print(f"- {item}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
