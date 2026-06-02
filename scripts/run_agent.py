"""Run any agent from the command line."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Agent Suite - Run AI agents from the command line")


@app.command()
def seo_analyst(domain: str = typer.Option(..., "--domain", "-d", help="Domain to analyze")):
    """Run SEO analysis on a domain."""
    from agents.seo_analyst import SEOAnalystWorkflow
    workflow = SEOAnalystWorkflow(domain)
    result = workflow.run()
    console.print(result)


@app.command()
def lead_scraper(
    industry: str = typer.Option(..., "--industry", "-i", help="Industry to search"),
    location: str = typer.Option("", "--location", "-l", help="Location filter"),
):
    """Scrape and enrich leads."""
    from agents.lead_scraper import LeadScraperWorkflow
    workflow = LeadScraperWorkflow(industry, location)
    result = workflow.run()
    console.print(result)


@app.command()
def research(topic: str = typer.Option("artificial intelligence", "--topic", "-t", help="Research topic")):
    """Run research and news summary."""
    from agents.research_agent import ResearchWorkflow
    workflow = ResearchWorkflow(topic)
    result = workflow.run()
    console.print(result)


@app.command()
def content(
    topic: str = typer.Option(..., "--topic", "-t", help="Content topic"),
    keywords: str = typer.Option("", "--keywords", "-k", help="Comma-separated keywords"),
    format: str = typer.Option("blog", "--format", "-f", help="Content format: blog, social, newsletter"),
):
    """Produce content."""
    from agents.content_production import ContentProductionWorkflow
    workflow = ContentProductionWorkflow()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if format == "social":
        result = workflow.create_social(topic)
    elif format == "newsletter":
        result = workflow.write_newsletter(topic)
    else:
        result = workflow.write_blog(topic, kw_list)
    console.print(result)


@app.command()
def investment(
    criteria: str = typer.Option("high growth, low P/E", "--criteria", "-c", help="Stock screening criteria"),
):
    """Run investment analysis."""
    from agents.investment_analyst import InvestmentAnalystWorkflow
    workflow = InvestmentAnalystWorkflow()
    result = workflow.screen_stocks(criteria)
    console.print(result)


if __name__ == "__main__":
    app()
