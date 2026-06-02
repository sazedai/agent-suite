"""SEO Analyst Agent.

Performs keyword research, competitor analysis, backlink monitoring,
and technical SEO audits for any domain.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page


def create_seo_analyst() -> Agent:
    return Agent(
        role="Senior SEO Analyst",
        goal="Analyze websites and provide actionable SEO insights to improve search rankings",
        backstory=(
            "You are an experienced SEO specialist with 10+ years in digital marketing. "
            "You excel at keyword research, competitor analysis, technical SEO audits, "
            "and backlink analysis. You always provide data-driven, actionable recommendations."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_keyword_research_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Perform comprehensive keyword research for {domain}. "
            "Use web search to find: top ranking keywords, search volume estimates, "
            "related long-tail keywords, competitor keywords, and keyword difficulty. "
            "Return structured keyword data with priority rankings."
        ),
        expected_output="A JSON list of keywords with columns: keyword, estimated_volume, difficulty, priority",
        agent=agent,
    )


def create_competitor_analysis_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze the top 3 SEO competitors for {domain}. "
            "Use web search to identify competitors, then compare: domain authority, "
            "top keywords, content strategy, backlink profile, and technical SEO. "
            "Return a comparison table with actionable gaps."
        ),
        expected_output="A competitor comparison matrix with key metrics and identified opportunities",
        agent=agent,
    )


def create_technical_audit_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Perform a technical SEO audit of {domain}. "
            f"Fetch the homepage at https://{domain} and analyze: page title, meta description, "
            "heading structure, URL structure, mobile-friendliness signals, "
            "page speed hints, schema markup, internal linking, and crawlability. "
            "Return a prioritized list of issues and fixes."
        ),
        expected_output="A technical SEO audit report with issue severity (critical/major/minor) and fix recommendations",
        agent=agent,
    )


class SEOAnalystWorkflow:
    """End-to-end SEO analysis workflow."""

    def __init__(self, domain: str):
        self.domain = domain
        self.agent = create_seo_analyst()

    def run(self, mode: str = "full") -> str:
        """Run the SEO analysis.

        Args:
            mode: ``"keyword"``, ``"competitor"``, ``"technical"``, or ``"full"``
        """
        tasks = []
        if mode in ("keyword", "full"):
            tasks.append(create_keyword_research_task(self.domain, self.agent))
        if mode in ("competitor", "full"):
            tasks.append(create_competitor_analysis_task(self.domain, self.agent))
        if mode in ("technical", "full"):
            tasks.append(create_technical_audit_task(self.domain, self.agent))

        crew = Crew(
            agents=[self.agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
