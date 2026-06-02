"""Lead Scraping + Enrichment Agent.

Scrapes the web for potential leads, enriches contact data,
deduplicates entries, and scores leads by quality.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page, write_csv
from core.storage import Lead


def create_lead_scraper() -> Agent:
    return Agent(
        role="Lead Generation Specialist",
        goal="Find, scrape, and enrich high-quality business leads from the web",
        backstory=(
            "You are an expert lead generation specialist who knows how to find "
            "prospects across LinkedIn, company directories, industry databases, "
            "and public web sources. You excel at data enrichment, deduplication, "
            "and lead scoring. You always verify email formats and company data."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_scraping_task(industry: str, location: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Scrape and find 20+ potential leads in the {industry} industry "
            f"located in {location}. Use web search to find: company websites, "
            "LinkedIn profiles, industry directories, and business listings. "
            "Extract: company name, contact name, email (if public), phone (if public), "
            "website, and LinkedIn URL. Return structured lead data."
        ),
        expected_output="A JSON array of lead objects with fields: company, contact_name, email, phone, website, linkedin_url, source",
        agent=agent,
    )


def create_enrichment_task(agent: Agent) -> Task:
    return Task(
        description=(
            "For each lead found, enrich the data by: verifying the email format, "
            "finding additional contact info from the company website, "
            "checking company size and industry classification via web search, "
            "and adding a lead score from 1-100 based on: company size match, "
            "contact seniority, data completeness, and industry relevance. "
            "Deduplicate entries by email and company domain."
        ),
        expected_output="Enriched lead list with scores, deduplication flags, and data quality ratings",
        agent=agent,
    )


class LeadScraperWorkflow:
    """End-to-end lead scraping + enrichment workflow."""

    def __init__(self, industry: str, location: str = ""):
        self.industry = industry
        self.location = location
        self.agent = create_lead_scraper()

    def run(self) -> str:
        scrape_task = create_scraping_task(self.industry, self.location, self.agent)
        enrich_task = create_enrichment_task(self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[scrape_task, enrich_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
