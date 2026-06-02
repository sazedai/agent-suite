"""Lead Scraping + Enrichment Agent.

Scrapes the web for potential leads, enriches contact data,
deduplicates entries, scores leads by quality, exports to CSV,
and stores results in the database.

Usage:
    from agents.lead_scraper import LeadScraperAgent, LeadEnrichmentPipeline

    agent = LeadScraperAgent(industry="SaaS", location="US")
    leads = agent.scrape()
    enriched = agent.enrich(leads)
    agent.export_csv(enriched, "leads.csv")
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.storage import Lead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ScrapedLead:
    """Intermediate representation of a lead before DB persistence."""
    contact_name: str = ""
    email: str = ""
    company: str = ""
    website: str = ""
    phone: str = ""
    linkedin_url: str = ""
    source: str = ""
    industry: str = ""
    company_size: str = ""
    score: float = 0.0
    enriched: str = "N"
    dedup_key: str = ""
    data_quality: float = 0.0
    tags: str = ""
    raw_data: str = ""

    def __post_init__(self):
        if not self.dedup_key:
            self.dedup_key = self._compute_dedup_key()
        if self.score == 0.0:
            self.score = self._basic_score()
        if self.data_quality == 0.0:
            self.data_quality = self._compute_data_quality()

    def _compute_dedup_key(self) -> str:
        email_part = self.email.lower().strip() if self.email else ""
        company_part = self.company.lower().strip() if self.company else ""
        raw = f"{email_part}|{company_part}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _basic_score(self) -> float:
        score = 0.0
        if self.email and self._validate_email(self.email):
            score += 30
        if self.company:
            score += 20
        if self.linkedin_url:
            score += 15
        if self.phone:
            score += 15
        if self.website:
            score += 10
        if self.contact_name:
            score += 10
        return min(score, 100.0)

    def _compute_data_quality(self) -> float:
        fields = [
            self.contact_name, self.email, self.company,
            self.website, self.phone, self.linkedin_url,
            self.industry, self.company_size,
        ]
        filled = sum(1 for f in fields if f)
        return round(filled / len(fields) * 100, 1)

    @staticmethod
    def _validate_email(email: str) -> bool:
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_lead_model(self) -> Lead:
        return Lead(
            contact_name=self.contact_name,
            email=self.email,
            company=self.company,
            website=self.website,
            phone=self.phone,
            linkedin_url=self.linkedin_url,
            source=self.source,
            industry=self.industry,
            company_size=self.company_size,
            score=self.score,
            enriched=self.enriched,
            dedup_key=self.dedup_key,
            data_quality=self.data_quality,
            tags=self.tags,
            raw_data=self.raw_data,
        )


# ---------------------------------------------------------------------------
# Scraping utilities (sync wrappers for crewAI tools)
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract domain from a URL."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower().lstrip("www.")


def _extract_email_from_text(text: str) -> list[str]:
    """Extract email addresses from raw text."""
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return list(set(re.findall(pattern, text)))


def _extract_phone_from_text(text: str) -> list[str]:
    """Extract phone numbers from text (US/international)."""
    pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    return list(set(re.findall(pattern, text)))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_leads(leads: list[ScrapedLead]) -> list[ScrapedLead]:
    """Remove duplicate leads based on dedup_key (email + company domain).

    When duplicates are found, merge data favouring the more complete record.
    """
    seen: dict[str, ScrapedLead] = {}
    for lead in leads:
        key = lead.dedup_key
        if key in seen:
            existing = seen[key]
            # Merge: fill empty fields from the incoming lead
            for attr in (
                "contact_name", "email", "company", "website",
                "phone", "linkedin_url", "industry", "company_size",
            ):
                if not getattr(existing, attr) and getattr(lead, attr):
                    setattr(existing, attr, getattr(lead, attr))
            # Keep higher score and data quality
            existing.score = max(existing.score, lead.score)
            existing.data_quality = max(existing.data_quality, lead.data_quality)
        else:
            seen[key] = lead
    return list(seen.values())


# ---------------------------------------------------------------------------
# Lead Scoring
# ---------------------------------------------------------------------------

def score_lead(lead: ScrapedLead, target_industry: str = "") -> float:
    """Compute a 0-100 lead score based on multiple quality signals."""
    score = 0.0

    # Completeness (max 40 pts)
    critical_fields = [lead.email, lead.contact_name, lead.company, lead.website]
    score += sum(8 for f in critical_fields if f)

    # Email validity (10 pts)
    if lead.email and ScrapedLead._validate_email(lead.email):
        score += 10

    # LinkedIn presence (10 pts)
    if lead.linkedin_url:
        score += 10

    # Phone number (10 pts)
    if lead.phone:
        score += 10

    # Industry match (15 pts)
    if target_industry and lead.industry:
        if target_industry.lower() in lead.industry.lower() or lead.industry.lower() in target_industry.lower():
            score += 15
        else:
            score += 5  # partial credit for any industry data

    # Company size provided (5 pts)
    if lead.company_size:
        score += 5

    # Data quality bonus (up to 10 pts)
    score += min(lead.data_quality / 10, 10)

    return min(round(score, 1), 100.0)


def score_leads(leads: list[ScrapedLead], target_industry: str = "") -> list[ScrapedLead]:
    """Score all leads using the scoring algorithm."""
    for lead in leads:
        lead.score = score_lead(lead, target_industry)
    return leads


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_to_csv(leads: list[ScrapedLead], path: str) -> str:
    """Export leads to CSV. Returns the file path."""
    import csv
    from pathlib import Path

    if not leads:
        return path

    fieldnames = [
        "contact_name", "email", "company", "website", "phone",
        "linkedin_url", "source", "industry", "company_size",
        "score", "dedup_key", "data_quality", "tags",
    ]
    rows = [lead.to_dict() for lead in leads]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Database Storage
# ---------------------------------------------------------------------------

async def store_leads(leads: list[ScrapedLead], db_url: str | None = None) -> list[Lead]:
    """Persist leads to the database, skipping duplicates by dedup_key."""
    from core.storage.database import init_db, async_session
    from sqlalchemy import select

    async with async_session() as session:
        # Ensure DB is initialised
        # (callers that already called init_db can skip, but safe to call)
        new_leads: list[Lead] = []
        for scraped in leads:
            # Check for existing lead with same dedup_key
            from sqlalchemy import text as sa_text  # noqa: F811
            existing = await session.execute(
                sa_text("SELECT id FROM leads WHERE dedup_key = :dk"),
                {"dk": scraped.dedup_key},
            )
            if existing.scalar_one_or_none():
                continue  # skip duplicate

            lead_model = scraped.to_lead_model()
            session.add(lead_model)
            new_leads.append(lead_model)

        await session.commit()
        for lead in new_leads:
            await session.refresh(lead)
        return new_leads


# ---------------------------------------------------------------------------
# CrewAI Agent
# ---------------------------------------------------------------------------


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
            "LinkedIn profiles, industry directories (Crunchbase, AngelList, Clutch.co), "
            "and business listings. Extract: company name, contact name/person's name, "
            "email (if public), phone (if public), website URL, and LinkedIn URL. "
            "Return structured lead data as a JSON array."
        ),
        expected_output=(
            "A JSON array of lead objects with fields: company, contact_name, email, "
            "phone, website, linkedin_url, source, industry, company_size"
        ),
        agent=agent,
    )


def create_enrichment_task(agent: Agent) -> Task:
    return Task(
        description=(
            "For each lead, enrich the data by: verifying the email format, "
            "finding additional contact info from the company website, "
            "checking company size and industry classification via web search, "
            "and adding a lead score from 1-100 based on: company size match, "
            "contact seniority, data completeness, and industry relevance. "
            "Deduplicate entries by email and company domain. "
            "Output the enriched, deduplicated lead list."
        ),
        expected_output=(
            "Enriched lead list with scores, deduplication applied, and data quality ratings. "
            "Each lead has: company, contact_name, email, phone, website, linkedin_url, "
            "score, data_quality, dedup_key, enriched, tags"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Orchestration Pipeline
# ---------------------------------------------------------------------------


class LeadScraperWorkflow:
    """End-to-end lead scraping + enrichment workflow via CrewAI."""

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


class LeadEnrichmentPipeline:
    """Programmatic (non-LLM) pipeline for enrichment, dedup, scoring."""

    def __init__(self, target_industry: str = ""):
        self.target_industry = target_industry

    def enrich(self, leads: list[ScrapedLead]) -> list[ScrapedLead]:
        """Full pipeline: enrich → deduplicate → score."""
        # Step 1: Deduplicate
        leads = deduplicate_leads(leads)

        # Step 2: Score
        leads = score_leads(leads, self.target_industry)

        # Step 3: Update enriched flag
        for lead in leads:
            lead.enriched = "Y"

        # Step 4: Sort by score descending
        leads.sort(key=lambda l: l.score, reverse=True)
        return leads


class LeadScraperAgent:
    """High-level API for scraping, enriching, exporting, and storing leads."""

    def __init__(self, industry: str, location: str = "", db_url: str | None = None):
        self.industry = industry
        self.location = location
        self.db_url = db_url
        self.workflow = LeadScraperWorkflow(industry, location)
        self.pipeline = LeadEnrichmentPipeline(industry)

    def scrape(self) -> list[ScrapedLead]:
        """Run the scraping workflow and return leads."""
        result = self.workflow.run()
        return self._parse_result(result)

    def enrich(self, leads: list[ScrapedLead]) -> list[ScrapedLead]:
        """Enrich, deduplicate, and score leads."""
        return self.pipeline.enrich(leads)

    def export_csv(self, leads: list[ScrapedLead], path: str) -> str:
        """Export leads to CSV file."""
        return export_to_csv(leads, path)

    async def store(self, leads: list[ScrapedLead]) -> list[Lead]:
        """Store leads in the database."""
        return await store_leads(leads, self.db_url)

    def run_pipeline(
        self,
        csv_path: str | None = None,
        store_db: bool = False,
    ) -> list[ScrapedLead]:
        """End-to-end: scrape → enrich → optional CSV export → optional DB store."""
        leads = self.scrape()
        leads = self.enrich(leads)
        if csv_path:
            export_to_csv(leads, csv_path)
        if store_db:
            asyncio.run(self.store(leads))
        return leads

    @staticmethod
    def _parse_result(result: Any) -> list[ScrapedLead]:
        """Parse the CrewAI string result into ScrapedLead objects."""
        import json

        if isinstance(result, list):
            return [ScrapedLead(**item) for item in result]

        raw = str(result).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [ScrapedLead(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return [ScrapedLead(**item) for item in data]
            except (json.JSONDecodeError, TypeError):
                pass

        return []
