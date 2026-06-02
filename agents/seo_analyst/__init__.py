"""SEO Analyst Agent.

Performs keyword research, competitor analysis, backlink monitoring,
and technical SEO audits for any domain.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Task, Crew, Process

from core.llm import get_llm
from core.tools import web_search, fetch_page, write_json, ensure_dir


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Keyword Research
# ---------------------------------------------------------------------------

def create_keyword_research_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Perform comprehensive keyword research for {domain}. "
            "Use web search to find: top ranking keywords, search volume estimates, "
            "related long-tail keywords, competitor keywords, and keyword difficulty. "
            "Return structured keyword data with priority rankings."
        ),
        expected_output=(
            "A JSON list of keywords with columns: "
            "keyword, estimated_volume, difficulty, priority"
        ),
        agent=agent,
    )


def format_keyword_results(raw: str) -> list[dict]:
    """Parse raw keyword research output into a structured list.

    Attempts JSON parse first; falls back to line-oriented parsing so the
    pipeline never crashes on unexpected LLM output.
    """
    if not raw:
        return []

    # Try direct JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [_normalise_keyword_row(r) for r in data]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: line-oriented heuristic parse
    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "---")):
            continue
        # strip leading bullet / number
        cleaned = line.lstrip("-*0123456789. ")
        if cleaned:
            rows.append({"keyword": cleaned, "estimated_volume": "", "difficulty": "", "priority": ""})
    return rows


def _normalise_keyword_row(row: dict | str) -> dict:
    """Normalise a single keyword row to a consistent dict shape."""
    if isinstance(row, str):
        return {"keyword": row, "estimated_volume": "", "difficulty": "", "priority": ""}
    return {
        "keyword": str(row.get("keyword", "")),
        "estimated_volume": str(row.get("estimated_volume", "")),
        "difficulty": str(row.get("difficulty", "")),
        "priority": str(row.get("priority", "")),
    }


def save_keyword_report(domain: str, keywords: list[dict], output_dir: str = "reports") -> str:
    """Write a keyword research report to disk and return the file path."""
    ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"keyword_report_{domain}_{stamp}.json")
    payload = {
        "domain": domain,
        "generated_at": datetime.now().isoformat() + "Z",
        "keyword_count": len(keywords),
        "keywords": keywords,
    }
    write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Competitor Analysis
# ---------------------------------------------------------------------------

def create_competitor_analysis_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze the top 3 SEO competitors for {domain}. "
            "Use web search to identify competitors, then compare: domain authority, "
            "top keywords, content strategy, backlink profile, and technical SEO. "
            "Return a comparison table with actionable gaps."
        ),
        expected_output=(
            "A competitor comparison matrix with key metrics and identified opportunities"
        ),
        agent=agent,
    )


def format_competitor_results(raw: str) -> list[dict]:
    """Parse raw competitor analysis output into structured data."""
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [_normalise_competitor_row(r) for r in data]
    except (json.JSONDecodeError, TypeError):
        pass

    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "---")):
            continue
        cleaned = line.lstrip("-*0123456789. ")
        if cleaned:
            rows.append({"competitor": cleaned, "metrics": "", "opportunities": ""})
    return rows


def _normalise_competitor_row(row: dict | str) -> dict:
    if isinstance(row, str):
        return {"competitor": row, "metrics": "", "opportunities": ""}
    return {
        "competitor": str(row.get("competitor", "")),
        "metrics": str(row.get("metrics", "")),
        "opportunities": str(row.get("opportunities", "")),
    }


def save_competitor_report(domain: str, competitors: list[dict], output_dir: str = "reports") -> str:
    ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"competitor_report_{domain}_{stamp}.json")
    payload = {
        "domain": domain,
        "generated_at": datetime.now().isoformat() + "Z",
        "competitor_count": len(competitors),
        "competitors": competitors,
    }
    write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Technical SEO Audit
# ---------------------------------------------------------------------------

def create_technical_audit_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Perform a technical SEO audit of {domain}. "
            f"Fetch the homepage at https://{domain} and analyze: page title, meta description, "
            "heading structure, URL structure, mobile-friendliness signals, "
            "page speed hints, schema markup, internal linking, and crawlability. "
            "Return a prioritized list of issues and fixes."
        ),
        expected_output=(
            "A technical SEO audit report with issue severity "
            "(critical/major/minor) and fix recommendations"
        ),
        agent=agent,
    )


def format_technical_audit_results(raw: str) -> list[dict]:
    """Parse raw audit output into structured issue list."""
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [_normalise_issue_row(r) for r in data]
        if isinstance(data, dict) and "issues" in data:
            return [_normalise_issue_row(r) for r in data["issues"]]
    except (json.JSONDecodeError, TypeError):
        pass

    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "---")):
            continue
        cleaned = line.lstrip("-*0123456789. ")
        if cleaned:
            rows.append({"issue": cleaned, "severity": "minor", "recommendation": ""})
    return rows


def _normalise_issue_row(row: dict | str) -> dict:
    if isinstance(row, str):
        return {"issue": row, "severity": "minor", "recommendation": ""}
    valid_severities = {"critical", "major", "minor"}
    sev = str(row.get("severity", "minor")).lower()
    if sev not in valid_severities:
        sev = "minor"
    return {
        "issue": str(row.get("issue", "")),
        "severity": sev,
        "recommendation": str(row.get("recommendation", "")),
    }


def save_technical_audit_report(domain: str, issues: list[dict], output_dir: str = "reports") -> str:
    ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"technical_audit_{domain}_{stamp}.json")
    critical = [i for i in issues if i["severity"] == "critical"]
    major = [i for i in issues if i["severity"] == "major"]
    minor = [i for i in issues if i["severity"] == "minor"]
    payload = {
        "domain": domain,
        "generated_at": datetime.now().isoformat() + "Z",
        "summary": {
            "total_issues": len(issues),
            "critical": len(critical),
            "major": len(major),
            "minor": len(minor),
        },
        "issues": issues,
    }
    write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Backlink Monitoring
# ---------------------------------------------------------------------------

def create_backlink_task(domain: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Research the backlink profile for {domain}. "
            "Use web search to find: estimated backlink count, referring domains, "
            "quality signals (educational/governmental links), anchor text distribution, "
            "and toxic link indicators. Summarise with actionable recommendations."
        ),
        expected_output=(
            "Backlink profile summary count, top referring domains, "
            "quality assessment, and actionable recommendations"
        ),
        agent=agent,
    )


def format_backlink_results(raw: str) -> dict:
    if not raw:
        return {}

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "estimated_backlink_count": str(data.get("estimated_backlink_count", "")),
                "referring_domains": str(data.get("referring_domains", "")),
                "quality_assessment": str(data.get("quality_assessment", "")),
                "recommendations": str(data.get("recommendations", "")),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {"summary": raw}


# ---------------------------------------------------------------------------
# Multi-mode workflow & report runner
# ---------------------------------------------------------------------------

class SEOReports:
    """Run SEO analysis sub-modules and persist results to disk."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir

    def run_keyword_report(self, domain: str, raw_output: str) -> str:
        keywords = format_keyword_results(raw_output)
        return save_keyword_report(domain, keywords, output_dir=self.output_dir)

    def run_competitor_report(self, domain: str, raw_output: str) -> str:
        competitors = format_competitor_results(raw_output)
        return save_competitor_report(domain, competitors, output_dir=self.output_dir)

    def run_technical_report(self, domain: str, raw_output: str) -> str:
        issues = format_technical_audit_results(raw_output)
        return save_technical_audit_report(domain, issues, output_dir=self.output_dir)


class SEOAnalystWorkflow:
    """End-to-end SEO analysis workflow."""

    def __init__(self, domain: str):
        self.domain = domain
        self.agent = create_seo_analyst()

    # -- individual mode runners -------------------------------------------

    def run_keyword_research(self) -> str:
        return self._run_single(create_keyword_research_task(self.domain, self.agent))

    def run_competitor_analysis(self) -> str:
        return self._run_single(create_competitor_analysis_task(self.domain, self.agent))

    def run_technical_audit(self) -> str:
        return self._run_single(create_technical_audit_task(self.domain, self.agent))

    def run_backlink_analysis(self) -> str:
        return self._run_single(create_backlink_task(self.domain, self.agent))

    # -- combined runner ---------------------------------------------------

    def run(self, mode: str = "full") -> str:
        """Run the SEO analysis.

        Args:
            mode: ``"keyword"``, ``"competitor"``, ``"technical"``,
                  ``"backlink"``, or ``"full"``.
        """
        if mode not in ("keyword", "competitor", "technical", "backlink", "full"):
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: keyword, competitor, technical, backlink, full"
            )

        tasks = []
        if mode in ("keyword", "full"):
            tasks.append(create_keyword_research_task(self.domain, self.agent))
        if mode in ("competitor", "full"):
            tasks.append(create_competitor_analysis_task(self.domain, self.agent))
        if mode in ("technical", "full"):
            tasks.append(create_technical_audit_task(self.domain, self.agent))
        if mode in ("backlink", "full"):
            tasks.append(create_backlink_task(self.domain, self.agent))

        crew = Crew(
            agents=[self.agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _run_single(task: Task) -> str:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()
