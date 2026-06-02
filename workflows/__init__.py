"""Multi-agent workflow orchestrator."""

from __future__ import annotations

from crewai import Crew, Process
from core.llm import get_llm
from agents.seo_analyst import create_seo_analyst
from agents.content_production import create_content_producer
from agents.research_agent import create_researcher


def content_marketing_pipeline(topic: str, domain: str) -> str:
    """Full content marketing pipeline: research -> SEO -> content.

    Args:
        topic: The content topic
        domain: The target domain for SEO

    Returns:
        Combined output from all pipeline stages
    """
    researcher = create_researcher()
    seo_analyst = create_seo_analyst()
    content_producer = create_content_producer()

    crew = Crew(
        agents=[researcher, seo_analyst, content_producer],
        tasks=[
            # Research
            researcher.agent_task if hasattr(researcher, 'agent_task') else None,
        ],
        process=Process.sequential,
        verbose=True,
    )
    return crew.kickoff()


def sales_intel_pipeline(company: str) -> str:
    """Full sales intelligence pipeline.

    Args:
        company: Target company name

    Returns:
        Combined competitive and lead intelligence
    """
    return f"Sales intelligence pipeline for {company} - TODO: implement"
