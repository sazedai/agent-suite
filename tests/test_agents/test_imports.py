"""Tests for agents."""

import pytest


def test_seo_analyst_creation():
    """SEO analyst agent should be created without errors."""
    from agents.seo_analyst import create_seo_analyst, SEOAnalystWorkflow
    agent = create_seo_analyst()
    assert agent.role == "Senior SEO Analyst"
    workflow = SEOAnalystWorkflow("example.com")
    assert workflow.domain == "example.com"


def test_research_agent_creation():
    """Research agent should be created without errors."""
    from agents.research_agent import create_researcher, ResearchWorkflow
    agent = create_researcher()
    assert agent.role == "AI & Technology Research Analyst"
    workflow = ResearchWorkflow("AI")
    assert workflow.topic == "AI"


def test_content_producer_creation():
    """Content producer should be created without errors."""
    from agents.content_production import create_content_producer, ContentProductionWorkflow
    agent = create_content_producer()
    assert agent.role == "Senior Content Producer"
    workflow = ContentProductionWorkflow()
    assert workflow.agent is not None


def test_investment_analyst_creation():
    """Investment analyst should be created without errors."""
    from agents.investment_analyst import create_investment_analyst, InvestmentAnalystWorkflow
    agent = create_investment_analyst()
    assert agent.role == "Senior Investment Analyst"
    workflow = InvestmentAnalystWorkflow()
    assert workflow.agent is not None


def test_advisory_council_creation():
    """Advisory council should have 5 advisors."""
    from agents.advisory_council import AdvisoryCouncilWorkflow
    workflow = AdvisoryCouncilWorkflow()
    assert len(workflow.advisors) == 5


def test_all_agents_importable():
    """All 15 agents should be importable."""
    from agents import seo_analyst, lead_scraper, crm_inbox, sales_call_analyst
    from agents import research_agent, content_production, content_repurpose
    from agents import second_brain, executive_assistant, business_analyst
    from agents import operations, customer_support, shopify_assistant
    from agents import investment_analyst, advisory_council

    assert seo_analyst is not None
    assert lead_scraper is not None
    assert crm_inbox is not None
    assert research_agent is not None
    assert content_production is not None
    assert advisory_council is not None
