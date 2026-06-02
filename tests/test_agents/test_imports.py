"""Tests for agents."""

import pytest


def test_seo_analyst_import():
    """SEO analyst module should import without errors."""
    from agents.seo_analyst import create_seo_analyst, SEOAnalystWorkflow
    assert create_seo_analyst is not None
    assert SEOAnalystWorkflow is not None


def test_research_agent_import():
    """Research agent module should import without errors."""
    from agents.research_agent import create_researcher, ResearchWorkflow
    assert create_researcher is not None
    assert ResearchWorkflow is not None


def test_content_producer_import():
    """Content producer module should import without errors."""
    from agents.content_production import create_content_producer, ContentProductionWorkflow
    assert create_content_producer is not None
    assert ContentProductionWorkflow is not None


def test_investment_analyst_import():
    """Investment analyst module should import without errors."""
    from agents.investment_analyst import create_investment_analyst, InvestmentAnalystWorkflow
    assert create_investment_analyst is not None
    assert InvestmentAnalystWorkflow is not None


def test_advisory_council_import():
    """Advisory council module should import without errors."""
    from agents.advisory_council import AdvisoryCouncilWorkflow, create_advisor
    assert AdvisoryCouncilWorkflow is not None
    assert create_advisor is not None


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
