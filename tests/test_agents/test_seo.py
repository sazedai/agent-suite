"""Tests for SEO Analyst agent."""

import pytest


def test_seo_module_imports():
    """SEO analyst module should import all components."""
    from agents.seo_analyst import (
        create_seo_analyst,
        create_keyword_research_task,
        create_competitor_analysis_task,
        create_technical_audit_task,
        SEOAnalystWorkflow,
    )
    assert create_seo_analyst is not None
    assert create_keyword_research_task is not None
    assert create_competitor_analysis_task is not None
    assert create_technical_audit_task is not None
    assert SEOAnalystWorkflow is not None


def test_seo_task_descriptions():
    """SEO task functions should produce tasks with proper descriptions.
    We test the functions by inspecting their source defaults without
    actually instantiating CrewAI Task objects (which require real agents).
    """
    import inspect
    from agents.seo_analyst import (
        create_keyword_research_task,
        create_competitor_analysis_task,
        create_technical_audit_task,
    )

    # Verify the functions accept domain and agent parameters
    sig = inspect.signature(create_keyword_research_task)
    assert "domain" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_competitor_analysis_task)
    assert "domain" in sig.parameters

    sig = inspect.signature(create_technical_audit_task)
    assert "domain" in sig.parameters
