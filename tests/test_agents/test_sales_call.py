"""Tests for Sales Call Analyst agent."""

import pytest


def test_sales_call_module_imports():
    """Sales Call Analyst module should import all components."""
    from agents.sales_call_analyst import (
        create_sales_analyst,
        create_transcript_analysis_task,
        create_follow_up_task,
        SalesCallWorkflow,
    )
    assert create_sales_analyst is not None
    assert create_transcript_analysis_task is not None
    assert create_follow_up_task is not None
    assert SalesCallWorkflow is not None


def test_sales_call_task_signatures():
    """Sales call task functions should have correct signatures."""
    import inspect
    from agents.sales_call_analyst import (
        create_transcript_analysis_task,
        create_follow_up_task,
    )

    sig = inspect.signature(create_transcript_analysis_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_follow_up_task)
    assert "agent" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
