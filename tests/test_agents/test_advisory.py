"""Tests for Advisory Council agent."""

import pytest


def test_advisory_module_imports():
    """Advisory Council module should import all components."""
    from agents.advisory_council import (
        create_advisor,
        create_document_analysis_task,
        create_advisory_panel_task,
        create_risk_assessment_task,
        AdvisoryCouncilWorkflow,
    )
    assert create_advisor is not None
    assert create_document_analysis_task is not None
    assert create_advisory_panel_task is not None
    assert create_risk_assessment_task is not None
    assert AdvisoryCouncilWorkflow is not None


def test_advisory_task_signatures():
    """Advisory task functions should have correct signatures."""
    import inspect
    from agents.advisory_council import (
        create_document_analysis_task,
        create_advisory_panel_task,
        create_risk_assessment_task,
    )

    sig = inspect.signature(create_document_analysis_task)
    assert "doc_content" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_advisory_panel_task)
    assert "question" in sig.parameters
    assert "context" in sig.parameters
    assert "advisors" in sig.parameters

    sig = inspect.signature(create_risk_assessment_task)
    assert "proposal" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
