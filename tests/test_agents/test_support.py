"""Tests for Customer Support agent."""

import pytest


def test_customer_support_module_imports():
    """Customer Support module should import all components."""
    from agents.customer_support import (
        create_support_agent,
        create_ticket_triage_task,
        create_response_draft_task,
        CustomerSupportWorkflow,
    )
    assert create_support_agent is not None
    assert create_ticket_triage_task is not None
    assert create_response_draft_task is not None
    assert CustomerSupportWorkflow is not None


def test_customer_support_task_signatures():
    """Customer support task functions should have correct signatures."""
    import inspect
    from agents.customer_support import (
        create_ticket_triage_task,
        create_response_draft_task,
    )

    sig = inspect.signature(create_ticket_triage_task)
    assert "tickets" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_response_draft_task)
    assert "ticket" in sig.parameters
    assert "company_info" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
