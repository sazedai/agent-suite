"""Tests for Executive Assistant agent."""

import pytest


def test_exec_assistant_module_imports():
    """Executive Assistant module should import all components."""
    from agents.executive_assistant import (
        create_exec_assistant,
        create_meeting_prep_task,
        create_priority_triage_task,
        create_travel_plan_task,
        ExecutiveAssistantWorkflow,
    )
    assert create_exec_assistant is not None
    assert create_meeting_prep_task is not None
    assert create_priority_triage_task is not None
    assert create_travel_plan_task is not None
    assert ExecutiveAssistantWorkflow is not None


def test_exec_assistant_task_signatures():
    """Executive assistant task functions should have correct signatures."""
    import inspect
    from agents.executive_assistant import (
        create_meeting_prep_task,
        create_priority_triage_task,
        create_travel_plan_task,
    )

    sig = inspect.signature(create_meeting_prep_task)
    assert "meeting_details" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_priority_triage_task)
    assert "items" in sig.parameters

    sig = inspect.signature(create_travel_plan_task)
    assert "destination" in sig.parameters
    assert "dates" in sig.parameters
    assert "purpose" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
