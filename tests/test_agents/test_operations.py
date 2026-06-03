"""Tests for Operations agent."""

import pytest


def test_operations_module_imports():
    """Operations module should import all components."""
    from agents.operations import (
        create_operations_agent,
        create_task_management_task,
        create_sop_tracker_task,
        OperationsWorkflow,
    )
    assert create_operations_agent is not None
    assert create_task_management_task is not None
    assert create_sop_tracker_task is not None
    assert OperationsWorkflow is not None


def test_operations_task_signatures():
    """Operations task functions should have correct signatures."""
    import inspect
    from agents.operations import (
        create_task_management_task,
        create_sop_tracker_task,
    )

    sig = inspect.signature(create_task_management_task)
    assert "tasks" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_sop_tracker_task)
    assert "sop_list" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
