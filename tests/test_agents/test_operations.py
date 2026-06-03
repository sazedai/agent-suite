"""Tests for Operations Employee Agent."""

import inspect

import pytest


def test_operations_agent_import():
    """Operations agent module should import without errors."""
    from agents.operations import create_operations_agent, OperationsWorkflow
    assert create_operations_agent is not None
    assert OperationsWorkflow is not None


def test_operations_all_task_functions_importable():
    """All task factory functions should be importable."""
    from agents.operations import (
        create_task_management_task,
        create_task_delegation_task,
        create_workflow_optimization_task,
        create_sop_tracker_task,
        create_sop_creation_task,
        create_vendor_evaluation_task,
        create_vendor_onboarding_task,
        create_automation_audit_task,
        create_process_mapping_task,
        create_operations_health_check_task,
    )
    assert create_task_management_task is not None
    assert create_task_delegation_task is not None
    assert create_workflow_optimization_task is not None
    assert create_sop_tracker_task is not None
    assert create_sop_creation_task is not None
    assert create_vendor_evaluation_task is not None
    assert create_vendor_onboarding_task is not None
    assert create_automation_audit_task is not None
    assert create_process_mapping_task is not None
    assert create_operations_health_check_task is not None


def test_operations_workflow_class_methods():
    """OperationsWorkflow should expose all expected workflow methods."""
    from agents.operations import OperationsWorkflow
    expected = [
        "manage_tasks",
        "delegate_tasks",
        "optimize_workflow",
        "audit_sops",
        "create_sop",
        "evaluate_vendors",
        "onboard_vendor",
        "audit_automation",
        "map_process",
        "health_check",
    ]
    for method_name in expected:
        assert hasattr(OperationsWorkflow, method_name), (
            f"OperationsWorkflow missing method: {method_name}"
        )
        assert callable(getattr(OperationsWorkflow, method_name)), (
            f"OperationsWorkflow.{method_name} is not callable"
        )


# ── Task Management Task Signatures ──────────────────────────────

def test_task_management_task_signature():
    """Task management task should accept tasks and agent params."""
    from agents.operations import create_task_management_task
    sig = inspect.signature(create_task_management_task)
    assert "tasks" in sig.parameters
    assert "agent" in sig.parameters


def test_task_delegation_task_signature():
    """Task delegation task should accept tasks, team, and agent params."""
    from agents.operations import create_task_delegation_task
    sig = inspect.signature(create_task_delegation_task)
    assert "tasks" in sig.parameters
    assert "team" in sig.parameters
    assert "agent" in sig.parameters


def test_workflow_optimization_task_signature():
    """Workflow optimization task should accept workflow and agent params."""
    from agents.operations import create_workflow_optimization_task
    sig = inspect.signature(create_workflow_optimization_task)
    assert "current_workflow" in sig.parameters
    assert "agent" in sig.parameters


# ── SOP Tracking Task Signatures ─────────────────────────────────

def test_sop_tracker_task_signature():
    """SOP tracker task should accept sop_list and agent params."""
    from agents.operations import create_sop_tracker_task
    sig = inspect.signature(create_sop_tracker_task)
    assert "sop_list" in sig.parameters
    assert "agent" in sig.parameters


def test_sop_creation_task_signature():
    """SOP creation task should accept process_name, details, and agent params."""
    from agents.operations import create_sop_creation_task
    sig = inspect.signature(create_sop_creation_task)
    assert "process_name" in sig.parameters
    assert "details" in sig.parameters
    assert "agent" in sig.parameters


# ── Vendor Management Task Signatures ────────────────────────────

def test_vendor_evaluation_task_signature():
    """Vendor evaluation task should accept vendors and agent params."""
    from agents.operations import create_vendor_evaluation_task
    sig = inspect.signature(create_vendor_evaluation_task)
    assert "vendors" in sig.parameters
    assert "agent" in sig.parameters


def test_vendor_onboarding_task_signature():
    """Vendor onboarding task should accept vendor_info and agent params."""
    from agents.operations import create_vendor_onboarding_task
    sig = inspect.signature(create_vendor_onboarding_task)
    assert "vendor_info" in sig.parameters
    assert "agent" in sig.parameters


# ── Process Automation Task Signatures ───────────────────────────

def test_automation_audit_task_signature():
    """Automation audit task should accept processes and agent params."""
    from agents.operations import create_automation_audit_task
    sig = inspect.signature(create_automation_audit_task)
    assert "processes" in sig.parameters
    assert "agent" in sig.parameters


def test_process_mapping_task_signature():
    """Process mapping task should accept process_name, steps, and agent params."""
    from agents.operations import create_process_mapping_task
    sig = inspect.signature(create_process_mapping_task)
    assert "process_name" in sig.parameters
    assert "steps" in sig.parameters
    assert "agent" in sig.parameters


def test_operations_health_check_task_signature():
    """Health check task should accept metrics and agent params."""
    from agents.operations import create_operations_health_check_task
    sig = inspect.signature(create_operations_health_check_task)
    assert "metrics" in sig.parameters
    assert "agent" in sig.parameters


# ── Agent Configuration ──────────────────────────────────────────

def test_operations_agent_configuration():
    """Operations agent should have correct role and goal configuration."""
    from agents.operations import create_operations_agent

    with pytest.MonkeyPatch.context() as mp:
        # CrewAI >= 1.0 accepts llm as a string identifier
        mp.setattr("agents.operations.get_llm", lambda: "gpt-4o")
        agent = create_operations_agent()
        assert agent.role == "Operations Manager"
        assert "task management" in agent.goal or "operations" in agent.goal.lower()
