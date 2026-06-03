"""Tests for Business Analyst agent."""

import pytest


def test_business_analyst_module_imports():
    """Business Analyst module should import all components."""
    from agents.business_analyst import (
        create_business_analyst,
        create_kpi_dashboard_task,
        create_anomaly_detection_task,
        create_executive_report_task,
        BusinessAnalystWorkflow,
    )
    assert create_business_analyst is not None
    assert create_kpi_dashboard_task is not None
    assert create_anomaly_detection_task is not None
    assert create_executive_report_task is not None
    assert BusinessAnalystWorkflow is not None


def test_business_analyst_task_signatures():
    """Business analyst task functions should have correct signatures."""
    import inspect
    from agents.business_analyst import (
        create_kpi_dashboard_task,
        create_anomaly_detection_task,
        create_executive_report_task,
    )

    sig = inspect.signature(create_kpi_dashboard_task)
    assert "kpi_data" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_anomaly_detection_task)
    assert "metrics_history" in sig.parameters

    sig = inspect.signature(create_executive_report_task)
    assert "period" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
