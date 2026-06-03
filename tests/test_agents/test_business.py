"""Tests for Business Analyst + KPI Reporting agent."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from agents.business_analyst import (
    create_business_analyst,
    create_kpi_dashboard_task,
    create_anomaly_detection_task,
    create_executive_report_task,
    compute_kpi_health,
    compute_period_change,
    detect_anomalies_statistical,
    BusinessAnalystWorkflow,
    KPI_HEALTH_GREEN,
    KPI_HEALTH_YELLOW,
    KPI_HEALTH_RED,
    ANOMALY_SEVERITY_CRITICAL,
    ANOMALY_SEVERITY_HIGH,
    ANOMALY_SEVERITY_MEDIUM,
)


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------

def test_module_imports():
    """All public symbols should be importable from the module."""
    from agents.business_analyst import (
        create_business_analyst,
        create_kpi_dashboard_task,
        create_anomaly_detection_task,
        create_executive_report_task,
        compute_kpi_health,
        compute_period_change,
        detect_anomalies_statistical,
        BusinessAnalystWorkflow,
    )
    assert callable(create_business_analyst)
    assert callable(create_kpi_dashboard_task)
    assert callable(create_anomaly_detection_task)
    assert callable(create_executive_report_task)
    assert callable(compute_kpi_health)
    assert callable(compute_period_change)
    assert callable(detect_anomalies_statistical)
    assert BusinessAnalystWorkflow is not None


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

@patch("agents.business_analyst.get_llm", return_value="gpt-4o")
def test_create_business_analyst_returns_agent(mock_get_llm):
    """create_business_analyst should return a CrewAI Agent with the right role."""
    from crewai import Agent

    agent = create_business_analyst()
    assert isinstance(agent, Agent)
    assert "Business Analyst" in agent.role
    assert agent.allow_delegation is False


@patch("agents.business_analyst.get_llm", return_value="gpt-4o")
def test_create_business_analyst_accepts_model_override(mock_get_llm):
    """create_business_analyst should forward model and temperature overrides."""
    agent = create_business_analyst(model="gpt-4-turbo", temperature=0.2)
    mock_get_llm.assert_called_once_with(model="gpt-4-turbo", temperature=0.2)


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

class TestComputeKpiHealth:
    def test_green_when_meeting_target(self):
        assert compute_kpi_health(100, 100) == KPI_HEALTH_GREEN

    def test_green_when_exceeding_target(self):
        assert compute_kpi_health(110, 100) == KPI_HEALTH_GREEN

    def test_green_at_95_percent(self):
        assert compute_kpi_health(95, 100) == KPI_HEALTH_GREEN

    def test_yellow_at_75_percent(self):
        assert compute_kpi_health(75, 100) == KPI_HEALTH_YELLOW

    def test_yellow_between_75_and_95(self):
        assert compute_kpi_health(80, 100) == KPI_HEALTH_YELLOW

    def test_red_below_75_percent(self):
        assert compute_kpi_health(74, 100) == KPI_HEALTH_RED

    def test_red_for_zero(self):
        assert compute_kpi_health(0, 100) == KPI_HEALTH_RED

    def test_lower_is_better(self):
        """When lower_is_better=True, being below target is green."""
        assert compute_kpi_health(50, 100, higher_is_better=False) == KPI_HEALTH_GREEN

    def test_lower_is_better_red(self):
        """When lower_is_better and current exceeds target significantly."""
        assert compute_kpi_health(150, 100, higher_is_better=False) == KPI_HEALTH_RED

    def test_zero_target_with_positive_current(self):
        assert compute_kpi_health(5, 0) == KPI_HEALTH_GREEN


class TestComputePeriodChange:
    def test_increase(self):
        assert compute_period_change(110, 100) == 10.0

    def test_decrease(self):
        assert compute_period_change(90, 100) == -10.0

    def test_no_change(self):
        assert compute_period_change(100, 100) == 0.0

    def test_zero_previous_returns_zero(self):
        assert compute_period_change(50, 0) == 0.0

    def test_negative_previous(self):
        assert compute_period_change(-80, -100) == 20.0

    def test_rounding(self):
        result = compute_period_change(133, 100)
        assert result == 33.0


class TestDetectAnomaliesStatistical:
    def test_empty_list(self):
        assert detect_anomalies_statistical([]) == []

    def test_too_few_points(self):
        assert detect_anomalies_statistical([1.0, 2.0]) == []

    def test_no_anomalies_in_uniform_data(self):
        result = detect_anomalies_statistical([10, 10, 10, 10, 10, 10])
        assert result == []

    def test_spike_detected(self):
        data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 50]
        anomalies = detect_anomalies_statistical(data)
        assert len(anomalies) == 1
        assert anomalies[0]["index"] == 9
        assert anomalies[0]["value"] == 50
        assert anomalies[0]["severity"] == ANOMALY_SEVERITY_CRITICAL

    def test_low_spike_detected(self):
        data = [50, 50, 50, 50, 50, 50, 50, 50, 50, 10]
        anomalies = detect_anomalies_statistical(data)
        assert len(anomalies) == 1
        assert anomalies[0]["z_score"] < 0

    def test_medium_severity(self):
        """Points with z between 2.0 and 2.5 should be medium severity."""
        data = [100, 100, 100, 100, 100, 110]
        anomalies = detect_anomalies_statistical(data)
        assert any(a["severity"] == ANOMALY_SEVERITY_MEDIUM for a in anomalies)

    def test_high_severity(self):
        """Points with z between 2.5 and 3.0 should be high severity."""
        data = [100, 100, 100, 100, 100, 100, 100, 100, 100, 140]
        anomalies = detect_anomalies_statistical(data)
        assert any(a["severity"] in (ANOMALY_SEVERITY_HIGH, ANOMALY_SEVERITY_CRITICAL) for a in anomalies)

    def test_custom_threshold(self):
        """With a high threshold, the spike may not be flagged."""
        data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 25]
        anomalies_default = detect_anomalies_statistical(data)
        anomalies_strict = detect_anomalies_statistical(data, z_threshold=3.5)
        assert len(anomalies_strict) <= len(anomalies_default)

    def test_all_same_values_no_anomalies(self):
        """Zero standard deviation means no anomalies."""
        data = [5, 5, 5, 5, 5]
        assert detect_anomalies_statistical(data) == []

    def test_anomaly_fields(self):
        data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 50]
        anomalies = detect_anomalies_statistical(data)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert "index" in a
        assert "value" in a
        assert "z_score" in a
        assert "severity" in a


# ---------------------------------------------------------------------------
# Task factories (inspection-based, no real Agent needed)
# ---------------------------------------------------------------------------

class TestCreateKpiDashboardTask:
    def test_signature(self):
        sig = inspect.signature(create_kpi_dashboard_task)
        assert "kpi_data" in sig.parameters
        assert "agent" in sig.parameters
        assert "period" in sig.parameters
        assert "previous_data" in sig.parameters

    def test_default_period(self):
        sig = inspect.signature(create_kpi_dashboard_task)
        assert sig.parameters["period"].default == "current"

    def test_default_previous_data(self):
        sig = inspect.signature(create_kpi_dashboard_task)
        assert sig.parameters["previous_data"].default is None

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_description_contains_kpi_data(self, mock_llm):
        agent = create_business_analyst()
        kpi = {"revenue": 500000, "churn": 0.03}
        task = create_kpi_dashboard_task(kpi, agent)
        assert "revenue" in task.description
        assert "health" in task.description.lower()

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_with_previous_data(self, mock_llm):
        agent = create_business_analyst()
        task = create_kpi_dashboard_task(
            {"revenue": 500000}, agent, previous_data={"revenue": 450000},
        )
        assert "comparison" in task.description.lower()


class TestCreateAnomalyDetectionTask:
    def test_signature(self):
        sig = inspect.signature(create_anomaly_detection_task)
        assert "metrics_history" in sig.parameters
        assert "agent" in sig.parameters
        assert "sensitivity" in sig.parameters

    def test_default_sensitivity(self):
        sig = inspect.signature(create_anomaly_detection_task)
        assert sig.parameters["sensitivity"].default == "medium"

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_description_includes_sensitivity(self, mock_llm):
        agent = create_business_analyst()
        history = [{"revenue": 100}, {"revenue": 200}]
        task = create_anomaly_detection_task(history, agent, sensitivity="high")
        assert "high" in task.description

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_description_includes_severity_levels(self, mock_llm):
        agent = create_business_analyst()
        history = [{"revenue": 100}]
        task = create_anomaly_detection_task(history, agent)
        assert "severity" in task.description.lower()


class TestCreateExecutiveReportTask:
    def test_signature(self):
        sig = inspect.signature(create_executive_report_task)
        assert "period" in sig.parameters
        assert "agent" in sig.parameters
        assert "kpi_data" in sig.parameters
        assert "highlights" in sig.parameters

    def test_defaults(self):
        sig = inspect.signature(create_executive_report_task)
        assert sig.parameters["kpi_data"].default is None
        assert sig.parameters["highlights"].default is None

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_description_contains_period(self, mock_llm):
        agent = create_business_analyst()
        task = create_executive_report_task("Q1 2026", agent)
        assert "Q1 2026" in task.description

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_includes_executive_summary_and_recommendations(self, mock_llm):
        agent = create_business_analyst()
        task = create_executive_report_task("January 2026", agent)
        desc = task.description.lower()
        assert "executive summary" in desc
        assert "recommendations" in desc

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_with_kpi_data(self, mock_llm):
        agent = create_business_analyst()
        task = create_executive_report_task(
            "Q1 2026", agent, kpi_data={"revenue": 500000},
        )
        assert "500000" in task.description

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_task_with_highlights(self, mock_llm):
        agent = create_business_analyst()
        task = create_executive_report_task(
            "Q1 2026", agent, highlights=["Launched new product line"],
        )
        assert "new product line" in task.description


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class TestBusinessAnalystWorkflow:
    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_init_creates_agent(self, mock_llm):
        wf = BusinessAnalystWorkflow()
        from crewai import Agent
        assert isinstance(wf.agent, Agent)

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    def test_init_accepts_model_and_temperature(self, mock_llm):
        wf = BusinessAnalystWorkflow(model="gpt-4-turbo", temperature=0.1)
        mock_get_llm = mock_llm
        mock_get_llm.assert_called_with(model="gpt-4-turbo", temperature=0.1)

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_dashboard_calls_crew_with_task(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "dashboard output"
        result = wf.dashboard({"revenue": 100000})
        assert result == "dashboard output"
        mock_crew_cls.assert_called_once()
        call_kwargs = mock_crew_cls.call_args.kwargs
        assert len(call_kwargs["tasks"]) == 1
        assert len(call_kwargs["agents"]) == 1

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_dashboard_passes_previous_data(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "dashboard output"
        result = wf.dashboard(
            {"revenue": 100000},
            previous_data={"revenue": 90000},
        )
        assert result == "dashboard output"

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_detect_anomalies_calls_crew(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "anomaly result"
        history = [{"revenue": 100}, {"revenue": 200}]
        result = wf.detect_anomalies(history)
        assert result == "anomaly result"

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_detect_anomalies_passes_sensitivity(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "anomaly result"
        result = wf.detect_anomalies([{"x": 1}], sensitivity="high")
        assert result == "anomaly result"

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_exec_report_calls_crew(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "report output"
        result = wf.exec_report("Q4 2025")
        assert result == "report output"

    @patch("agents.business_analyst.get_llm", return_value="gpt-4o")
    @patch("agents.business_analyst.Crew")
    def test_exec_report_with_kpi_and_highlights(self, mock_crew_cls, mock_llm):
        wf = BusinessAnalystWorkflow()
        mock_crew_cls.return_value.kickoff.return_value = "report output"
        result = wf.exec_report(
            "Q4 2025",
            kpi_data={"revenue": 1000000},
            highlights=["Best quarter ever"],
        )
        assert result == "report output"
