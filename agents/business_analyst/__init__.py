"""Business Analyst + KPI Reporting Agent.

Creates dashboards, tracks KPIs, detects anomalies,
and generates business performance reports.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, write_json, write_csv


def create_business_analyst() -> Agent:
    return Agent(
        role="Senior Business Analyst",
        goal="Transform raw data into actionable business intelligence and KPI insights",
        backstory=(
            "You are a data-driven business analyst who excels at KPI tracking, "
            "dashboard creation, anomaly detection, and executive reporting. "
            "You understand SaaS metrics, e-commerce KPIs, financial analysis, "
            "and operational metrics. You translate numbers into narratives."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_kpi_dashboard_task(kpi_data: dict, agent: Agent) -> Task:
    return Task(
        description=(
            "Analyze the following KPI data and create a dashboard summary:\n\n"
            f"{kpi_data}\n\n"
            "Calculate: period-over-period changes, trend identification, "
            "metric health scoring (red/yellow/green), top 3 positives, "
            "top 3 concerns, and recommended actions for each underperforming metric."
        ),
        expected_output="KPI dashboard with health scores, trends, and prioritized recommendations",
        agent=agent,
    )


def create_anomaly_detection_task(metrics_history: list[dict], agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze this metrics history for anomalies:\n\n{metrics_history}\n\n"
            "Detect: statistical outliers, sudden changes, trend breaks, "
            "seasonal anomalies, and correlated metric divergences. "
            "For each anomaly: assess severity, suggest root causes, "
            "and recommend investigation steps."
        ),
        expected_output="Anomaly report with severity ratings, root cause hypotheses, and investigation plans",
        agent=agent,
    )


def create_executive_report_task(period: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Generate an executive business report for {period}. "
            "Structure: executive summary, financial highlights, "
            "operational metrics, growth metrics, risk indicators, "
            "competitive landscape (search for competitor news), "
            "and forward-looking recommendations."
        ),
        expected_output="Executive business report with all standard sections, formatted for leadership review",
        agent=agent,
    )


class BusinessAnalystWorkflow:
    """Business analysis and KPI reporting workflow."""

    def __init__(self):
        self.agent = create_business_analyst()

    def dashboard(self, kpi_data: dict) -> str:
        task = create_kpi_dashboard_task(kpi_data, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def detect_anomalies(self, metrics_history: list[dict]) -> str:
        task = create_anomaly_detection_task(metrics_history, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def exec_report(self, period: str) -> str:
        task = create_executive_report_task(period, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
