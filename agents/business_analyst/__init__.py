"""Business Analyst + KPI Reporting Agent.

Creates dashboards, tracks KPIs, detects anomalies,
and generates business performance reports.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, write_json, write_csv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KPI_HEALTH_GREEN = "green"
KPI_HEALTH_YELLOW = "yellow"
KPI_HEALTH_RED = "red"

ANOMALY_SEVERITY_CRITICAL = "critical"
ANOMALY_SEVERITY_HIGH = "high"
ANOMALY_SEVERITY_MEDIUM = "medium"
ANOMORITY_SEVERITY_LOW = "low"

REPORT_PERIOD_WEEKLY = "weekly"
REPORT_PERIOD_MONTHLY = "monthly"
REPORT_PERIOD_QUARTERLY = "quarterly"
REPORT_PERIOD_ANNUAL = "annual"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_business_analyst(
    model: str | None = None,
    temperature: float | None = None,
) -> Agent:
    """Create a Business Analyst agent.

    Args:
        model: Optional LLM model override (e.g. ``"gpt-4o"``).
        temperature: Optional temperature override.

    Returns:
        A configured CrewAI Agent.
    """
    return Agent(
        role="Senior Business Analyst",
        goal="Transform raw data into actionable business intelligence and KPI insights",
        backstory=(
            "You are a data-driven business analyst who excels at KPI tracking, "
            "dashboard creation, anomaly detection, and executive reporting. "
            "You understand SaaS metrics, e-commerce KPIs, financial analysis, "
            "and operational metrics. You translate numbers into narratives. "
            "You are precise with numbers, clear in your recommendations, "
            "and always tie insights back to business objectives."
        ),
        llm=get_llm(model=model, temperature=temperature),
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# KPI data helpers
# ---------------------------------------------------------------------------

def compute_kpi_health(current: float, target: float, *, higher_is_better: bool = True) -> str:
    """Assign a health colour to a single KPI.

    Args:
        current: Current KPI value.
        target: Target / threshold value.
        higher_is_better: Whether exceeding the target is good.

    Returns:
        ``KPI_HEALTH_GREEN``, ``KPI_HEALTH_YELLOW``, or ``KPI_HEALTH_RED``.
    """
    if target == 0:
        return KPI_HEALTH_GREEN if current >= 0 else KPI_HEALTH_RED
    ratio = current / target if higher_is_better else target / current
    if ratio >= 0.95:
        return KPI_HEALTH_GREEN
    if ratio >= 0.75:
        return KPI_HEALTH_YELLOW
    return KPI_HEALTH_RED


def compute_period_change(current: float, previous: float) -> float:
    """Compute period-over-period percent change.

    Returns:
        Percentage change (e.g. 12.5 for +12.5%).  Returns 0.0 when
        ``previous`` is zero to avoid division-by-zero.
    """
    if previous == 0:
        return 0.0
    return round(((current - previous) / abs(previous)) * 100, 2)


def detect_anomalies_statistical(
    data: list[float | int],
    *,
    z_threshold: float = 2.0,
) -> list[dict]:
    """Detect anomalies in a numeric time-series using z-score.

    Args:
        data: Ordered list of metric values (oldest to newest).
        z_threshold: Absolute z-score above which a point is anomalous.

    Returns:
        List of dicts with keys ``index``, ``value``, ``z_score``,
        ``severity`` for each anomaly found.
    """
    n = len(data)
    if n < 3:
        return []

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = variance ** 0.5
    if std == 0:
        return []

    anomalies: list[dict] = []
    for i, value in enumerate(data):
        z = round((value - mean) / std, 3)
        abs_z = abs(z)
        if abs_z < z_threshold:
            continue
        if abs_z >= 3.0:
            severity = ANOMALY_SEVERITY_CRITICAL
        elif abs_z >= 2.5:
            severity = ANOMALY_SEVERITY_HIGH
        else:
            severity = ANOMALY_SEVERITY_MEDIUM
        anomalies.append({"index": i, "value": value, "z_score": z, "severity": severity})
    return anomalies


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------

def create_kpi_dashboard_task(
    kpi_data: dict,
    agent: Agent,
    period: str = "current",
    previous_data: dict | None = None,
) -> Task:
    """Create a KPI dashboard analysis task.

    Args:
        kpi_data: Mapping of KPI name → current value.
        agent: The analyst agent.
        period: Label for the reporting period.
        previous_data: Optional mapping of KPI name → prior period value
            for period-over-period comparison.
    """
    prev_section = ""
    if previous_data:
        prev_section = (
            f"\n\nPrevious period data for comparison:\n{previous_data}\n"
            "Include period-over-period percent change for each metric."
        )
    return Task(
        description=(
            f"Analyze the following KPI data for the {period} period and create a "
            "comprehensive dashboard summary:\n\n"
            f"{kpi_data}\n"
            "For each metric calculate: health status (red/yellow/green), "
            "trend direction, performance vs target, and actionable insight. "
            "Then provide: overall business health score, top 3 positives, "
            "top 3 concerns, and prioritized recommendations."
            f"{prev_section}"
        ),
        expected_output=(
            "KPI dashboard with health scores, period changes, trend arrows, "
            "overall health rating, top positives, top concerns, and "
            "prioritized recommendations"
        ),
        agent=agent,
    )


def create_anomaly_detection_task(
    metrics_history: list[dict],
    agent: Agent,
    sensitivity: str = "medium",
) -> Task:
    """Create an anomaly detection task.

    Args:
        metrics_history: List of dicts, each mapping metric name → value,
            ordered chronologically (oldest first).
        agent: The analyst agent.
        sensitivity: One of ``"low"``, ``"medium"``, ``"high"`` — controls
            how aggressively anomalies are flagged.
    """
    sensitivity_note = {
        "low": "Only flag statistically significant outliers (very high confidence).",
        "medium": "Flag moderate and high-severity anomalies.",
        "high": "Flag all suspicious patterns including mild deviations.",
    }.get(sensitivity, "Flag moderate and high-severity anomalies.")

    return Task(
        description=(
            f"Analyze this metrics history for anomalies "
            f"(sensitivity: {sensitivity}):\n\n"
            f"{metrics_history}\n\n"
            f"{sensitivity_note}\n"
            "Detect: statistical outliers, sudden changes, trend breaks, "
            "seasonal anomalies, and correlated metric divergences. "
            "For each anomaly: assess severity (critical/high/medium/low), "
            "suggest root causes, estimate business impact, and "
            "recommend investigation steps."
        ),
        expected_output=(
            "Anomaly report with severity ratings, root cause hypotheses, "
            "business impact estimates, and ranked investigation plan"
        ),
        agent=agent,
    )


def create_executive_report_task(
    period: str,
    agent: Agent,
    kpi_data: dict | None = None,
    highlights: list[str] | None = None,
) -> Task:
    """Create an executive report generation task.

    Args:
        period: Reporting period label (e.g. ``"Q4 2025"``).
        agent: The analyst agent.
        kpi_data: Optional KPI snapshot to anchor the report.
        highlights: Optional list of notable events or wins.
    """
    sections = [
        "executive summary",
        "financial highlights",
        "operational metrics",
        "growth metrics",
        "risk indicators",
        "competitive landscape (search for recent industry/competitor news)",
        "forward-looking recommendations and strategic priorities",
    ]
    section_text = ", ".join(sections)

    kpi_section = ""
    if kpi_data:
        kpi_section = f"\n\nKPI data for the period:\n{kpi_data}\n"

    highlights_section = ""
    if highlights:
        highlights_section = f"\n\nKey highlights to include:\n{highlights}\n"

    return Task(
        description=(
            f"Generate an executive business report for {period}. "
            f"Include the following sections: {section_text}."
            f"{kpi_section}{highlights_section}"
            "Format for leadership review: clear headings, key numbers in "
            "bold, trend arrows where applicable, and a concise narrative. "
            "Keep the full report under 1000 words."
        ),
        expected_output=(
            "Executive business report formatted for leadership review, "
            "with all standard sections, key metrics, and strategic recommendations"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class BusinessAnalystWorkflow:
    """Business analysis and KPI reporting workflow.

    Provides three high-level operations:

    * :meth:`dashboard`       – KPI dashboard creation
    * :meth:`detect_anomalies` – anomaly detection on metric history
    * :meth:`exec_report`      – executive report generation
    """

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.agent = create_business_analyst(model=model, temperature=temperature)

    def dashboard(
        self,
        kpi_data: dict,
        period: str = "current",
        previous_data: dict | None = None,
    ) -> str:
        """Run a KPI dashboard analysis.

        Args:
            kpi_data: Mapping of KPI name → current value.
            period: Reporting period label.
            previous_data: Optional prior period data for comparison.

        Returns:
            The CrewAI task output string.
        """
        task = create_kpi_dashboard_task(
            kpi_data, self.agent, period=period, previous_data=previous_data,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def detect_anomalies(
        self,
        metrics_history: list[dict],
        sensitivity: str = "medium",
    ) -> str:
        """Run anomaly detection on a metric history.

        Args:
            metrics_history: Chronological list of metric snapshots.
            sensitivity: Anomaly detection sensitivity.

        Returns:
            The CrewAI task output string.
        """
        task = create_anomaly_detection_task(
            metrics_history, self.agent, sensitivity=sensitivity,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def exec_report(
        self,
        period: str,
        kpi_data: dict | None = None,
        highlights: list[str] | None = None,
    ) -> str:
        """Generate an executive business report.

        Args:
            period: Reporting period label.
            kpi_data: Optional KPI snapshot.
            highlights: Optional list of notable events.

        Returns:
            The CrewAI task output string.
        """
        task = create_executive_report_task(
            period, self.agent, kpi_data=kpi_data, highlights=highlights,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
