"""Operations Employee Agent.

Manages tasks, tracks SOPs, handles vendors,
and automates operational processes.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


def create_operations_agent() -> Agent:
    return Agent(
        role="Operations Manager",
        goal="Keep business operations running smoothly through task management, SOPs, and process optimization",
        backstory=(
            "You are a detail-oriented operations manager who keeps everything "
            "running like clockwork. You manage task queues, standard operating "
            "procedures, vendor relationships, and process documentation. "
            "You identify bottlenecks before they become problems."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_task_management_task(tasks: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Process and organize the following tasks:\n\n{tasks}\n\n"
            "For each task: assign priority (P0-P3), estimate effort, "
            "identify dependencies, suggest assignee role, "
            "flag blockers, and create a suggested execution order. "
            "Also identify tasks that can be batched or automated."
        ),
        expected_output="Prioritized task board with effort estimates, dependencies, and execution order",
        agent=agent,
    )


def create_sop_tracker_task(sop_list: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Review and update the following SOPs:\n\n{sop_list}\n\n"
            "For each SOP: check completeness, identify gaps, "
            "suggest improvements, flag outdated steps, "
            "and create a revision checklist. Also check "
            "cross-SOP consistency and identify duplicate processes."
        ),
        expected_output="SOP audit report with gap analysis, improvement suggestions, and revision priorities",
        agent=agent,
    )


class OperationsWorkflow:
    """Operations management workflow."""

    def __init__(self):
        self.agent = create_operations_agent()

    def manage_tasks(self, tasks: str) -> str:
        task = create_task_management_task(tasks, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def audit_sops(self, sop_list: str) -> str:
        task = create_sop_tracker_task(sop_list, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
