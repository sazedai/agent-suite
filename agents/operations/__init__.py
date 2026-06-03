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
        goal="Keep business operations running smoothly through task management, SOPs, vendor coordination, and process automation",
        backstory=(
            "You are a detail-oriented operations manager who keeps everything "
            "running like clockwork. You manage task queues, standard operating "
            "procedures, vendor relationships, and process documentation. "
            "You identify bottlenecks before they become problems and always "
            "look for ways to automate repetitive work."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Task Management ──────────────────────────────────────────────

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


def create_task_delegation_task(tasks: str, team: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Delegate these tasks to the team:\n\n{tasks}\n\n"
            f"Team members and their roles:\n{team}\n\n"
            "For each task: identify the best assignee based on skills and "
            "workload, write a clear task brief with acceptance criteria, "
            "suggest a deadline, flag tasks that need collaboration, "
            "and identify any tasks better handled by automation."
        ),
        expected_output="Delegation plan with assignee mapping, task briefs, and timeline",
        agent=agent,
    )


def create_workflow_optimization_task(current_workflow: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze and optimize this workflow:\n\n{current_workflow}\n\n"
            "Identify: bottlenecks, redundant steps, manual processes that "
            "can be automated, missing error handling, and unclear handoffs. "
            "Propose: a streamlined workflow diagram, automation candidates "
            "with estimated time savings, recommended tools/integrations, "
            "and a phased implementation plan."
        ),
        expected_output="Workflow optimization report with bottleneck analysis, automation candidates, and implementation roadmap",
        agent=agent,
    )


# ── SOP Tracking ─────────────────────────────────────────────────

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


def create_sop_creation_task(process_name: str, details: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Create a new Standard Operating Procedure for: {process_name}\n\n"
            f"Process details:\n{details}\n\n"
            "Produce a complete SOP with: purpose scope, prerequisites, "
            "step-by-step instructions, decision points and branching logic, "
            "error handling procedures, quality checkpoints, "
            "roles and responsibilities, and a revision history template."
        ),
        expected_output="Complete SOP document with all standard sections ready for team adoption",
        agent=agent,
    )


# ── Vendor Management ────────────────────────────────────────────

def create_vendor_evaluation_task(vendors: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Evaluate and compare these vendors:\n\n{vendors}\n\n"
            "For each vendor score: reliability (1-10), cost-competitiveness "
            "(1-10), quality (1-10), responsiveness (1-10), and "
            "strategic fit (1-10). Produce: a comparison matrix, "
            "risk assessment for each vendor, recommended vendor ranking, "
            "negotiation leverage points, and contract red flags to watch."
        ),
        expected_output="Vendor evaluation matrix with scores, rankings, risk assessments, and negotiation points",
        agent=agent,
    )


def create_vendor_onboarding_task(vendor_info: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Create a vendor onboarding plan for:\n\n{vendor_info}\n\n"
            "Produce: pre-onboarding checklist, NDA and compliance requirements, "
            "integration steps (technical and operational), SLA definition, "
            "communication protocol, escalation contacts, "
            "trial period criteria, and success metrics."
        ),
        expected_output="Vendor onboarding plan with checklist, compliance steps, SLAs, and success criteria",
        agent=agent,
    )


# ── Process Automation ───────────────────────────────────────────

def create_automation_audit_task(processes: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Audit these processes for automation potential:\n\n{processes}\n\n"
            "For each process assess: automation feasibility (high/medium/low), "
            "estimated time savings per week, implementation complexity, "
            "recommended automation approach (script/integration/AI), "
            "dependencies and prerequisites, and ROI estimate. "
            "Prioritize by impact vs effort matrix."
        ),
        expected_output="Automation audit with feasibility scores, time savings estimates, and prioritized implementation plan",
        agent=agent,
    )


def create_process_mapping_task(process_name: str, steps: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Create a detailed process map for: {process_name}\n\n"
            f"Known steps:\n{steps}\n\n"
            "Produce: a complete flow diagram description (start to finish), "
            "decision points and branching paths, input/output at each step, "
            "time estimates per step, responsible roles, "
            "exception handling procedures, and KPIs to measure process health."
        ),
        expected_output="Detailed process map with flow, decision points, time estimates, roles, and KPIs",
        agent=agent,
    )


# ── Health Check & Reporting ─────────────────────────────────────

def create_operations_health_check_task(metrics: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Run an operations health check on these metrics:\n\n{metrics}\n\n"
            "Assess: operational efficiency score (1-100), SOP compliance rate, "
            "task completion rate, vendor performance summary, "
            "automation coverage percentage, top 3 operational risks, "
            "top 3 improvement opportunities, and a weekly action plan."
        ),
        expected_output="Operations health report with efficiency score, compliance rate, risks, and action plan",
        agent=agent,
    )


# ── Workflow Class ───────────────────────────────────────────────

class OperationsWorkflow:
    """Operations management workflow."""

    def __init__(self):
        self.agent = create_operations_agent()

    # Task management
    def manage_tasks(self, tasks: str) -> str:
        task = create_task_management_task(tasks, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def delegate_tasks(self, tasks: str, team: str) -> str:
        task = create_task_delegation_task(tasks, team, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def optimize_workflow(self, current_workflow: str) -> str:
        task = create_workflow_optimization_task(current_workflow, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # SOP tracking
    def audit_sops(self, sop_list: str) -> str:
        task = create_sop_tracker_task(sop_list, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def create_sop(self, process_name: str, details: str) -> str:
        task = create_sop_creation_task(process_name, details, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Vendor management
    def evaluate_vendors(self, vendors: str) -> str:
        task = create_vendor_evaluation_task(vendors, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def onboard_vendor(self, vendor_info: str) -> str:
        task = create_vendor_onboarding_task(vendor_info, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Process automation
    def audit_automation(self, processes: str) -> str:
        task = create_automation_audit_task(processes, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def map_process(self, process_name: str, steps: str) -> str:
        task = create_process_mapping_task(process_name, steps, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Health check
    def health_check(self, metrics: str) -> str:
        task = create_operations_health_check_task(metrics, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
