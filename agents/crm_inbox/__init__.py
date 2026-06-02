"""CRM + Inbox Management Agent.

Manages contacts, triages emails, schedules follow-ups,
and tracks pipeline stages with HubSpot integration.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search
from core.integrations import (
    search_contacts,
    create_contact,
    update_contact,
    get_contact,
    create_deal,
    update_deal_stage,
    list_deals,
    get_deal,
    list_pipelines,
)


# ── Agent factory ────────────────────────────────────────────────────────────

def create_crm_agent() -> Agent:
    """Create the CRM + Inbox Management agent.

    Returns:
        A CrewAI Agent configured for CRM operations, email triage,
        contact sync, and pipeline tracking.
    """
    return Agent(
        role="CRM + Inbox Manager",
        goal=(
            "Manage contacts, triage communications, track sales pipeline, "
            "and optimize CRM efficiency through HubSpot integration"
        ),
        backstory=(
            "You are a CRM expert who manages contacts, emails, and sales pipelines "
            "for growing businesses. You excel at contact segmentation, email triage, "
            "follow-up scheduling, and pipeline analytics. You integrate with HubSpot "
            "to sync contacts, create deals, and track pipeline stages. You can search "
            "and enrich contact records, categorize inbound emails by urgency, and "
            "generate pipeline health reports."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Task factories ───────────────────────────────────────────────────────────

def create_contact_sync_task(contacts: list[dict], agent: Agent) -> Task:
    """Build a task that syncs and deduplicates a batch of contacts.

    Args:
        contacts: Raw contact dicts with at minimum an ``email`` key.
        agent: The CRM agent that will execute the task.

    Returns:
        A CrewAI Task.
    """
    return Task(
        description=(
            f"Sync and deduplicate {len(contacts)} contacts. "
            "For each contact: validate email format, enrich with company data "
            "if missing, tag by industry and stage (lead/qualified/customer/churned), "
            "and prepare for CRM import. Flag duplicates and conflicts. "
            "Use search_contacts to check for existing records before creating new ones."
        ),
        expected_output=(
            "Cleaned and tagged contact list ready for CRM import with conflict "
            "resolution notes and a summary of actions taken (created/updated/skipped)"
        ),
        agent=agent,
    )


def create_email_triage_task(email_batch: str, agent: Agent) -> Task:
    """Build a task that triages a batch of emails.

    Args:
        email_batch: Raw text containing one or more emails.
        agent: The CRM agent that will execute the task.

    Returns:
        A CrewAI Task.
    """
    return Task(
        description=(
            f"Triage and categorize the following batch of emails:\n\n{email_batch}\n\n"
            "For each email classify: urgency (critical/high/medium/low), "
            "category (sales/support/general/spam), suggested action "
            "(reply/forward/delegate/archive), and draft a short response "
            "for important emails. Also identify any emails that should trigger "
            "a new deal or contact update in the CRM."
        ),
        expected_output=(
            "Email triage table with classification, priority, suggested actions, "
            "draft responses, and CRM action items (new contacts/deals to create)"
        ),
        agent=agent,
    )


def create_pipeline_report_task(agent: Agent) -> Task:
    """Build a task that generates a pipeline health report.

    Args:
        agent: The CRM agent that will execute the task.

    Returns:
        A CrewAI Task.
    """
    return Task(
        description=(
            "Generate a comprehensive pipeline health report. "
            "Use list_pipelines to get all pipeline stages, then use list_deals "
            "to get current deals. Calculate: total pipeline value, deals per stage, "
            "average deal value, and identify stale deals (no activity in 30+ days). "
            "Recommend actions to move deals forward."
        ),
        expected_output=(
            "Pipeline health report with stage breakdown, value metrics, "
            "stale deal alerts, and recommended next actions"
        ),
        agent=agent,
    )


def create_follow_up_task(contact_ids: list[str], agent: Agent) -> Task:
    """Build a task that schedules follow-ups for a list of contacts.

    Args:
        contact_ids: HubSpot contact IDs to follow up with.
        agent: The CRM agent that will execute the task.

    Returns:
        A CrewAI Task.
    """
    return Task(
        description=(
            f"Schedule follow-up actions for {len(contact_ids)} contacts. "
            "For each contact ID, use get_contact to retrieve their details and "
            "current lifecycle stage. Determine the appropriate follow-up action "
            "based on stage: lead → send intro email, qualified → schedule demo, "
            "opportunity → send proposal, customer → check-in call, "
            "churned → re-engagement campaign. "
            "Generate a follow-up schedule with dates and action items."
        ),
        expected_output=(
            "Follow-up schedule table with contact name, current stage, "
            "recommended action, due date, and draft message for each contact"
        ),
        agent=agent,
    )


# ── Workflow class ───────────────────────────────────────────────────────────

class CRMInboxWorkflow:
    """CRM + Inbox management workflow.

    Provides high-level operations for contact sync, email triage,
    pipeline reporting, and follow-up scheduling.
    """

    def __init__(self):
        self.agent = create_crm_agent()

    def sync_contacts(self, contacts: list[dict]) -> str:
        """Sync and deduplicate a batch of contacts.

        Args:
            contacts: Raw contact dicts.

        Returns:
            Crew kickoff result string.
        """
        task = create_contact_sync_task(contacts, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def triage_emails(self, email_batch: str) -> str:
        """Triage a batch of emails.

        Args:
            email_batch: Raw email text.

        Returns:
            Crew kickoff result string.
        """
        task = create_email_triage_task(email_batch, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def pipeline_report(self) -> str:
        """Generate a pipeline health report.

        Returns:
            Crew kickoff result string.
        """
        task = create_pipeline_report_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def schedule_follow_ups(self, contact_ids: list[str]) -> str:
        """Schedule follow-ups for given contacts.

        Args:
            contact_ids: HubSpot contact IDs.

        Returns:
            Crew kickoff result string.
        """
        task = create_follow_up_task(contact_ids, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
