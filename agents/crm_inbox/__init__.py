"""CRM + Inbox Management Agent.

Manages contacts, triages emails, schedules follow-ups,
and tracks pipeline stages.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search
from core.integrations import search_contacts, create_contact


def create_crm_agent() -> Agent:
    return Agent(
        role="CRM Manager",
        goal="Manage contacts, triage communications, and optimize pipeline efficiency",
        backstory=(
            "You are a CRM expert who manages contacts, emails, and sales pipelines "
            "for growing businesses. You excel at contact segmentation, email triage, "
            "follow-up scheduling, and pipeline analytics. You integrate with HubSpot "
            "and other CRM platforms."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_contact_sync_task(contacts: list[dict], agent: Agent) -> Task:
    return Task(
        description=(
            f"Sync and deduplicate {len(contacts)} contacts. "
            "For each contact: validate email format, enrich with company data "
            "if missing, tag by industry and stage (lead/qualified/customer/churned), "
            "and prepare for CRM import. Flag duplicates and conflicts."
        ),
        expected_output="Cleaned and tagged contact list ready for CRM import with conflict resolution notes",
        agent=agent,
    )


def create_email_triage_task(email_batch: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Triage and categorize the following batch of emails:\n\n{email_batch}\n\n"
            "For each email classify: urgency (critical/high/medium/low), "
            "category (sales/support/general/spam), suggested action "
            "(reply/forward/delegate/archive), and draft a short response "
            "for important emails."
        ),
        expected_output="Email triage table with classification, priority, suggested actions, and draft responses",
        agent=agent,
    )


class CRMInboxWorkflow:
    """CRM + Inbox management workflow."""

    def __init__(self):
        self.agent = create_crm_agent()

    def sync_contacts(self, contacts: list[dict]) -> str:
        task = create_contact_sync_task(contacts, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def triage_emails(self, email_batch: str) -> str:
        task = create_email_triage_task(email_batch, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
