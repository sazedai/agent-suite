"""Customer Support Workflows.

Ticket triage, response drafting, escalation handling,
FAQ management, and support analytics.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


def create_support_agent() -> Agent:
    return Agent(
        role="Customer Support Specialist",
        goal="Resolve customer issues quickly and professionally while maintaining high satisfaction",
        backstory=(
            "You are a customer support expert known for empathy, speed, "
            "and first-contact resolution. You handle ticket triage, "
            "draft responses, manage escalations, and maintain FAQ databases. "
            "You know when to escalate and how to de-escalate tense situations."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_ticket_triage_task(tickets: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Triage and prioritize these support tickets:\n\n{tickets}\n\n"
            "For each ticket classify: severity (critical/high/medium/low), "
            "category (billing/technical/feature_request/bug/general), "
            "sentiment (angry/frustrated/neutral/positive), "
            "requires escalation (yes/no), suggested first response, "
            "and estimated resolution complexity (S/M/L)."
        ),
        expected_output="Triage report with classification, priority, escalation flags, and suggested first responses",
        agent=agent,
    )


def create_response_draft_task(ticket: str, company_info: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Draft a customer support response for this ticket:\n\n{ticket}\n\n"
            f"Company context: {company_info}\n\n"
            "Create: empathetic opening, clear explanation, "
            "step-by-step resolution (if applicable), "
            "alternative solutions, escalation path, and "
            "professional closing. Match the customer's communication style."
        ),
        expected_output="Draft customer support response with empathy, resolution steps, and follow-up plan",
        agent=agent,
    )


class CustomerSupportWorkflow:
    """Customer support workflow."""

    def __init__(self):
        self.agent = create_support_agent()

    def triage_tickets(self, tickets: str) -> str:
        task = create_ticket_triage_task(tickets, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def draft_response(self, ticket: str, company_info: str = "") -> str:
        task = create_response_draft_task(ticket, company_info, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
