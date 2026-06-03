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
            "You know when to escalate and how to de-escalate tense situations. "
            "You track support metrics and continuously improve response quality."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Ticket Triage ──────────────────────────────────────────────────────────────

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


# ── Response Drafting ──────────────────────────────────────────────────────────

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


# ── Escalation Handling ────────────────────────────────────────────────────────

def create_escalation_handling_task(
    ticket: str,
    escalation_policy: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Handle the escalation for this ticket:\n\n{ticket}\n\n"
            f"Escalation policy:\n{escalation_policy}\n\n"
            "Produce: "
            "(1) root cause summary, "
            "(2) severity reassessment, "
            "(3) recommended escalation path (tier-2/tier-3/management/legal), "
            "(4) suggested compensation or goodwill gesture if applicable, "
            "(5) internal handoff notes for the receiving team, "
            "(6) customer-facing update message, and "
            "(7) prevention recommendations to avoid recurrence."
        ),
        expected_output=(
            "Escalation packet with root cause, reassessment, escalation path, "
            "handoff notes, customer update, and prevention recommendations"
        ),
        agent=agent,
    )


def create_escalation_classification_task(
    ticket: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Determine if this ticket requires escalation and to which level:\n\n"
            f"{ticket}\n\n"
            "Evaluate against these escalation criteria:\n"
            "- Tier 2: technical bug > 24h, data loss, workaround exhausted\n"
            "- Tier 3: security incident, legal threat, system outage\n"
            "- Management: VIP customer, >3 unresolved contacts, PR risk\n"
            "- Legal: litigation threat, regulatory complaint, privacy breach\n\n"
            "Return: escalation level (none/tier2/tier3/management/legal), "
            "justulence confidence (low/medium/high), "
            "urgency rationale, recommended assignee role, "
            "and maximum acceptable response time."
        ),
        expected_output="Escalation level, confidence, rationale, assignee role, and SLA deadline",
        agent=agent,
    )


# ── FAQ Management ─────────────────────────────────────────────────────────────

def create_faq_management_task(
    faq_entries: str,
    action: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Perform FAQ management action: {action}\n\n"
            f"Current FAQ entries:\n{faq_entries}\n\n"
            "Supported actions:\n"
            "- 'organize': group into categories, deduplicate, "
            "identify gaps, suggest new entries\n"
            "- 'update': revise answers for accuracy, clarity, "
            "and tone; flag outdated entries\n"
            "- 'expand': for each entry add related questions, "
            "variations, and troubleshooting tips\n"
            "- 'audit': score each entry quality (1-10), "
            "flag broken links, check for consistency\n\n"
            "Always return structured output with category, "
            "question, answer, tags, and last-reviewed timestamp."
        ),
        expected_output=(
            "FAQ management report with organized/updated/expanded/audited entries, "
            "quality scores, and gap analysis"
        ),
        agent=agent,
    )


def create_faq_answer_task(
    question: str,
    faq_database: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Answer this customer question using the FAQ database:\n\n"
            f"Question: {question}\n\n"
            f"FAQ Database:\n{faq_database}\n\n"
            "Find the best matching FAQ entry. If found, return a "
            "polished, empathetic answer. If no match is found, "
            "return 'NO_MATCH' and draft a new FAQ entry that "
            "should be added to cover this question."
        ),
        expected_output="Best FAQ match with polished answer, or NO_MATCH with draft new entry",
        agent=agent,
    )


# ── Support Analytics ──────────────────────────────────────────────────────────

def create_support_analytics_task(
    ticket_log: str,
    period: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Analyze support ticket data for period: {period}\n\n"
            f"Ticket log:\n{ticket_log}\n\n"
            "Produce analytics report covering:\n"
            "1. Total tickets, resolved, pending, escalated\n"
            "2. Average first response time & resolution time\n"
            "3. Top categories and trending issues\n"
            "4. Agent performance (tickets handled, resolution rate)\n"
            "5. Customer satisfaction drivers\n"
            "6. Bottlenecks and recommended process improvements\n"
            "7. Volume forecast for next period"
        ),
        expected_output=(
            "Support analytics dashboard with metrics, trends, "
            "bottlenecks, and actionable recommendations"
        ),
        agent=agent,
    )


def create_satisfaction_prediction_task(
    ticket_thread: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Predict customer satisfaction from this support thread:\n\n"
            f"{ticket_thread}\n\n"
            "Analyze: sentiment trajectory, resolution completeness, "
            "response quality, tone matching, follow-through. "
            "Return: predicted CSAT score (1-5), NPS likelihood "
            "(detractor/passive/promoter), churn risk (low/medium/high/critical), "
            "and specific improvement suggestions."
        ),
        expected_output=(
            "CSAT prediction, NPS likelihood, churn risk, "
            "and improvement recommendations"
        ),
        agent=agent,
    )


# ── Workflow Class ─────────────────────────────────────────────────────────────

class CustomerSupportWorkflow:
    """End-to-end customer support workflow."""

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

    def handle_escalation(self, ticket: str, escalation_policy: str = "") -> str:
        escal_task = create_escalation_handling_task(
            ticket, escalation_policy, self.agent
        )
        crew = Crew(agents=[self.agent], tasks=[escal_task], verbose=True)
        return crew.kickoff()

    def classify_escalation(self, ticket: str) -> str:
        task = create_escalation_classification_task(ticket, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def manage_faq(self, faq_entries: str, action: str = "audit") -> str:
        task = create_faq_management_task(faq_entries, action, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def answer_faq(self, question: str, faq_database: str) -> str:
        task = create_faq_answer_task(question, faq_database, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_support(self, ticket_log: str, period: str = "last_30_days") -> str:
        task = create_support_analytics_task(ticket_log, period, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def predict_satisfaction(self, ticket_thread: str) -> str:
        task = create_satisfaction_prediction_task(ticket_thread, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def run_pipeline(
        self,
        tickets: str,
        company_info: str,
        escalation_policy: str,
        faq_entries: str,
        ticket_log: str,
        period: str = "last_30_days",
    ) -> str:
        """Run the full support pipeline: triage → response → escalation check → FAQ → analytics."""
        triage_task = create_ticket_triage_task(tickets, self.agent)
        response_task = create_response_draft_task(tickets, company_info, self.agent)
        escalation_task = create_escalation_handling_task(
            tickets, escalation_policy, self.agent
        )
        faq_task = create_faq_management_task(faq_entries, "audit", self.agent)
        analytics_task = create_support_analytics_task(ticket_log, period, self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[triage_task, response_task, escalation_task, faq_task, analytics_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
