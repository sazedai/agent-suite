"""Advisory Council with NotebookLM.

Multi-perspective advisory panel that simulates expert opinions,
analyzes documents, and provides structured decision support.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page, write_json


def create_advisor(name: str, expertise: str, perspective: str) -> Agent:
    return Agent(
        role=f"Advisor: {name}",
        goal=f"Provide expert {expertise} perspective on decisions and documents",
        backstory=(
            f"You are {name}, a {expertise} expert. "
            f"Your perspective is: {perspective}. "
            "You advise based on your area of expertise, challenge assumptions, "
            "and provide structured recommendations. You are direct but constructive."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_document_analysis_task(doc_content: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze the following document from your area of expertise:\n\n{doc_content}\n\n"
            "Identify: key claims, supporting evidence, logical gaps, "
            "risks and blind spots, comparison with industry best practices, "
            "and your assessment of the quality of the analysis."
        ),
        expected_output="Expert analysis of the document with claims assessment, gap analysis, and quality rating",
        agent=agent,
    )


def create_advisory_panel_task(
    question: str,
    context: str,
    advisors: list[Agent],
) -> Task:
    return Task(
        description=(
            f"Advisory Council Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Synthesize all advisory perspectives into a unified recommendation. "
            "Structure: executive summary, points of agreement, "
            "points of disagreement, risk-adjusted recommendation, "
            "confidence level (high/medium/low), and suggested next steps."
        ),
        expected_output="Unified advisory recommendation synthesizing all expert perspectives",
        agent=advisors[0] if advisors else None,
    )


def create_risk_assessment_task(proposal: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Assess the risks of this proposal:\n\n{proposal}\n\n"
            "Identify: financial risks, operational risks, market risks, "
            "regulatory risks, reputation risks, and black swan scenarios. "
            "Rate each risk by probability and impact. "
            "Provide mitigation strategies for the top 5 risks."
        ),
        expected_output="Risk assessment matrix with probability-impact ratings and mitigation strategies",
        agent=agent,
    )


class AdvisoryCouncilWorkflow:
    """Multi-agent advisory council with document analysis (NotebookLM-style)."""

    def __init__(self):
        self.advisors = [
            create_advisor("Finance Expert", "financial analysis", "Conservative, risk-aware financial perspective focused on ROI, cash flow, and capital efficiency"),
            create_advisor("Strategy Expert", "business strategy", "Growth-oriented strategic perspective focused on market positioning, competitive advantage, and long-term value"),
            create_advisor("Operations Expert", "operations management", "Execution-focused perspective focused on feasibility, resource constraints, and operational risk"),
            create_advisor("Technology Expert", "technology and AI", "Innovation-focused perspective on technical feasibility, AI implications, and digital transformation"),
            create_advisor("Legal Expert", "legal and compliance", "Risk and compliance perspective focused on regulatory requirements, contracts, and liability"),
        ]

    def analyze_document(self, doc_content: str, focus: str = "general") -> str:
        """Analyze a document with all advisors, each from their perspective."""
        advisor = self.advisors[0]
        for a in self.advisors:
            if focus.lower() in a.role.lower():
                advisor = a
                break
        task = create_document_analysis_task(doc_content, advisor)
        crew = Crew(agents=[advisor], tasks=[task], verbose=True)
        return crew.kickoff()

    def council_vote(self, question: str, context: str = "") -> str:
        """Get the full advisory council's perspective on a question."""
        all_advisors = self.advisors
        tasks = []
        for advisor in all_advisors:
            tasks.append(Task(
                description=(
                    f"Advisory Question: {question}\n\n"
                    f"Context: {context}\n\n"
                    "Provide your expert opinion with: "
                    "your assessment, reasoning, conditions for your support, "
                    "and any dissenting views."
                ),
                expected_output=f"Expert opinion from {advisor.role} with assessment and conditions",
                agent=advisor,
            ))

        crew = Crew(agents=all_advisors, tasks=tasks, process=Process.parallel, verbose=True)
        return crew.kickoff()

    def assess_risks(self, proposal: str) -> str:
        """Run a structured risk assessment on a proposal."""
        task = create_risk_assessment_task(proposal, self.advisors[0])
        crew = Crew(agents=[self.advisors[0]], tasks=[task], verbose=True)
        return crew.kickoff()
