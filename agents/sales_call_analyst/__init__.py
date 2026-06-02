"""Sales Call Analyst Agent.

Analyzes sales call recordings/transcripts for sentiment,
key moments, objections, competitor mentions, and action items.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


def create_sales_analyst() -> Agent:
    return Agent(
        role="Sales Call Analyst",
        goal="Extract maximum insight from sales conversations to improve close rates",
        backstory=(
            "You are a VP of Sales turned AI analyst. You've listened to thousands "
            "of sales calls and know exactly what wins deals. You analyze sentiment, "
            "identify objections, track competitor mentions, extract buying signals, "
            "and generate actionable coaching notes for sales reps."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_transcript_analysis_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze the following sales call transcript:\n\n{transcript}\n\n"
            "Extract: overall sentiment (buyer and seller), key conversation moments, "
            "objections raised, competitor mentions, buying signals, "
            "price sensitivity indicators, next steps agreed, and action items. "
            "Rate the call quality 1-10 with reasoning."
        ),
        expected_output="Call analysis report with sentiment timeline, objections, buying signals, and scored action items",
        agent=agent,
    )


def create_follow_up_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Based on the call analysis, generate: a personalized follow-up email "
            "that addresses objections raised, key value propositions to reinforce, "
            "a risk of inaction statement, and a clear next step with timing. "
            "Also generate a CRM activity log entry."
        ),
        expected_output="Follow-up email draft, value reinforcement points, next step recommendation, CRM activity log",
        agent=agent,
    )


class SalesCallWorkflow:
    """Sales call analysis workflow."""

    def __init__(self):
        self.agent = create_sales_analyst()

    def analyze(self, transcript: str) -> str:
        analysis_task = create_transcript_analysis_task(transcript, self.agent)
        followup_task = create_follow_up_task(self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[analysis_task, followup_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
