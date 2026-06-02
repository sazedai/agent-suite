"""Research Agent for AI/News Summaries.

Monitors news sources, summarizes articles, identifies trends,
and generates research reports on specified topics.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page, write_json


def create_researcher() -> Agent:
    return Agent(
        role="AI & Technology Research Analyst",
        goal="Monitor, summarize, and analyze news and developments in AI and technology",
        backstory=(
            "You are a technology research analyst who tracks AI developments, "
            "startup funding, product launches, and industry trends. You read "
            "HackerNews, TechCrunch, arXiv, and research blogs daily. You distill "
            "complex topics into executive summaries and identify market-moving trends."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_news_scan_task(topic: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Scan the latest news and developments about '{topic}'. "
            "Search across: tech news sites, AI research papers, HackerNews, "
            "Reddit r/MachineLearning, and company blogs. "
            "Collect the top 10 most relevant articles/developments from the past 7 days."
        ),
        expected_output="List of top 10 news items with title, URL, source, date, and brief description",
        agent=agent,
    )


def create_summary_task(agent: Agent) -> Task:
    return Task(
        description=(
            "For each news item found, fetch the full article and create a "
            "structured summary covering: what happened, why it matters, "
            "key quotes, technical details, and implications for the industry. "
            "Also identify any emerging patterns across multiple stories."
        ),
        expected_output="Detailed summaries with key insights and cross-story trend analysis",
        agent=agent,
    )


def create_report_task(topic: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Compile a weekly research briefing on '{topic}' that includes: "
            "executive summary, top 5 stories, key trends, companies to watch, "
            "and predictions for the next period. Format as a professional report "
            "suitable for executive consumption."
        ),
        expected_output="Executive research briefing document with trends, predictions, and actionable insights",
        agent=agent,
    )


class ResearchWorkflow:
    """End-to-end research and news summary workflow."""

    def __init__(self, topic: str = "artificial intelligence"):
        self.topic = topic
        self.agent = create_researcher()

    def run(self) -> str:
        scan_task = create_news_scan_task(self.topic, self.agent)
        summary_task = create_summary_task(self.agent)
        report_task = create_report_task(self.topic, self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[scan_task, summary_task, report_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
