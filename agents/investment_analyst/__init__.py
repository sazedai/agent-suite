"""Investment Analyst Agent.

Market research, stock screening, portfolio analysis,
risk assessment, and investment thesis generation.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page


def create_investment_analyst() -> Agent:
    return Agent(
        role="Senior Investment Analyst",
        goal="Identify profitable investment opportunities and manage portfolio risk",
        backstory=(
            "You are a CFA-qualified investment analyst with deep expertise "
            "in equity research, fundamental analysis, technical analysis, "
            "and portfolio management. You cover both developed and emerging "
            "markets including Bangladesh (DSE), US equities, and global ETFs. "
            "You balance growth potential with risk management."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_stock_screening_task(criteria: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Screen stocks matching these criteria: {criteria}. "
            "Use web search to find stocks matching the screening parameters. "
            "For each candidate: current price, P/E ratio, market cap, "
            "dividend yield, 52-week range, recent news sentiment, "
            "and a brief investment thesis. Rank by risk-adjusted return potential."
        ),
        expected_output="Screened stock list with key metrics, thesis, and risk rating for each candidate",
        agent=agent,
    )


def create_portfolio_analysis_task(holdings: dict, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze this investment portfolio:\n\n{holdings}\n\n"
            "Calculate: total value, allocation by sector/region, "
            "concentration risk, dividend income projection, "
            "beta-weighted risk, correlation analysis, "
            "underperforming positions, and rebalancing recommendations. "
            "Search for latest news on each holding."
        ),
        expected_output="Portfolio analysis report with risk metrics, income projections, and rebalancing recommendations",
        agent=agent,
    )


def create_market_outlook_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Generate a weekly market outlook covering: "
            "major indices performance, sector rotation trends, "
            "interest rate and inflation impact, geopolitical developments, "
            "emerging opportunities, and risk factors to monitor. "
            "Include both US and Bangladesh (DSE) market perspectives."
        ),
        expected_output="Weekly market outlook report with opportunities and risk factors",
        agent=agent,
    )


class InvestmentAnalystWorkflow:
    """Investment analysis workflow."""

    def __init__(self):
        self.agent = create_investment_analyst()

    def screen_stocks(self, criteria: str) -> str:
        task = create_stock_screening_task(criteria, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_portfolio(self, holdings: dict) -> str:
        task = create_portfolio_analysis_task(holdings, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def market_outlook(self) -> str:
        task = create_market_outlook_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
