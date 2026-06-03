"""Investment Analyst Agent.

Market research, stock screening, portfolio analysis,
risk assessment, and investment thesis generation.
Covers DSE Bangladesh and global equity markets.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page


# ---------------------------------------------------------------------------
# DSE Bangladesh market constants
# ---------------------------------------------------------------------------

DSE_BLUE_CHIPS: list[dict] = [
    {"symbol": "BRACBANK", "name": "BRAC Bank Ltd.", "sector": "Financial"},
    {"symbol": "SQPHARMA", "name": "Square Pharmaceuticals", "sector": "Pharmaceuticals"},
    {"symbol": "EBL", "name": "Eastern Bank Ltd.", "sector": "Financial"},
    {"symbol": "JAMUNABANK", "name": "Jamuna Bank Ltd.", "sector": "Financial"},
    {"symbol": "BSRMSTEEL", "name": "BSRM Steels Ltd.", "sector": "Engineering"},
    {"symbol": "LAPTOP", "name": "LafargeHolcim Bangladesh", "sector": "Cement"},
    {"symbol": "MARICO", "name": "Marico Bangladesh", "sector": "Consumer"},
    {"symbol": "BERGERPBL", "name": "Berger Paints Bangladesh", "sector": "Paints"},
    {"symbol": "SUMITPOWER", "name": "Summit Power Ltd.", "sector": "Power"},
    {"symbol": "OLYMPIC", "name": "Olympic Industries", "sector": "Consumer"},
]

DSE_SECTORS: dict[str, list[str]] = {
    "Financial": ["BRACBANK", "EBL", "JAMUNABANK", "ONEBANK", "EXIMBANK", "IFIC"],
    "Pharmaceuticals": ["SQPHARMA", "BXPHARMA", "IBNSINA", "ACI"],
    "Engineering": ["BSRMSTEEL", "ATLASBANG", "RAKCERAMIC"],
    "Power": ["SUMITPOWER", "GENEXIL", "DOREENPOWER"],
    "Telecommunication": ["GP", "ROBI", "BANGLALINK"],
    "Consumer": ["MARICO", "OLYMPIC", "BRITANNIA", "UNILEVERCL"],
    "Cement": ["LAPTOP", "HEIDELBCEM", "ARGCEMENT"],
    "Textile": ["BSC", "BSRMLTD", "DELTALIFE"],
    "Fuel & Power": ["PADMAOIL", "MPETROLEUM", "JAMUNAOIL"],
}

DSE_DATA_SOURCES: dict[str, str] = {
    "amarstock": "https://www.amarstock.com",
    "dsebd": "https://www.dsebd.org",
    "simplywall": "https://simplywall.st/markets/bd",
    "tbsnews": "https://www.tbsnews.net/economy/stocks",
    "financial_express": "https://thefinancialexpress.com.bd/stock",
    "biniyog": "https://biniyog.com.bd",
}


def _build_dse_context() -> str:
    """Build DSE market context string for task descriptions."""
    blue_chip_line = ", ".join(
        f"{s['symbol']} ({s['name']})" for s in DSE_BLUE_CHIPS[:7]
    )
    sectors_line = ", ".join(f"{k} ({len(v)} stocks)" for k, v in DSE_SECTORS.items())
    return (
        "DSE Bangladesh Market Context:\n"
        f"- Blue chips: {blue_chip_line}\n"
        f"- Sectors: {sectors_line}\n"
        "- Market PE ~11.7x (3Y avg 13.7x) → broadly undervalued\n"
        "- ~101/360 companies are Z-category (speculative junk — avoid on gainers lists)\n"
        "- Loss-makers dominate gainers via syndicate manipulation\n"
        "- Sources: amarstock.com (real-time), simplywall.st, tbsnews.net\n"
        f"- Data URLs: {DSE_DATA_SOURCES['amarstock']}, {DSE_DATA_SOURCES['dsebd']}\n"
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_investment_analyst() -> Agent:
    return Agent(
        role="Senior Investment Analyst",
        goal="Identify profitable investment opportunities and manage portfolio risk across DSE Bangladesh and global markets",
        backstory=(
            "You are a CFA-qualified investment analyst with deep expertise "
            "in equity research, fundamental analysis, technical analysis, "
            "and portfolio management. You cover both developed and emerging "
            "markets including Bangladesh (DSE), US equities, and global ETFs. "
            "You have specific expertise in DSE Bangladesh market structure: "
            "blue-chip valuations, Z-category risk, sector PE analysis, "
            "and identifying syndicate manipulation patterns. "
            "You balance growth potential with risk management."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Stock Screening
# ---------------------------------------------------------------------------

def create_stock_screening_task(
    criteria: str,
    agent: Agent,
    *,
    market: str = "DSE",
    focus_sectors: list[str] | None = None,
    exclude_z_category: bool = True,
    max_pe: float | None = None,
    min_dividend_yield: float | None = None,
) -> Task:
    """Create a stock screening task.

    Args:
        criteria: Natural-language screening criteria.
        agent: The Investment Analyst agent.
        market: Target market (DSE, US, GLOBAL).
        focus_sectors: Optional sector filter (uses DSE_SECTORS taxonomy).
        exclude_z_category: If True (default), filter out DSE Z-category stocks.
        max_pe: Maximum P/E ratio threshold.
        min_dividend_yield: Minimum dividend yield (decimal, e.g. 0.03 for 3%).
    """
    filters: list[str] = []
    if exclude_z_category and market.upper() == "DSE":
        filters.append("EXCLUDE Z-category stocks (speculative/junk in DSE)")
    if focus_sectors:
        valid = [s for s in focus_sectors if s in DSE_SECTORS]
        if valid:
            tickers = [t for s in valid for t in DSE_SECTORS[s]]
            filters.append(f"Focus sectors: {', '.join(valid)} (tickers: {', '.join(tickers)})")
    if max_pe is not None:
        filters.append(f"P/E ratio <= {max_pe}")
    if min_dividend_yield is not None:
        filters.append(f"Dividend yield >= {min_dividend_yield:.1%}")

    filter_block = "\n".join(f"  - {f}" for f in filters) if filters else "  - None"
    dse_context = _build_dse_context() if market.upper() == "DSE" else ""

    return Task(
        description=(
            f"Screen {market} stocks matching these criteria:\n{criteria}\n\n"
            f"Active filters:\n{filter_block}\n\n"
            f"{dse_context}\n"
            "For each candidate stock, provide:\n"
            "  1. Current price and 52-week range\n"
            "  2. P/E ratio, P/B ratio, market cap\n"
            "  3. Dividend yield and payout consistency\n"
            "  4. Sector comparison vs peers\n"
            "  5. Recent news sentiment and catalysts\n"
            "  6. Technical trend (support/resistance, momentum)\n"
            "  7. Brief investment thesis (bull + bear cases)\n"
            "  8. Risk rating: Low / Medium / High / Speculative\n\n"
            "Rank candidates by risk-adjusted return potential. "
            "Include a summary table at the top."
        ),
        expected_output=(
            "Screened stock list with key metrics table, individual company analysis "
            "(price, PE, dividend yield, technicals, thesis), risk ratings, "
            "and ranking by risk-adjusted return"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Portfolio Analysis
# ---------------------------------------------------------------------------

def create_portfolio_analysis_task(
    holdings: dict,
    agent: Agent,
    *,
    benchmark: str = "DSEX",
    risk_free_rate: float = 0.08,
) -> Task:
    """Create a portfolio analysis task.

    Args:
        holdings: Dict mapping symbol → {"shares": int, "avg_cost": float, "currency": str}.
        agent: The Investment Analyst agent.
        benchmark: Benchmark index (DSEX, S&P500, etc.).
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default 8% for BD).
    """
    return Task(
        description=(
            f"Perform a comprehensive portfolio analysis.\n\n"
            f"Holdings:\n{holdings}\n\n"
            f"Benchmark: {benchmark}\n"
            f"Risk-free rate: {risk_free_rate:.1%} (annual)\n\n"
            "Analyze and report on:\n"
            "  1. Total portfolio value and cost basis\n"
            "  2. Unrealised P&L (absolute and %)\n"
            "  3. Allocation breakdown by sector and region\n"
            "  4. Concentration risk (top 1/3/5 weight, HHI index)\n"
            "  5. Weighted average portfolio P/E and dividend yield\n"
            "  6. Beta-weighted market exposure\n"
            "  7. Estimated annual dividend income\n"
            "  8. Sharpe ratio estimation (if return history available)\n"
            "  9. Underperforming positions with analysis\n"
            " 10. Rebalancing recommendations with target weights\n"
            " 11. Tax-loss harvesting opportunities (if any)\n\n"
            "Search for latest news on each holding.\n"
            f"{_build_dse_context() if 'DSE' in benchmark else ''}"
        ),
        expected_output=(
            "Portfolio analysis report with: valuation summary, allocation charts, "
            "risk metrics (HHI, beta, Sharpe), income projection, "
            "underperformer analysis, and rebalancing plan with target weights"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Market Outlook
# ---------------------------------------------------------------------------

def create_market_outlook_task(
    agent: Agent,
    *,
    scope: str = "GLOBAL",
    period: str = "weekly",
    include_dse: bool = True,
) -> Task:
    """Create a market outlook generation task.

    Args:
        agent: The Investment Analyst agent.
        scope: GLOBAL, US, or EMERGING.
        period: Outlook period (weekly, monthly, quarterly).
        include_dse: Whether to always include DSE Bangladesh section.
    """
    dse_section = ""
    if include_dse or scope in ("GLOBAL", "EMERGING"):
        dse_section = (
            "\n- DSE Bangladesh section:\n"
            "  * DSEX index trend and key levels\n"
            "  * Market PE vs historical average\n"
            "  * Sector rotation within DSE\n"
            "  * IPO and listing activity\n"
            "  * Foreign portfolio investment flow\n"
            "  * Taka/USD exchange rate impact\n"
            "  * Regulatory updates from BSEC\n"
            "  * Top DSE gainers/losers analysis (filter Z-category noise)\n"
        )

    return Task(
        description=(
            f"Generate a {period} market outlook report (scope: {scope}).\n\n"
            "Cover the following sections:\n"
            "  1. Executive Summary (3-5 bullet points)\n"
            "  2. Global Macro Snapshot\n"
            "     - Major indices performance (S&P 500, Nasdaq, DAX, Nikkei)\n"
            "     - Interest rate outlook (Fed, ECB, Bangladesh Bank)\n"
            "     - Inflation trends (US CPI, Bangladesh CPI)\n"
            "     - Currency movements (USD index, BDT/USD)\n"
            "     - Geopolitical risk factors\n"
            "  3. Sector Rotation & Themes\n"
            "     - Leading and lagging sectors\n"
            "     - AI/tech momentum, energy transition, rate-sensitive plays\n"
            "     - Emerging market themes\n"
            "  4. Emerging Opportunities\n"
            "     - 3-5 specific investment ideas with risk/reward profiles\n"
            "  5. Risk Factors to Monitor\n"
            "     - Tail risks, event calendar, economic data releases\n"
            "  6. Allocation Recommendations\n"
            "     - Strategic equity/fixed-income/cash split\n"
            "     - Sector underweight/overweight calls\n"
            f"  {dse_section}\n"
            "Use web search for latest data. Format as a professional report."
        ),
        expected_output=(
            "Weekly/market outlook report with executive summary, macro snapshot, "
            "sector analysis, opportunity spotlights, risk calendar, "
            "and allocation recommendations including DSE Bangladesh section"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Risk Assessment
# ---------------------------------------------------------------------------

def create_risk_assessment_task(
    holdings: dict | None,
    agent: Agent,
    *,
    market_data: dict | None = None,
    scenario: str = "standard",
) -> Task:
    """Create a dedicated risk assessment task.

    Args:
        holdings: Portfolio holdings (or None for market-level risk assessment).
        agent: The Investment Analyst agent.
        market_data: Optional market indicators (VIX, DSEX volatility, etc.).
        scenario: Scenario type — standard, stress, black_swan, rate_shock.
    """
    scenario_descriptions: dict[str, str] = {
        "standard": "Normal market conditions — baseline risk measurement.",
        "stress": (
            "Stress scenario: 2008-style crisis, 30% market drawdown, "
            "liquidity freeze in emerging markets."
        ),
        "black_swan": (
            "Black swan: Pandemic-style event, sovereign default risk, "
            "banking sector contagion, sudden regulatory overhaul."
        ),
        "rate_shock": (
            "Rate shock: Central bank hikes 300bps in 6 months, "
            "emerging market currency crisis (BDT -15%), capital flight."
        ),
    }

    scenario_desc = scenario_descriptions.get(scenario, scenario_descriptions["standard"])
    holdings_block = f"\nPortfolio holdings:\n{holdings}\n" if holdings else ""
    market_block = f"\nMarket indicators:\n{market_data}\n" if market_data else ""

    return Task(
        description=(
            f"Perform a comprehensive investment risk assessment.\n"
            f"Scenario: {scenario}\n"
            f"  → {scenario_desc}\n"
            f"{holdings_block}{market_block}\n"
            "Analyse and report on:\n"
            "  1. Market Risk (systematic)\n"
            "     - Beta exposure and correlation to benchmark\n"
            "     - VaR (Value at Risk) estimation at 95% and 99% confidence\n"
            "     - Conditional VaR (Expected Shortfall)\n"
            "  2. Concentration Risk\n"
            "     - Single-name risk (any position > 10%)\n"
            "     - Sector concentration\n"
            "     - Geographic/currency concentration\n"
            "  3. Liquidity Risk\n"
            "     - Average daily volume analysis\n"
            "     - Bid-ask spread estimation\n"
            "     - Days-to-liquidate analysis\n"
            "  4. Credit/Counterparty Risk (for bond/fixed income holdings)\n"
            "  5. Currency Risk\n"
            "     - FX exposure for USD/EUR-denominated holdings\n"
            "     - BDT depreciation scenarios\n"
            "  6. Regulatory & Political Risk (especially DSE)\n"
            "     - BSEC regulation changes\n"
            "     - Tax policy risk\n"
            "     - Political stability impact\n"
            "  7. Scenario Analysis\n"
            "     - P&L impact under current scenario\n"
            "     - Breakdown by source of loss\n"
            "  8. Risk Mitigation Recommendations\n"
            "     - Hedging strategies (stop-losses, options, diversification)\n"
            "     - Position sizing adjustments\n"
            "     - Cash reserve recommendations\n\n"
            "Provide a final Risk Score: 1 (low) to 10 (extreme) with justification.\n"
            f"{_build_dse_context() if holdings else ''}"
        ),
        expected_output=(
            "Risk assessment report covering market, concentration, liquidity, "
            "currency, credit, regulatory risks; VaR estimates; scenario P&L impact; "
            "risk score (1-10), and specific mitigation recommendations"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Investment Thesis Generator
# ---------------------------------------------------------------------------

def create_investment_thesis_task(
    symbol: str,
    agent: Agent,
    *,
    conviction: str = "medium",
) -> Task:
    """Create a concentrated investment thesis task for a single name.

    Args:
        symbol: Stock symbol (DQ, DSE, or global ticker).
        agent: The Investment Analyst agent.
        conviction: Analyst's conviction level (low, medium, high).
    """
    is_dse = symbol.upper() in {
        s["symbol"] for s in DSE_BLUE_CHIPS
    } or any(
        symbol.upper() in tickers for tickers in DSE_SECTORS.values()
    )
    dse_note = (
        f"\n\nNote: {symbol.upper()} is a DSE-listed stock. "
        "Apply DSE-specific analysis: check category (A/B/N/Z/NR), "
        "free-float, sponsor holding %, and dividend history.\n"
        if is_dse else ""
    )

    return Task(
        description=(
            f"Write a detailed investment thesis for {symbol.upper()}.\n"
            f"Conviction level: {conviction}\n\n"
            "Structure the thesis as follows:\n"
            "  1. Company Overview (what they do, competitive position)\n"
            "  2. Investment Summary (bull case in 3 sentences)\n"
            "  3. Key Financial Metrics (revenue growth, margins, ROE, FCF)\n"
            "  4. Valuation Analysis (PE, PB, EV/EBITDA vs peers and history)\n"
            "  5. Catalysts Timeline (what could drive the stock in 3-12 months)\n"
            "  6. Risk Factors (top 3-5 risks with probability and impact)\n"
            "  7. Technical Analysis (key levels, trend, momentum)\n"
            "  8. Comparable Companies Analysis\n"
            "  9. Price Target and Time Horizon\n"
            " 10. Conviction Check — does the risk/reward support the conviction level?\n"
            f" 11. Final Verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL{dse_note}\n"
            "Search for the latest financials and news."
        ),
        expected_output=(
            f"Complete investment thesis for {symbol.upper()} with financial analysis, "
            "valuation, catalysts, risk factors, price target, and final recommendation"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# DSE Market Scanner — specialised scanner for DSE Bangladesh
# ---------------------------------------------------------------------------

def create_dse_scanner_task(
    agent: Agent,
    *,
    scan_type: str = "full",
) -> Task:
    """Create a DSE Bangladesh market scanning task.

    Args:
        agent: The Investment Analyst agent.
        scan_type: Type of scan — full, gainers, losers, volume, dividend, undervalued.
    """
    scan_descriptions: dict[str, str] = {
        "full": "Comprehensive scan of DSE: top gainers, losers, volume leaders, dividend plays, and undervalued stocks.",
        "gainers": "Scan DSE top gainers. CRITICAL: Filter out Z-category stocks — they are often manipulated. Focus on A/B-category with real fundamentals.",
        "losers": "Scan DSE top losers. Identify whether decline is technical, fundamental, or panic-driven. Look for contrarian opportunities.",
        "volume": "Scan DSE for unusual volume activity. Volume > 2x average with price movement signals institutional interest.",
        "dividend": "Scan DSE for high-dividend-yield stocks. Target yield > 4% with consistent payout history. Exclude Z-category.",
        "undervalued": "Scan DSE for undervalued stocks: PE below sector average, PB < 2, positive earnings, A/B-category only.",
    }

    scan_desc = scan_descriptions.get(scan_type, scan_descriptions["full"])

    return Task(
        description=(
            f"DSE Bangladesh Market Scanner (scan type: {scan_type})\n\n"
            f"{scan_desc}\n\n"
            f"{_build_dse_context()}\n\n"
            "For each stock found, report:\n"
            "  - Symbol, company name, category (A/B/N/Z/NR)\n"
            "  - Current price, % change, volume\n"
            "  - P/E, market cap, dividend yield\n"
            "  - Brief reason for the move\n"
            "  - Risk flag if Z-category or low free-float\n\n"
            "Provide a summary: market breadth (advancing vs declining), "
            "sector leaders, and overall DSE market mood.\n"
            "Data sources: amarstock.com, dsebd.org, simplywall.st/markets/bd"
        ),
        expected_output=(
            f"DSE Bangladesh market scan ({scan_type}) with stock list, "
            "metrics, sector summary, risk flags, and market mood assessment"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class InvestmentAnalystWorkflow:
    """End-to-end investment analysis workflow with DSE Bangladesh focus."""

    def __init__(self):
        self.agent = create_investment_analyst()

    def screen_stocks(
        self,
        criteria: str,
        *,
        market: str = "DSE",
        focus_sectors: list[str] | None = None,
        exclude_z_category: bool = True,
        max_pe: float | None = None,
        min_dividend_yield: float | None = None,
    ) -> str:
        """Screen stocks matching criteria.

        Args:
            criteria: Natural-language screening criteria.
            market: Target market (DSE, US, GLOBAL).
            focus_sectors: Optional sector filter.
            exclude_z_category: Filter DSE Z-category junk.
            max_pe: Maximum P/E threshold.
            min_dividend_yield: Minimum dividend yield.
        """
        task = create_stock_screening_task(
            criteria,
            self.agent,
            market=market,
            focus_sectors=focus_sectors,
            exclude_z_category=exclude_z_category,
            max_pe=max_pe,
            min_dividend_yield=min_dividend_yield,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_portfolio(
        self,
        holdings: dict,
        *,
        benchmark: str = "DSEX",
        risk_free_rate: float = 0.08,
    ) -> str:
        """Analyze an investment portfolio.

        Args:
            holdings: Dict of symbol → {shares, avg_cost, currency}.
            benchmark: Benchmark index.
            risk_free_rate: Annual risk-free rate for Sharpe calc.
        """
        task = create_portfolio_analysis_task(
            holdings,
            self.agent,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def market_outlook(
        self,
        *,
        scope: str = "GLOBAL",
        period: str = "weekly",
        include_dse: bool = True,
    ) -> str:
        """Generate a market outlook report.

        Args:
            scope: GLOBAL, US, or EMERGING.
            period: weekly, monthly, quarterly.
            include_dse: Include DSE Bangladesh section.
        """
        task = create_market_outlook_task(
            self.agent,
            scope=scope,
            period=period,
            include_dse=include_dse,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def assess_risk(
        self,
        holdings: dict | None = None,
        *,
        market_data: dict | None = None,
        scenario: str = "standard",
    ) -> str:
        """Perform dedicated risk assessment.

        Args:
            holdings: Portfolio holdings (None for market-level only).
            market_data: Market indicators (VIX, volatility, etc.).
            scenario: standard, stress, black_swan, or rate_shock.
        """
        task = create_risk_assessment_task(
            holdings,
            self.agent,
            market_data=market_data,
            scenario=scenario,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def investment_thesis(self, symbol: str, conviction: str = "medium") -> str:
        """Generate a detailed investment thesis for a single stock.

        Args:
            symbol: Stock ticker.
            conviction: low, medium, or high.
        """
        task = create_investment_thesis_task(
            symbol,
            self.agent,
            conviction=conviction,
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def scan_dse(
        self,
        *,
        scan_type: str = "full",
    ) -> str:
        """Run the DSE Bangladesh market scanner.

        Args:
            scan_type: full, gainers, losers, volume, dividend, undervalued.
        """
        task = create_dse_scanner_task(self.agent, scan_type=scan_type)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
