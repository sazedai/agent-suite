"""Research Agent for AI/News Summaries.

Monitors news sources, summarizes articles, identifies trends,
and generates research reports on specified topics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """Represents a single news article or development."""
    title: str
    url: str
    source: str
    date: str = ""
    summary: str = ""
    content: str = ""
    category: str = ""
    key_quotes: list[str] = field(default_factory=list)
    technical_details: str = ""
    implications: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "date": self.date,
            "summary": self.summary,
            "category": self.category,
            "key_quotes": self.key_quotes,
            "technical_details": self.technical_details,
            "implications": self.implications,
        }


@dataclass
class TrendSignal:
    """Represents a detected trend across multiple articles."""
    name: str
    description: str
    strength: float  # 0.0 - 1.0
    related_articles: list[str] = field(default_factory=list)
    category: str = ""
    predicted_impact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strength": self.strength,
            "related_articles": self.related_articles,
            "category": self.category,
            "predicted_impact": self.predicted_impact,
        }


@dataclass
class ResearchReport:
    """Complete research report for a topic and time period."""
    topic: str
    period: str
    generated_at: str = ""
    executive_summary: str = ""
    articles: list[NewsArticle] = field(default_factory=list)
    trends: list[TrendSignal] = field(default_factory=list)
    companies_to_watch: list[dict[str, str]] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    methodology: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "period": self.period,
            "generated_at": self.generated_at,
            "executive_summary": self.executive_summary,
            "articles": [a.to_dict() for a in self.articles],
            "trends": [t.to_dict() for t in self.trends],
            "companies_to_watch": self.companies_to_watch,
            "predictions": self.predictions,
            "methodology": self.methodology,
        }


# ---------------------------------------------------------------------------
# News scanning
# ---------------------------------------------------------------------------

# Curated news sources for AI/tech research
NEWS_SOURCES = [
    "TechCrunch", "Wired", "The Verge", "Ars Technica",
    "MIT Technology Review", "VentureBeat", "IEEE Spectrum",
    "Hacker News", "arXiv", "Semantic Scholar",
    "OpenAI Blog", "Google AI Blog", "Anthropic Blog",
    "Microsoft Research", "DeepMind Blog",
    "Reddit r/MachineLearning", "Reddit r/artificial",
]

# Common categories for classification
CATEGORIES = [
    "Large Language Models", "Computer Vision", "Robotics",
    "AI Safety & Alignment", "AI Regulation & Policy",
    "Startup & Funding", "Research Papers",
    "Product Launch", "Industry Analysis",
    "Open Source", "Hardware & Chips",
]


async def scan_news(
    topic: str,
    sources: list[str] | None = None,
    limit_per_source: int = 5,
    max_articles: int = 20,
) -> list[NewsArticle]:
    """Scan multiple news sources for articles on a topic.

    Args:
        subject:           Topic to search for.
        sources:           Specific sources to scan (defaults to NEWS_SOURCES).
        limit_per_source:  Max results per source.
        max_articles:      Overall cap on returned articles.

    Returns:
        Deduplicated list of NewsArticle objects.
    """
    sources = sources or NEWS_SOURCES
    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    for source in sources:
        if len(articles) >= max_articles:
            break
        query = f"{topic} site:{source.lower().replace(' ', '')}" if source not in (
            "Hacker News", "Reddit r/MachineLearning", "Reddit r/artificial",
            "arXiv", "Semantic Scholar",
        ) else f"{topic} {source}"

        try:
            raw = await web_search(query, limit=limit_per_source)
        except Exception:
            continue

        for item in raw.get("results", []):
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(
                NewsArticle(
                    title=item.get("title", "Untitled"),
                    url=url,
                    source=source,
                    date=item.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                    summary=item.get("snippet", "")[:500],
                )
            )
            if len(articles) >= max_articles:
                break

    return articles[:max_articles]


# ---------------------------------------------------------------------------
# Article summarization
# ---------------------------------------------------------------------------

async def fetch_and_summarize(article: NewsArticle) -> NewsArticle:
    """Fetch full article content and enrich with summary and analysis.

    Args:
        article: A NewsArticle with at minimum title and url set.

    Returns:
        The same article mutated with content, key_quotes, technical_details,
        and implications populated.
    """
    try:
        content = await fetch_page(article.url)
        article.content = content[:8000]  # keep within token budget
    except Exception:
        content = article.summary

    # Classify category
    article.category = _classify_category(article.title + " " + content)

    # Extract key quotes (heuristic: lines with quotation marks in content)
    quotes: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if '"' in line and 20 < len(line) < 300:
            quotes.append(line)
        if len(quotes) >= 5:
            break
    article.key_quotes = quotes

    article.technical_details = _extract_technical_details(content)
    article.implications = ""
    return article


def _classify_category(text: str) -> str:
    """Simple keyword-based category classifier."""
    text_lower = text.lower()
    keyword_map: dict[str, list[str]] = {
        "Large Language Models": ["llm", "large language model", "gpt", "transformer", "token"],
        "Computer Vision": ["vision", "image", "visual", "diffusion", "multimodal"],
        "Robotics": ["robot", "embodied", "manipulation", "grasping"],
        "AI Safety & Alignment": ["safety", "alignment", "constitution", "rlhf", "harm"],
        "AI Regulation & Policy": ["regulation", "policy", "law", "government", "ban", "act"],
        "Startup & Funding": ["funding", "raised", "series", "valuation", "startup"],
        "Research Papers": ["paper", "arxiv", "benchmark", "sota", "state of the art"],
        "Product Launch": ["launch", "release", "announce", "introducing", "new"],
        "Open Source": ["open source", "open-source", "github", "hugging face"],
        "Hardware & Chips": ["gpu", "chip", "nvidia", "tpu", "hardware", "accelerator"],
    }
    scores: Counter[str] = Counter()
    for category, words in keyword_map.items():
        for w in words:
            if w in text_lower:
                scores[category] += 1

    if scores:
        return scores.most_common(1)[0][0]
    return "Industry Analysis"


def _extract_technical_details(content: str) -> str:
    """Extract technical snippets from article content.

    Looks for paragraphs or sentences with numbers, percentages,
    or technical terminology.
    """
    technical_snippets: list[str] = []
    tech_indicators = ["%", "accuracy", "f1", "bleu", "rouge", "latency",
                       "throughput", "parameters", "tokens", "flops",
                       "training", "fine-tun", "benchmark", "evaluation"]

    for line in content.splitlines():
        line = line.strip()
        if not line or len(line) < 30:
            continue
        lower = line.lower()
        if any(ind in lower for ind in tech_indicators):
            technical_snippets.append(line)
        if len(technical_snippets) >= 5:
            break

    return "\n".join(technical_snippets)


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def detect_trends(articles: list[NewsArticle]) -> list[TrendSignal]:
    """Detect emerging trends across a collection of articles.

    Uses keyword co-occurrence, category clustering, and recency
    weighting to surface the strongest signals.

    Args:
        articles: List of (ideally summarised) NewsArticle objects.

    Returns:
        Ranked list of TrendSignal objects, strongest first.
    """
    if not articles:
        return []

    # Category distribution
    cat_counter: Counter[str] = Counter()
    for a in articles:
        cat_counter[a.category] += 1

    # Noun-phrase extraction (simple bigram heuristic)
    bigram_counter: Counter[str] = Counter()
    stop_words = {"the", "a", "an", "and", "or", "but", "with", "for",
                  "from", "this", "that", "are", "was", "were", "been",
                  "has", "have", "had", "will", "would", "could", "should",
                  "not", "its", "into", "more", "most", "such", "than",
                  "they", "their", "them", "what", "when", "which", "who",
                  "how", "all", "each", "every", "both", "few", "some",
                  "new", "also", "can", "may", "about", "after", "over",
                  "said", "just", "only", "very", "too", "so", "if",
                  "your", "you", "we", "it", "is", "in", "of", "to"}

    for a in articles:
        title_words = re.findall(r"[A-Za-z][A-Za-z0-9]+", a.title.lower())
        filtered = [w for w in title_words if w not in stop_words]
        for i in range(len(filtered) - 1):
            bigram = f"{filtered[i]} {filtered[i+1]}"
            bigram_counter[bigram] += 1

    trends: list[TrendSignal] = []

    # Category-based trends
    total = len(articles)
    for category, count in cat_counter.most_common(5):
        if count < 2:
            continue
        strength = min(count / max(total, 1), 1.0)
        related = [a.title for a in articles if a.category == category][:5]
        trends.append(TrendSignal(
            name=f"Growing focus on {category}",
            description=(
                f"{count} out of {total} articles discussed {category}, "
                f"indicating it as a dominant theme in this period."
            ),
            strength=round(strength, 2),
            related_articles=related,
            category=category,
            predicted_impact=_impact_prediction(category),
        ))

    # Bigram-based trends (cross-cutting themes)
    for bigram, count in bigram_counter.most_common(3):
        if count < 2:
            continue
        related = [a.title for a in articles if bigram in a.title.lower()][:5]
        related += [
            a.title for a in articles
            if bigram in (a.content or "").lower() and a.title not in related
        ][:3]
        strength = min(count / max(total, 1), 1.0)
        if any(t.name == bigram.title() for t in trends):
            continue
        trends.append(TrendSignal(
            name=bigram.title(),
            description=(
                f"The term '{bigram}' appears {count} times across article "
                f"titles, suggesting an emerging theme."
            ),
            strength=round(strength, 2),
            related_articles=related,
        ))

    trends.sort(key=lambda t: t.strength, reverse=True)
    return trends[:10]


def _impact_prediction(category: str) -> str:
    """Return a generic impact prediction for well-known categories."""
    predictions: dict[str, str] = {
        "Large Language Models": (
            "Expect continued rapid iteration on model capabilities, with "
            "potential breakthroughs in reasoning and multi-modal understanding."
        ),
        "AI Safety & Alignment": (
            "Increased investment in safety research and potential industry-wide "
            "safety benchmarks within the next quarter."
        ),
        "AI Regulation & Policy": (
            "Regulatory clarity may reshape competitive dynamics, favouring "
            "organisations with strong compliance frameworks."
        ),
        "Startup & Funding": (
            "Funding activity signals market confidence; expect follow-on rounds "
            "and potential acquisitions."
        ),
        "Hardware & Chips": (
            "Supply constraints and new chip architectures will influence "
            "training costs and model accessibility."
        ),
        "Research Papers": (
            "Strong research output accelerates open-source solutions and "
            "narrows the gap between academia and industry."
        ),
        "Product Launch": (
            "New products will drive adoption cycles and raise user expectations "
            "for AI-native experiences."
        ),
        "Open Source": (
            "Open-source momentum will commoditise baseline capabilities and "
            "push differentiation toward fine-tuning and deployment."
        ),
    }
    return predictions.get(category, (
        "Continued developments in this area will contribute to the broader "
        "evolution of the AI landscape."
    ))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    topic: str,
    articles: list[NewsArticle],
    trends: list[TrendSignal],
) -> ResearchReport:
    """Generate a structured research report from articles and trends.

    Args:
        topic:   The subject of research.
        articles: Analysed NewsArticle objects.
        trends:   Detected TrendSignal objects.

    Returns:
        A fully populated ResearchReport.
    """
    now = datetime.now(timezone.utc)
    period = f"{now.strftime('%Y-%m-%d')}"

    # Companies to watch — extract from titles & content
    org_pattern = re.compile(
        r"\b(OpenAI|Google|Anthropic|Microsoft|Meta|Mistral|Cohere|DeepMind|Nvidia|Intel|AMD|Apple|Amazon|Databricks|Hugging\s*Face|Stability\s*AI|Midjourney|xAI|Groq|Cerebras|SambaNova)\b",
        re.IGNORECASE,
    )
    company_counter: Counter[str] = Counter()
    all_titles = " ".join(a.title for a in articles)
    all_content = " ".join((a.content or "")[:500] for a in articles)
    for match in org_pattern.finditer(all_titles + " " + all_content):
        company_counter[match.group().title()] += 1

    companies_to_watch = [
        {"name": name, "mentions": str(count)}
        for name, count in company_counter.most_common(10)
    ]

    # Derive predictions from top trends
    predictions: list[str] = []
    for t in trends[:5]:
        if t.predicted_impact:
            predictions.append(f"[{t.name}] {t.predicted_impact}")
    if not predictions:
        predictions.append(
            "The AI landscape continues to evolve rapidly; monitoring these "
            "sources daily is recommended for early signal detection."
        )

    # Build executive summary
    exec_parts: list[str] = [
        f"## Research Briefing: {topic}",
        f"**Period:** {period}",
        f"**Articles Analysed:** {len(articles)}",
        f"**Trends Identified:** {len(trends)}",
        "",
        "### Executive Summary",
    ]

    # Top categories
    cat_counts: Counter[str] = Counter(a.category for a in articles if a.category)
    top_cats = [c for c, _ in cat_counts.most_common(3)]

    if top_cats:
        exec_parts.append(
            f"The most prominent themes this period were: {', '.join(top_cats)}."
        )
    if trends:
        top = trends[0]
        exec_parts.append(
            f"The strongest trend signal is **{top.name}** "
            f"(strength: {top.strength:.0%}), indicating {top.description.lower()}"
        )
    if companies_to_watch:
        exec_parts.append(
            f"Key companies mentioned: {', '.join(c['name'] for c in companies_to_watch[:5])}."
        )

    exec_parts.append("")

    # Top stories
    exec_parts.append("### Top Stories")
    for i, a in enumerate(articles[:5], 1):
        exec_parts.append(f"{i}. **{a.title}** ({a.source})")
        if a.summary:
            exec_parts.append(f"   {a.summary[:200]}")
        exec_parts.append("")

    report = ResearchReport(
        topic=topic,
        period=period,
        generated_at=now.isoformat(),
        executive_summary="\n".join(exec_parts),
        articles=articles,
        trends=trends,
        companies_to_watch=companies_to_watch,
        predictions=predictions,
        methodology=(
            f"News was scanned across {len(NEWS_SOURCES)} sources using web search. "
            f"Articles were deduplicated, categorised by keyword analysis, and "
            f"trends were detected via category clustering and bigram co-occurrence. "
            f"Companies were extracted via named-entity matching."
        ),
    )
    return report


def report_to_markdown(report: ResearchReport) -> str:
    """Render a ResearchReport as a Markdown document.

    Args:
        report: The report to render.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = [
        f"# Research Report: {report.topic}",
        f"**Generated:** {report.generated_at}",
        f"**Period:** {report.period}",
        "",
    ]

    # Executive summary
    lines.append("## Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")

    # Top stories
    lines.append("## Top Stories")
    for i, a in enumerate(report.articles[:10], 1):
        lines.append(f"### {i}. {a.title}")
        lines.append(f"- **Source:** {a.source}")
        if a.url:
            lines.append(f"- **URL:** {a.url}")
        if a.category:
            lines.append(f"- **Category:** {a.category}")
        if a.summary:
            lines.append(f"- **Summary:** {a.summary}")
        if a.key_quotes:
            lines.append("- **Key Quotes:**")
            for q in a.key_quotes[:3]:
                lines.append(f"  > {q}")
        if a.technical_details:
            lines.append(f"- **Technical Details:** {a.technical_details[:300]}")
        lines.append("")

    # Trends
    lines.append("## Detected Trends")
    for t in report.trends:
        lines.append(f"### {t.name}")
        lines.append(f"- **Strength:** {t.strength:.0%}")
        lines.append(f"- **Description:** {t.description}")
        if t.predicted_impact:
            lines.append(f"- **Predicted Impact:** {t.predicted_impact}")
        lines.append("")

    # Companies to watch
    if report.companies_to_watch:
        lines.append("## Companies to Watch")
        lines.append("| Company | Mentions |")
        lines.append("|---------|----------|")
        for c in report.companies_to_watch:
            lines.append(f"| {c['name']} | {c['mentions']} |")
        lines.append("")

    # Predictions
    lines.append("## Predictions")
    for p in report.predictions:
        lines.append(f"- {p}")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append(report.methodology)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CrewAI agent & tasks (existing + enhanced)
# ---------------------------------------------------------------------------

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
