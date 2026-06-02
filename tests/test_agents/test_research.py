"""Tests for the Research Agent."""

from __future__ import annotations

import json
import pytest

from agents.research_agent import (
    NewsArticle,
    TrendSignal,
    ResearchReport,
    scan_news,
    fetch_and_summarize,
    detect_trends,
    generate_report,
    report_to_markdown,
    create_researcher,
    create_news_scan_task,
    create_summary_task,
    create_report_task,
    ResearchWorkflow,
    NEWS_SOURCES,
    CATEGORIES,
    _classify_category,
    _extract_technical_details,
    _impact_prediction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_article():
    """A minimal NewsArticle for testing."""
    return NewsArticle(
        title="OpenAI releases GPT-5 with breakthrough reasoning",
        url="https://example.com/openai-gpt5",
        source="TechCrunch",
        date="2026-06-03",
        summary="OpenAI announced GPT-5 showing major advances in reasoning.",
    )


@pytest.fixture
def sample_articles() -> list[NewsArticle]:
    """A collection of NewsArticles spanning multiple categories."""
    return [
        NewsArticle(
            title="OpenAI releases GPT-5 with breakthrough reasoning",
            url="https://example.com/1",
            source="TechCrunch",
            summary="Major advances in LLM reasoning capability.",
            category="Large Language Models",
            content="The new model shows 95% accuracy on complex benchmarks.",
        ),
        NewsArticle(
            title="Google DeepMind unveils robot that learns from video",
            url="https://example.com/2",
            source="Wired",
            summary="Robotics breakthrough using video-based learning.",
            category="Robotics",
            content="The robot achieves 80% success rate in manipulation tasks.",
        ),
        NewsArticle(
            title="Mistral releases new open-weight LLM",
            url="https://example.com/2b",
            source="arXiv",
            summary="New open-weight model rivals GPT-4 class.",
            category="Large Language Models",
            content="The model achieves 92% on standard benchmarks.",
        ),
        NewsArticle(
            title="Anthropic raises $2B Series D for AI safety research",
            url="https://example.com/3",
            source="VentureBeat",
            summary="Funding round focused on AI safety and alignment.",
            category="Startup & Funding",
            content="Anthropic raised $2 billion to advance safety research.",
        ),
        NewsArticle(
            title="New EU AI Act regulations take effect",
            url="https://example.com/4",
            source="The Verge",
            summary="EU AI regulation framework begins enforcement.",
            category="AI Regulation & Policy",
            content="The AI Act requires compliance from all major providers.",
        ),
        NewsArticle(
            title="NVIDIA unveils next-gen AI training chips",
            url="https://example.com/5",
            source="Ars Technica",
            summary="New hardware promises 10x training speedup.",
            category="Hardware & Chips",
            content="The new GPU delivers 10x throughput with 50% less power.",
        ),
    ]


@pytest.fixture
def sample_trends() -> list[TrendSignal]:
    """A list of sample trends."""
    return [
        TrendSignal(
            name="Growing focus on Large Language Models",
            description="3 out of 5 articles discussed Large Language Models.",
            strength=0.6,
            related_articles=["Article 1", "Article 2", "Article 3"],
            category="Large Language Models",
            predicted_impact="Continued rapid iteration expected.",
        ),
        TrendSignal(
            name="AI Safety",
            description="Safety-related themes appear across multiple articles.",
            strength=0.4,
            related_articles=["Article 4", "Article 5"],
            predicted_impact="More safety benchmarks coming.",
        ),
    ]


@pytest.fixture
def sample_report(sample_articles, sample_trends) -> ResearchReport:
    """A sample research report."""
    return generate_report("artificial intelligence", sample_articles, sample_trends)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestNewsArticle:
    def test_creation(self, sample_article):
        assert sample_article.title == "OpenAI releases GPT-5 with breakthrough reasoning"
        assert sample_article.url == "https://example.com/openai-gpt5"
        assert sample_article.source == "TechCrunch"
        assert sample_article.key_quotes == []

    def test_to_dict(self, sample_article):
        d = sample_article.to_dict()
        assert d["title"] == sample_article.title
        assert d["url"] == sample_article.url
        assert d["source"] == sample_article.source
        assert isinstance(d["key_quotes"], list)

    def test_default_values(self):
        article = NewsArticle(title="Test", url="https://test.com", source="Test")
        assert article.date == ""
        assert article.summary == ""
        assert article.content == ""
        assert article.category == ""
        assert article.key_quotes == []
        assert article.technical_details == ""
        assert article.implications == ""


class TestTrendSignal:
    def test_creation(self):
        trend = TrendSignal(
            name="LLM Boom",
            description="LLM mentions increasing",
            strength=0.8,
            related_articles=["A", "B"],
        )
        assert trend.name == "LLM Boom"
        assert trend.strength == 0.8
        assert trend.predicted_impact == ""

    def test_to_dict(self, sample_trends):
        d = sample_trends[0].to_dict()
        assert d["name"] == "Growing focus on Large Language Models"
        assert d["strength"] == 0.6
        assert isinstance(d["related_articles"], list)


class TestResearchReport:
    def test_creation(self):
        report = ResearchReport(topic="AI", period="2026-06-03")
        assert report.topic == "AI"
        assert report.period == "2026-06-03"
        assert report.articles == []
        assert report.trends == []

    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert d["topic"] == "artificial intelligence"
        assert "generated_at" in d
        assert isinstance(d["articles"], list)
        assert isinstance(d["trends"], list)
        assert isinstance(d["companies_to_watch"], list)
        assert isinstance(d["predictions"], list)


# ---------------------------------------------------------------------------
# Category classification tests
# ---------------------------------------------------------------------------

class TestClassifyCategory:
    def test_llm_category(self):
        text = "OpenAI released a new large language model with improved token processing"
        assert _classify_category(text) == "Large Language Models"

    def test_vision_category(self):
        text = "The image diffusion model generates high quality visual content"
        assert _classify_category(text) == "Computer Vision"

    def test_robotics_category(self):
        text = "The robot can perform complex manipulation tasks in new environments"
        assert _classify_category(text) == "Robotics"

    def test_safety_category(self):
        text = "AI safety and alignment research prevents harm from RLHF training"
        assert _classify_category(text) == "AI Safety & Alignment"

    def test_regulation_category(self):
        text = "New government policy and law regulate AI development"
        assert _classify_category(text) == "AI Regulation & Policy"

    def test_funding_category(self):
        text = "The startup raised series funding at a high valuation"
        assert _classify_category(text) == "Startup & Funding"

    def test_research_category(self):
        text = "A new arxiv paper shows sota results on benchmark evaluation"
        assert _classify_category(text) == "Research Papers"

    def test_product_category(self):
        text = "The company will release and announce the new product launching soon"
        assert _classify_category(text) == "Product Launch"

    def test_opensource_category(self):
        text = "The open source project on GitHub and Hugging Face is gaining traction"
        assert _classify_category(text) == "Open Source"

    def test_hardware_category(self):
        text = "NVIDIA unveils new GPU chips and hardware accelerators"
        assert _classify_category(text) == "Hardware & Chips"

    def test_default_category(self):
        text = "Weather today is sunny with clear skies"
        assert _classify_category(text) == "Industry Analysis"


# ---------------------------------------------------------------------------
# Technical detail extraction tests
# ---------------------------------------------------------------------------

class TestExtractTechnicalDetails:
    def test_percentage_detection(self):
        content = "The model achieved 95% accuracy on the test set. Results show improvement."
        result = _extract_technical_details(content)
        assert "95%" in result

    def test_throughput_detection(self):
        content = "The system processes 500 tokens per second with low latency."
        result = _extract_technical_details(content)
        assert "latency" in result.lower() or "tokens" in result.lower()

    def test_no_technical_content(self):
        content = "The company announced a new CEO. The board felt this was the right move."
        result = _extract_technical_details(content)
        assert result == ""

    def test_max_snippets(self):
        lines = [f"Metric {i}: 9{i}% accuracy on benchmark {i}" for i in range(10)]
        content = "\n".join(lines)
        result = _extract_technical_details(content)
        assert result.count("\n") <= 4  # at most 5 snippets


# ---------------------------------------------------------------------------
# Trend detection tests
# ---------------------------------------------------------------------------

class TestDetectTrends:
    def test_empty_list(self):
        result = detect_trends([])
        assert result == []

    def test_category_trend(self, sample_articles):
        trends = detect_trends(sample_articles)
        assert len(trends) >= 1
        # Each category should have 1 article; need 2+ for a trend
        # Since each category appears once, no category trend expected

    def test_repeated_category_produces_trend(self):
        articles = [
            NewsArticle(title="LLM one", url="u1", source="S1", category="Large Language Models"),
            NewsArticle(title="LLM two", url="u2", source="S2", category="Large Language Models"),
            NewsArticle(title="LLM three", url="u3", source="S3", category="Large Language Models"),
        ]
        trends = detect_trends(articles)
        assert any("Large Language Models" in t.name for t in trends)
        assert len(trends) >= 1

    def test_trends_sorted_by_strength(self):
        articles = [
            NewsArticle(title="A", url="u1", source="S1", category="LLM"),
            NewsArticle(title="B", url="u2", source="S2", category="LLM"),
            NewsArticle(title="C", url="u3", source="S3", category="Robotics"),
        ]
        trends = detect_trends(articles)
        strengths = [t.strength for t in trends]
        assert strengths == sorted(strengths, reverse=True)

    def test_bigram_trend(self):
        articles = [
            NewsArticle(title="LLM safety alignment one", url="u1", source="S1"),
            NewsArticle(title="LLM safety alignment two", url="u2", source="S2"),
            NewsArticle(title="LLM safety alignment three", url="u3", source="S3"),
        ]
        trends = detect_trends(articles)
        # "llm safety" or "safety alignment" bigram should trigger
        assert len(trends) >= 1


# ---------------------------------------------------------------------------
# Impact prediction tests
# ---------------------------------------------------------------------------

class TestImpactPrediction:
    def test_known_category(self):
        result = _impact_prediction("Large Language Models")
        assert "iteration" in result.lower() or "model" in result.lower()

    def test_unknown_category(self):
        result = _impact_prediction("Something New")
        assert "evolution" in result.lower()

    def test_all_categories_covered(self):
        for cat in CATEGORIES:
            result = _impact_prediction(cat)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_basic_report(self, sample_articles, sample_trends):
        report = generate_report("AI", sample_articles, sample_trends)
        assert report.topic == "AI"
        assert len(report.articles) >= 5
        assert len(report.trends) == 2
        assert report.generated_at != ""

    def test_report_exec_summary(self, sample_articles, sample_trends):
        report = generate_report("AI", sample_articles, sample_trends)
        assert "Executive Summary" in report.executive_summary or "##" in report.executive_summary

    def test_report_companies(self):
        articles = [
            NewsArticle(
                title="OpenAI and Google compete on LLM benchmarks",
                url="u1", source="TechCrunch",
                content="OpenAI released a model. Google responded with their own. "
                        "Anthropic and Microsoft also joined the race.",
            ),
        ]
        report = generate_report("AI", articles, [])
        company_names = [c["name"] for c in report.companies_to_watch]
        assert any("OpenAI" in n or "Openai" in n for n in company_names)
        assert any("Google" in n for n in company_names)

    def test_report_predictions_from_trends(self, sample_trends):
        articles = [NewsArticle(title="A", url="u1", source="S1")]
        report = generate_report("AI", articles, sample_trends)
        assert len(report.predictions) >= 1

    def test_report_predictions_default(self):
        articles = [NewsArticle(title="A", url="u1", source="S1")]
        report = generate_report("AI", articles, [])
        assert any("monitor" in p.lower() or "recommend" in p.lower() or "evolve" in p.lower() for p in report.predictions)

    def test_report_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert isinstance(d, dict)
        assert len(d["articles"]) >= 5
        assert d["companies_to_watch"] is not None

    def test_empty_articles(self):
        report = generate_report("AI", [], [])
        assert report.topic == "AI"
        assert len(report.articles) == 0
        assert report.executive_summary != ""


# ---------------------------------------------------------------------------
# Markdown rendering tests
# ---------------------------------------------------------------------------

class TestReportToMarkdown:
    def test_markdown_contains_title(self, sample_report):
        md = report_to_markdown(sample_report)
        assert "# Research Report: artificial intelligence" in md

    def test_markdown_contains_sections(self, sample_report):
        md = report_to_markdown(sample_report)
        assert "## Executive Summary" in md
        assert "## Top Stories" in md
        assert "## Detected Trends" in md

    def test_markdown_contains_articles(self, sample_report):
        md = report_to_markdown(sample_report)
        assert "OpenAI releases GPT-5" in md
        assert "Google DeepMind" in md

    def test_markdown_contains_companies_table(self, sample_report):
        md = report_to_markdown(sample_report)
        if sample_report.companies_to_watch:
            assert "## Companies to Watch" in md
            assert "| Company | Mentions |" in md

    def test_markdown_contains_predictions(self, sample_report):
        md = report_to_markdown(sample_report)
        assert "## Predictions" in md

    def test_markdown_contains_methodology(self, sample_report):
        md = report_to_markdown(sample_report)
        assert "## Methodology" in md

    def test_markdown_trend_details(self, sample_report):
        md = report_to_markdown(sample_report)
        for trend in sample_report.trends:
            assert trend.name in md
            assert trend.description in md


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_news_sources_not_empty(self):
        assert len(NEWS_SOURCES) > 0
        assert "TechCrunch" in NEWS_SOURCES
        assert "arXiv" in NEWS_SOURCES

    def test_categories_not_empty(self):
        assert len(CATEGORIES) > 0
        assert "Large Language Models" in CATEGORIES
        assert "AI Safety & Alignment" in CATEGORIES


# ---------------------------------------------------------------------------
# Integration: data flow tests
# ---------------------------------------------------------------------------

class TestDataFlow:
    def test_full_pipeline_with_sample_data(self):
        """Simulate the full pipeline: articles → trends → report → markdown."""
        articles = [
            NewsArticle(
                title="OpenAI GPT-5 achieves 98% on reasoning benchmarks",
                url="https://example.com/gpt5",
                source="TechCrunch",
                summary="GPT-5 shows breakthrough reasoning performance.",
                category="Large Language Models",
                content="The model achieves 98% accuracy. Latency reduced by 50%.",
            ),
            NewsArticle(
                title="Meta releases open-source LLM for vision tasks",
                url="https://example.com/meta",
                source="The Verge",
                summary="Open source multimodal model released.",
                category="Open Source",
                content="The model has 70B parameters. Training took 30 days on GPU cluster.",
            ),
            NewsArticle(
                title="Anthropic raises $5B for AI safety alignment research",
                url="https://example.com/anthropic",
                source="VentureBeat",
                summary="Massive funding round for safety.",
                category="Startup & Funding",
                content="Anthropic raised $5 billion. RLHF techniques improved.",
            ),
        ]

        trends = detect_trends(articles)
        report = generate_report("AI", articles, trends)
        md = report_to_markdown(report)

        assert report.topic == "AI"
        assert len(report.articles) == 3
        assert md.startswith("# Research Report:")

    def test_article_to_dict_json_roundtrip(self, sample_article):
        """Ensure to_dict output is JSON-serializable."""
        d = sample_article.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["title"] == sample_article.title
        assert parsed["url"] == sample_article.url

    def test_report_to_dict_json_roundtrip(self, sample_report):
        d = sample_report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["topic"] == "artificial intelligence"
        assert len(parsed["articles"]) >= 5


# ---------------------------------------------------------------------------
# CrewAI agent tests (existing compatibility)
# ---------------------------------------------------------------------------

class TestCrewAIAgent:
    def test_create_researcher(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_researcher()
        assert agent.role == "AI & Technology Research Analyst"
        assert agent.allow_delegation is False

    def test_create_news_scan_task(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_researcher()
        task = create_news_scan_task("generative AI", agent)
        assert task.agent == agent
        assert "generative AI" in task.description

    def test_create_summary_task(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_researcher()
        task = create_summary_task(agent)
        assert task.agent == agent
        assert "structured summary" in task.description

    def test_create_report_task(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_researcher()
        task = create_report_task("transformers", agent)
        assert task.agent == agent
        assert "transformers" in task.description

    def test_research_workflow_creation(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        workflow = ResearchWorkflow("generative AI")
        assert workflow.topic == "generative AI"
        assert workflow.agent is not None

    def test_research_workflow_default_topic(self, monkeypatch):
        monkeypatch.setattr(
            "agents.research_agent.get_llm",
            lambda: "gpt-4o",
        )
        workflow = ResearchWorkflow()
        assert workflow.topic == "artificial intelligence"
