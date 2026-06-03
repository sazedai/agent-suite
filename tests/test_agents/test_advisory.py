"""Tests for Advisory Council agent."""

import json
import inspect
import pytest
from unittest.mock import patch, MagicMock

from agents.advisory_council import (
    AdvisoryCouncilWorkflow,
    AdvisoryReport,
    AdvisorOpinion,
    Confidence,
    DecisionMatrix,
    DocumentChunk,
    RiskItem,
    RiskLevel,
    chunk_document,
    create_advisor,
    create_decision_matrix_task,
    create_document_analysis_task,
    create_document_qa_task,
    create_risk_assessment_task,
    create_synthesis_task,
    create_advisor_opinion_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_get_llm():
    """Patch get_llm so CrewAI Agent creation does not hit real APIs."""
    with patch("agents.advisory_council.get_llm", return_value="gpt-4o") as mock:
        yield mock


@pytest.fixture
def council(mock_get_llm):
    """AdvisoryCouncilWorkflow with mocked LLM."""
    return AdvisoryCouncilWorkflow()


@pytest.fixture
def sample_long_document():
    """A multi-paragraph document for testing chunking."""
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Section {i}: " + "This is a detailed paragraph about business strategy. " * 20
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_risks():
    """Sample risk data for build_risk_matrix testing."""
    return [
        {
            "category": "Financial",
            "description": "Market downturn reduces revenue by 30%",
            "probability": 0.4,
            "impact": 0.8,
            "mitigation": "Diversify revenue streams and maintain cash reserves",
        },
        {
            "category": "Operational",
            "description": "Key team member departure",
            "probability": 0.6,
            "impact": 0.5,
            "mitigation": "Cross-train team and document processes",
        },
        {
            "category": "Regulatory",
            "description": "New compliance requirements",
            "probability": 0.3,
            "impact": 0.6,
            "mitigation": "Engage legal counsel early and monitor regulatory changes",
        },
        {
            "category": "Technology",
            "description": "Critical system outage",
            "probability": 0.2,
            "impact": 0.9,
            "mitigation": "Implement redundancy and disaster recovery",
        },
        {
            "category": "Reputation",
            "description": "Negative press coverage",
            "probability": 0.5,
            "impact": 0.4,
            "mitigation": "Proactive PR and rapid response protocol",
        },
    ]


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------

def test_module_imports():
    """All public symbols should be importable."""
    from agents.advisory_council import (
        AdvisoryCouncilWorkflow,
        chunk_document,
        create_advisor,
    )
    assert AdvisoryCouncilWorkflow is not None
    assert create_advisor is not None
    assert chunk_document is not None


# ---------------------------------------------------------------------------
# RiskItem tests
# ---------------------------------------------------------------------------

class TestRiskItem:

    def test_risk_score_calculation(self):
        risk = RiskItem(
            category="Financial",
            description="Test risk",
            probability=0.5,
            impact=0.8,
        )
        assert risk.risk_score == 0.4

    def test_risk_level_critical(self):
        risk = RiskItem(category="Test", description="d", probability=0.9, impact=0.9)
        assert risk.level == RiskLevel.CRITICAL

    def test_risk_level_high(self):
        risk = RiskItem(category="Test", description="d", probability=0.7, impact=0.8)
        assert risk.level == RiskLevel.HIGH

    def test_risk_level_medium(self):
        risk = RiskItem(category="Test", description="d", probability=0.5, impact=0.6)
        assert risk.level == RiskLevel.MEDIUM

    def test_risk_level_low(self):
        risk = RiskItem(category="Test", description="d", probability=0.4, impact=0.4)
        assert risk.level == RiskLevel.LOW

    def test_risk_level_negligible(self):
        risk = RiskItem(category="Test", description="d", probability=0.05, impact=0.1)
        assert risk.level == RiskLevel.NEGLIGIBLE

    def test_risk_with_mitigation(self):
        risk = RiskItem(
            category="Ops",
            description="Failure",
            probability=0.5,
            impact=0.5,
            mitigation="Have a backup plan",
        )
        assert risk.mitigation == "Have a backup plan"


# ---------------------------------------------------------------------------
# build_risk_matrix tests
# ---------------------------------------------------------------------------

class TestBuildRiskMatrix:

    def test_builds_sorted_risks(self, council, sample_risks):
        items = council.build_risk_matrix(sample_risks)
        assert len(items) == 5
        # Should be sorted by risk_score descending
        for i in range(len(items) - 1):
            assert items[i].risk_score >= items[i + 1].risk_score

    def test_highest_risk_is_first(self, council, sample_risks):
        items = council.build_risk_matrix(sample_risks)
        # Highest: Financial 0.4*0.8=0.32, Ops 0.6*0.5=0.3, Reg 0.3*0.6=0.18, Tech 0.2*0.9=0.18, Rep 0.5*0.4=0.2
        assert items[0].category == "Financial"
        assert items[0].risk_score == 0.32

    def test_empty_list(self, council):
        items = council.build_risk_matrix([])
        assert items == []

    def test_single_risk(self, council):
        items = council.build_risk_matrix([
            {"category": "Test", "description": "d", "probability": 0.5, "impact": 0.5}
        ])
        assert len(items) == 1
        assert items[0].risk_score == 0.25

    def test_mitigation_preserved(self, council, sample_risks):
        items = council.build_risk_matrix(sample_risks)
        financial = [i for i in items if i.category == "Financial"][0]
        assert financial.mitigation == "Diversify revenue streams and maintain cash reserves"


# ---------------------------------------------------------------------------
# chunk_document tests
# ---------------------------------------------------------------------------

class TestChunkDocument:

    def test_short_document_single_chunk(self):
        doc = "Short document content."
        chunks = chunk_document(doc, chunk_size=2000)
        assert len(chunks) == 1
        assert chunks[0].content == doc
        assert chunks[0].index == 0

    def test_chunk_count(self, sample_long_document):
        chunks = chunk_document(sample_long_document, chunk_size=500, overlap=100)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self, sample_long_document):
        chunks = chunk_document(sample_long_document, chunk_size=500, overlap=100)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunks_have_content(self, sample_long_document):
        chunks = chunk_document(sample_long_document, chunk_size=500, overlap=100)
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert chunk.source == "document"

    def test_overlap_preserves_content(self, sample_long_document):
        chunks = chunk_document(sample_long_document, chunk_size=800, overlap=200)
        # Total covered content should be close to the full document
        total_content = sum(len(c.content) for c in chunks)
        # With overlap, total will be more than the document length
        assert total_content > len(sample_long_document)

    def test_custom_chunk_size(self):
        doc = "x" * 10000
        chunks = chunk_document(doc, chunk_size=1000, overlap=100)
        assert len(chunks) >= 9

    def test_empty_document(self):
        chunk = chunk_document("", chunk_size=2000)
        assert len(chunk) == 1
        assert chunk[0].content == ""


# ---------------------------------------------------------------------------
# AdvisoryReport tests
# ---------------------------------------------------------------------------

class TestAdvisoryReport:

    def test_to_dict_structure(self):
        report = AdvisoryReport(
            question="Should we expand?",
            context="Market analysis data",
            executive_summary="Yes, with caution",
            recommendation="Expand to 2 new markets",
            confidence=Confidence.HIGH,
            next_steps=["Hire team", "Secure funding"],
        )
        d = report.to_dict()
        assert d["question"] == "Should we expand?"
        assert d["confidence"] == "high"
        assert d["next_steps"] == ["Hire team", "Secure funding"]
        assert "opinions" in d
        assert "risk_assessment" in d

    def test_to_json_is_valid(self):
        report = AdvisoryReport(
            question="Test",
            context="Context",
            executive_summary="Summary",
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["question"] == "Test"

    def test_with_risks(self):
        risks = [
            RiskItem(category="Fin", description="Revenue drop", probability=0.4, impact=0.8),
        ]
        report = AdvisoryReport(
            question="Test",
            context="C",
            risk_assessment=risks,
        )
        d = report.to_dict()
        assert len(d["risk_assessment"]) == 1
        assert d["risk_assessment"][0]["category"] == "Fin"


# ---------------------------------------------------------------------------
# AdvisoryCouncilWorkflow unit tests
# ---------------------------------------------------------------------------

class TestAdvisoryCouncilWorkflow:

    def test_init_creates_five_advisors(self, council):
        assert len(council.advisors) == 5

    def test_advisor_roles(self, council):
        roles = [a.role for a in council.advisors]
        assert "Advisor: Finance Expert" in roles
        assert "Advisor: Strategy Expert" in roles
        assert "Advisor: Operations Expert" in roles
        assert "Advisor: Technology Expert" in roles
        assert "Advisor: Legal Expert" in roles

    def test_chair_is_set(self, council):
        assert council.chair is not None
        assert council.chair == council.advisors[0]

    def test_find_advisor_by_focus(self, council):
        advisor = council._find_advisor("Finance")
        assert "Finance" in advisor.role

    def test_find_advisor_fallback(self, council):
        advisor = council._find_advisor("nonexistent")
        assert advisor == council.advisors[0]

    def test_ingest_document_returns_chunks(self, council, sample_long_document):
        chunks = council.ingest_document(sample_long_document, chunk_size=500, overlap=100)
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert len(chunks) > 1

    def test_ingest_document_short_single_chunk(self, council):
        chunks = council.ingest_document("Short text.")
        assert len(chunks) == 1

    # --- Task creation signature tests ---

    def test_create_advisor_opinion_task_signature(self):
        sig = inspect.signature(create_advisor_opinion_task)
        assert "question" in sig.parameters
        assert "context" in sig.parameters
        assert "agent" in sig.parameters

    def test_create_decision_matrix_task_signature(self):
        sig = inspect.signature(create_decision_matrix_task)
        assert "question" in sig.parameters
        assert "options" in sig.parameters
        assert "criteria" in sig.parameters
        assert "agent" in sig.parameters

    def test_create_document_qa_task_signature(self):
        sig = inspect.signature(create_document_qa_task)
        assert "question" in sig.parameters
        assert "chunk" in sig.parameters
        assert "agent" in sig.parameters

    def test_create_synthesis_task_signature(self):
        sig = inspect.signature(create_synthesis_task)
        assert "question" in sig.parameters
        assert "context" in sig.parameters

    def test_create_risk_assessment_task_signature(self):
        sig = inspect.signature(create_risk_assessment_task)
        assert "proposal" in sig.parameters
        assert "agent" in sig.parameters

    # --- Task creation return tests (with mock agent) ---

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_document_analysis_task_returns_task(self, _):
        from crewai import Task
        agent = create_advisor("Test", "testing", "test perspective")
        task = create_document_analysis_task("Test content", agent)
        assert isinstance(task, Task)

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_risk_assessment_returns_task(self, _):
        from crewai import Task
        agent = create_advisor("Test", "testing", "test perspective")
        task = create_risk_assessment_task("Proposal text", agent)
        assert isinstance(task, Task)
        assert "Proposal text" in task.description

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_advisor_opinion_task_returns_task(self, _):
        from crewai import Task
        agent = create_advisor("Test", "testing", "test perspective")
        task = create_advisor_opinion_task("What to do?", "Context info", agent)
        assert isinstance(task, Task)
        assert "What to do?" in task.description

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_synthesis_task_returns_task(self, _):
        from crewai import Task
        task = create_synthesis_task("Question", "Context")
        assert isinstance(task, Task)
        assert "Chair" in task.description

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_decision_matrix_task_returns_task(self, _):
        from crewai import Task
        agent = create_advisor("Test", "testing", "test perspective")
        task = create_decision_matrix_task(
            "Build or buy?",
            ["Build in-house", "Buy SaaS"],
            ["Cost", "Time to market", "Flexibility"],
            agent,
        )
        assert isinstance(task, Task)
        assert "Build or buy?" in task.description

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_create_document_qa_task_returns_task(self, _):
        from crewai import Task
        agent = create_advisor("Test", "testing", "test perspective")
        chunk = DocumentChunk(index=0, content="Source text about revenue.", source="doc")
        task = create_document_qa_task("What is the revenue?", chunk, agent)
        assert isinstance(task, Task)
        assert "What is the revenue?" in task.description
        assert "Source text about revenue" in task.description


# ---------------------------------------------------------------------------
# Integration-style tests (mock crew.kickoff)
# ---------------------------------------------------------------------------

class TestAdvisoryCouncilIntegration:

    @patch("agents.advisory_council.Crew")
    def test_analyze_document_short(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Analysis result"
        mock_crew_cls.return_value = mock_crew

        result = council.analyze_document("Short doc content", focus="general")
        assert result == "Analysis result"
        mock_crew_cls.assert_called_once()
        mock_crew.kickoff.assert_called_once()

    @patch("agents.advisory_council.Crew")
    def test_council_vote(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = ["Opinion 1", "Opinion 2"]
        mock_crew_cls.return_value = mock_crew

        result = council.council_vote("Should we pivot?", context="Market data")
        assert isinstance(result, str)
        mock_crew_cls.assert_called_once()

    @patch("agents.advisory_council.Crew")
    def test_assess_risks(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Risk report"
        mock_crew_cls.return_value = mock_crew

        result = council.assess_risks("Launch new product")
        assert result == "Risk report"

    @patch("agents.advisory_council.Crew")
    def test_decision_matrix(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Matrix result"
        mock_crew_cls.return_value = mock_crew

        result = council.decision_matrix(
            "Which framework?",
            ["Option A", "Option B"],
            ["Speed", "Cost", "Scalability"],
        )
        assert result == "Matrix result"

    @patch("agents.advisory_council.Crew")
    def test_answer_question_short_doc(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "The revenue is $1M"
        mock_crew_cls.return_value = mock_crew

        result = council.answer_question(
            "What is the revenue?",
            "Annual report content with revenue figures.",
        )
        assert result == "The revenue is $1M"

    @patch("agents.advisory_council.Crew")
    def test_answer_question_long_doc(self, mock_crew_cls, council, sample_long_document):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Answer from document"
        mock_crew_cls.return_value = mock_crew

        result = council.answer_question(
            "What is the strategy?",
            sample_long_document,
        )
        assert result == "Answer from document"

    @patch("agents.advisory_council.Crew")
    def test_council_deliberate(self, mock_crew_cls, council):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Synthesized report"
        mock_crew_cls.return_value = mock_crew

        report = council.council_deliberate(
            "Should we expand internationally?",
            context="Current revenue $5M, 3 markets",
        )
        assert isinstance(report, AdvisoryReport)
        assert report.question == "Should we expand internationally?"
        assert report.context == "Current revenue $5M, 3 markets"


# ---------------------------------------------------------------------------
# Confidence and RiskLevel enum tests
# ---------------------------------------------------------------------------

class TestEnums:

    def test_confidence_values(self):
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"

    def test_risk_level_values(self):
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.NEGLIGIBLE.value == "negligible"


# ---------------------------------------------------------------------------
# create_advisor tests
# ---------------------------------------------------------------------------

class TestCreateAdvisor:

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_returns_agent(self, _):
        agent = create_advisor("Test Person", "testing", "test perspective")
        assert agent.role == "Advisor: Test Person"
        assert "testing" in agent.goal
        assert "test perspective" in agent.backstory
        assert agent.allow_delegation is False

    @patch("agents.advisory_council.get_llm", return_value="gpt-4o")
    def test_agent_name_in_backstory(self, _):
        agent = create_advisor("Alice", "finance", "conservative view")
        assert "Alice" in agent.backstory
        assert "conservative view" in agent.backstory


# ---------------------------------------------------------------------------
# AdvisorOpinion dataclass tests
# ---------------------------------------------------------------------------

class TestAdvisorOpinion:

    def test_defaults(self):
        opinion = AdvisorOpinion(
            advisor_name="Finance",
            expertise="financial analysis",
            assessment="Support",
            reasoning="Strong ROI",
        )
        assert opinion.confidence == Confidence.MEDIUM
        assert opinion.supports is True
        assert opinion.dissenting_views == ""

    def test_dissenting_opinion(self):
        opinion = AdvisorOpinion(
            advisor_name="Legal",
            expertise="compliance",
            assessment="Oppose",
            reasoning="High regulatory risk",
            supports=False,
            dissenting_views="GDPR concerns",
            confidence=Confidence.HIGH,
        )
        assert opinion.supports is False
        assert opinion.dissenting_views == "GDPR concerns"


# ---------------------------------------------------------------------------
# DecisionMatrix tests
# ---------------------------------------------------------------------------

class TestDecisionMatrix:

    def test_empty_defaults(self):
        dm = DecisionMatrix(question="Test?")
        assert dm.question == "Test?"
        assert dm.options == []
        assert dm.criteria == []
        assert dm.recommendation == ""
