"""Advisory Council with NotebookLM.

Multi-perspective advisory panel that simulates expert opinions,
analyzes documents (NotebookLM-style), provides structured decision support,
and produces risk assessments with probability-impact matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RiskItem:
    """Single risk entry in the assessment matrix."""
    category: str
    description: str
    probability: float  # 0.0 – 1.0
    impact: float       # 0.0 – 1.0
    risk_score: float = 0.0
    mitigation: str = ""
    level: RiskLevel = RiskLevel.LOW

    def __post_init__(self):
        self.risk_score = round(self.probability * self.impact, 3)
        if self.risk_score >= 0.7:
            self.level = RiskLevel.CRITICAL
        elif self.risk_score >= 0.5:
            self.level = RiskLevel.HIGH
        elif self.risk_score >= 0.3:
            self.level = RiskLevel.MEDIUM
        elif self.risk_score >= 0.1:
            self.level = RiskLevel.LOW
        else:
            self.level = RiskLevel.NEGLIGIBLE


@dataclass
class AdvisorOpinion:
    """Structured opinion from a single advisor."""
    advisor_name: str
    expertise: str
    assessment: str
    reasoning: str
    conditions: str = ""
    confidence: Confidence = Confidence.MEDIUM
    supports: bool = True
    dissenting_views: str = ""


@dataclass
class DecisionMatrix:
    """Structured decision support matrix."""
    question: str
    options: list[dict] = field(default_factory=list)
    # Each option: {"name": str, "scores": {criterion: float}, "total": float}
    criteria: list[str] = field(default_factory=list)
    recommendation: str = ""
    confidence: Confidence = Confidence.MEDIUM
    rationale: str = ""


@dataclass
class DocumentChunk:
    """A chunk of an ingested document (NotebookLM-style)."""
    index: int
    content: str
    source: str
    relevance_score: float = 0.0


@dataclass
class AdvisoryReport:
    """Full advisory council output."""
    question: str
    context: str
    opinions: list[AdvisorOpinion] = field(default_factory=list)
    points_of_agreement: list[str] = field(default_factory=list)
    points_of_disagreement: list[str] = field(default_factory=list)
    risk_assessment: list[RiskItem] = field(default_factory=list)
    decision_matrix: DecisionMatrix | None = None
    executive_summary: str = ""
    recommendation: str = ""
    confidence: Confidence = Confidence.MEDIUM
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "executive_summary": self.executive_summary,
            "recommendation": self.recommendation,
            "confidence": self.confidence.value,
            "opinions": [
                {
                    "advisor": o.advisor_name,
                    "expertise": o.expertise,
                    "assessment": o.assessment,
                    "supports": o.supports,
                    "confidence": o.confidence.value,
                    "dissenting_views": o.dissenting_views,
                }
                for o in self.opinions
            ],
            "points_of_agreement": self.points_of_agreement,
            "points_of_disagreement": self.points_of_disagreement,
            "risk_assessment": [
                {
                    "category": r.category,
                    "description": r.description,
                    "probability": r.probability,
                    "impact": r.impact,
                    "risk_score": r.risk_score,
                    "level": r.level.value,
                    "mitigation": r.mitigation,
                }
                for r in self.risk_assessment
            ],
            "next_steps": self.next_steps,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Advisor factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Document processing (NotebookLM-style)
# ---------------------------------------------------------------------------

def chunk_document(content: str, chunk_size: int = 2000, overlap: int = 200) -> list[DocumentChunk]:
    """Split a document into overlapping chunks for analysis.

    NotebookLM-style: breaks long documents into manageable pieces
    with overlap to preserve context at boundaries.
    """
    if len(content) <= chunk_size:
        return [DocumentChunk(index=0, content=content, source="document")]

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    idx = 0
    pos = 0
    content_len = len(content)
    while pos < content_len:
        end = min(pos + chunk_size, content_len)
        # Try to break at paragraph boundary
        if end < content_len:
            para_break = content.rfind("\n\n", pos, end)
            if para_break > pos + chunk_size // 2:
                end = para_break
            else:
                # Fall back to sentence boundary
                sent_break = content.rfind(". ", pos, end)
                if sent_break > pos + chunk_size // 2:
                    end = sent_break + 1

        chunks.append(DocumentChunk(
            index=idx,
            content=content[pos:end].strip(),
            source="document",
        ))
        idx += 1
        pos += step
        if end >= content_len:
            break

    return chunks


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


def create_document_qa_task(question: str, chunk: DocumentChunk, agent: Agent) -> Task:
    return Task(
        description=(
            f"Answer the following question based ONLY on the provided document excerpt.\n\n"
            f"Question: {question}\n\n"
            f"Document Excerpt (chunk {chunk.index}):\n{chunk.content}\n\n"
            "Rules:\n"
            "- Only use information from the provided excerpt.\n"
            "- If the excerpt does not contain enough information, say so clearly.\n"
            "- Cite specific passages from the excerpt to support your answer.\n"
            "- Be precise and avoid speculation."
        ),
        expected_output="Source-grounded answer with citations from the document excerpt",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Multi-perspective analysis tasks
# ---------------------------------------------------------------------------

def create_advisor_opinion_task(
    question: str,
    context: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Advisory Council Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Provide your expert opinion structured as:\n"
            "1. Your assessment (clear position)\n"
            "2. Your reasoning (evidence-based)\n"
            "3. Conditions under which you would support this\n"
            "4. Any dissenting views or caveats\n"
            "5. Your confidence level (high/medium/low)\n"
            "6. Key risks from your perspective"
        ),
        expected_output="Structured expert opinion with assessment, reasoning, conditions, and confidence",
        agent=agent,
    )


def create_synthesis_task(
    question: str,
    context: str,
) -> Task:
    return Task(
        description=(
            f"You are the Advisory Council Chair. Synthesize all advisor opinions into a unified report.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            "Produce a structured advisory report with:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Points of Agreement (list)\n"
            "3. Points of Disagreement (list)\n"
            "4. Risk Assessment (top risks with probability and impact ratings)\n"
            "5. Risk-Adjusted Recommendation\n"
            "6. Confidence Level (high/medium/low)\n"
            "7. Suggested Next Steps (actionable)"
        ),
        expected_output="Unified advisory report synthesizing all expert perspectives with structured sections",
        agent=None,  # Will be set to the chair agent
    )


# ---------------------------------------------------------------------------
# Decision support tasks
# ---------------------------------------------------------------------------

def create_decision_matrix_task(
    question: str,
    options: list[str],
    criteria: list[str],
    agent: Agent,
) -> Task:
    options_list = "\n".join(f"  - {o}" for o in options)
    criteria_list = "\n".join(f"  - {c}" for c in criteria)
    return Task(
        description=(
            f"Build a decision matrix for the following question:\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_list}\n\n"
            f"Evaluation Criteria:\n{criteria_list}\n\n"
            "For each option, score every criterion on a scale of 1-10.\n"
            "Calculate weighted totals (equal weights unless specified).\n"
            "Provide a clear recommendation with rationale.\n"
            "Highlight trade-offs between the top 2 options."
        ),
        expected_output="Decision matrix with scored options, totals, recommendation, and trade-off analysis",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Risk assessment tasks
# ---------------------------------------------------------------------------

def create_risk_assessment_task(proposal: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Assess the risks of this proposal:\n\n{proposal}\n\n"
            "Identify and rate risks in these categories:\n"
            "- Financial risks\n"
            "- Operational risks\n"
            "- Market risks\n"
            "- Regulatory/compliance risks\n"
            "- Reputation risks\n"
            "- Technology risks\n"
            "- Black swan scenarios\n\n"
            "For each risk provide:\n"
            "1. Description\n"
            "2. Probability (0.0 to 1.0)\n"
            "3. Impact (0.0 to 1.0)\n"
            "4. Risk score (probability × impact)\n"
            "5. Mitigation strategy\n\n"
            "Sort by risk score descending. Highlight the top 5 risks."
        ),
        expected_output="Risk assessment matrix with probability-impact ratings and mitigation strategies",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Advisory Council Workflow
# ---------------------------------------------------------------------------

class AdvisoryCouncilWorkflow:
    """Multi-agent advisory council with document analysis (NotebookLM-style).

    Features:
        - Multi-perspective analysis from 5 expert advisors
        - Document ingestion, chunking, and source-grounded Q&A
        - Structured decision support with scoring matrices
        - Risk assessment with probability-impact ratings
        - Structured advisory reports (JSON-serializable)
    """

    def __init__(self):
        self.advisors = [
            create_advisor(
                "Finance Expert", "financial analysis",
                "Conservative, risk-aware financial perspective focused on ROI, cash flow, and capital efficiency",
            ),
            create_advisor(
                "Strategy Expert", "business strategy",
                "Growth-oriented strategic perspective focused on market positioning, competitive advantage, "
                "and long-term value",
            ),
            create_advisor(
                "Operations Expert", "operations management",
                "Execution-focused perspective focused on feasibility, resource constraints, and operational risk",
            ),
            create_advisor(
                "Technology Expert", "technology and AI",
                "Innovation-focused perspective on technical feasibility, AI implications, and digital transformation",
            ),
            create_advisor(
                "Legal Expert", "legal and compliance",
                "Risk and compliance perspective focused on regulatory requirements, contracts, and liability",
            ),
        ]
        self.chair = self.advisors[0]  # Finance Expert chairs the council

    # ----- Document Processing (NotebookLM-style) -----

    def ingest_document(self, content: str, chunk_size: int = 2000, overlap: int = 200) -> list[DocumentChunk]:
        """Ingest a document and split into chunks for analysis."""
        return chunk_document(content, chunk_size=chunk_size, overlap=overlap)

    def analyze_document(self, doc_content: str, focus: str = "general") -> str:
        """Analyze a document with all advisors, each from their perspective.

        If ``focus`` matches an advisor's expertise area, that advisor leads.
        Otherwise all advisors analyze in parallel.
        """
        chunks = chunk_document(doc_content)
        if len(chunks) == 1:
            return self._analyze_single_chunk(doc_content, focus)
        return self._analyze_chunked_document(chunks, focus)

    def _analyze_single_chunk(self, content: str, focus: str) -> str:
        advisor = self._find_advisor(focus)
        task = create_document_analysis_task(content, advisor)
        crew = Crew(agents=[advisor], tasks=[task], verbose=True)
        return str(crew.kickoff())

    def _analyze_chunked_document(self, chunks: list[DocumentChunk], focus: str) -> str:
        """Analyze a multi-chunk document, then synthesize."""
        advisor = self._find_advisor(focus)
        chunk_tasks = [
            create_document_analysis_task(chunk.content, advisor)
            for chunk in chunks
        ]
        crew = Crew(agents=[advisor], tasks=chunk_tasks, verbose=True)
        results = crew.kickoff()
        return str(results)

    def answer_question(self, question: str, doc_content: str) -> str:
        """NotebookLM-style: answer a question grounded in a document.

        Chunks the document, finds the most relevant chunk, and produces
        a source-grounded answer.
        """
        chunks = chunk_document(doc_content)
        # Use the first chunk for Q&A (in production, would use embeddings)
        best_chunk = chunks[0]
        if len(chunks) > 1:
            # Simple keyword matching to find most relevant chunk
            question_words = set(question.lower().split())
            best_score = 0
            for chunk in chunks:
                chunk_words = set(chunk.content.lower().split())
                score = len(question_words & chunk_words)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            best_chunk.relevance_score = best_score

        task = create_document_qa_task(question, best_chunk, self.chair)
        crew = Crew(agents=[self.chair], tasks=[task], verbose=True)
        return str(crew.kickoff())

    # ----- Multi-Perspective Analysis -----

    def council_deliberate(self, question: str, context: str = "") -> AdvisoryReport:
        """Run a full multi-perspective deliberation and return a structured report.

        Each advisor gives their opinion, then the chair synthesizes into an
        ``AdvisoryReport``.
        """
        # Phase 1: Gather opinions from all advisors
        opinion_tasks = [
            create_advisor_opinion_task(question, context, advisor)
            for advisor in self.advisors
        ]
        opinion_crew = Crew(
            agents=self.advisors,
            tasks=opinion_tasks,
            process=Process.sequential,
            verbose=True,
        )
        opinion_crew.kickoff()

        # Phase 2: Synthesize
        synth_task = create_synthesis_task(question, context)
        synth_task.agent = self.chair
        synth_crew = Crew(agents=[self.chair], tasks=[synth_task], verbose=True)
        synthesis = str(synth_crew.kickoff())

        report = AdvisoryReport(
            question=question,
            context=context,
            executive_summary=synthesis,
            recommendation=synthesis,
        )
        return report

    def council_vote(self, question: str, context: str = "") -> str:
        """Get the full advisory council's perspective on a question.

        Each advisor provides their expert opinion in parallel.
        """
        all_advisors = self.advisors
        tasks = [
            create_advisor_opinion_task(question, context, advisor)
            for advisor in all_advisors
        ]
        crew = Crew(agents=all_advisors, tasks=tasks, process=Process.sequential, verbose=True)
        return str(crew.kickoff())

    # ----- Decision Support -----

    def decision_matrix(
        self,
        question: str,
        options: list[str],
        criteria: list[str],
    ) -> str:
        """Build a scored decision matrix for a set of options against criteria."""
        task = create_decision_matrix_task(question, options, criteria, self.chair)
        crew = Crew(agents=[self.chair], tasks=[task], verbose=True)
        return str(crew.kickoff())

    # ----- Risk Assessment -----

    def assess_risks(self, proposal: str) -> str:
        """Run a structured risk assessment on a proposal.

        All advisors assess risks from their perspective; the chair synthesizes.
        """
        tasks = [
            create_risk_assessment_task(proposal, advisor)
            for advisor in self.advisors
        ]
        crew = Crew(agents=self.advisors, tasks=tasks, process=Process.sequential, verbose=True)
        return str(crew.kickoff())

    def build_risk_matrix(self, risks: list[dict]) -> list[RiskItem]:
        """Build a structured RiskItem list from raw risk dicts.

        Each dict should have keys: category, description, probability, impact.
        Optional: mitigation.
        """
        items = []
        for r in risks:
            item = RiskItem(
                category=r["category"],
                description=r["description"],
                probability=float(r["probability"]),
                impact=float(r["impact"]),
                mitigation=r.get("mitigation", ""),
            )
            items.append(item)
        items.sort(key=lambda x: x.risk_score, reverse=True)
        return items

    # ----- Helpers -----

    def _find_advisor(self, focus: str) -> Agent:
        focus_lower = focus.lower()
        for advisor in self.advisors:
            if focus_lower in advisor.role.lower():
                return advisor
        return self.advisors[0]
