"""Second Brain Systems.

Knowledge base management, note linking, spaced repetition,
and context retrieval for personal knowledge management.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import write_json, read_json, ensure_dir


def create_second_brain() -> Agent:
    return Agent(
        role="Knowledge Management Specialist",
        goal="Build and maintain a personal knowledge base that surfaces the right information at the right time",
        backstory=(
            "You are a personal knowledge management expert inspired by "
            "Building a Second Brain (BASB) and Zettelkasten methods. You "
            "organize notes, create bidirectional links, generate summaries, "
            "and build a searchable knowledge graph. You understand spaced "
            "repetition and progressive summarization."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_note_ingestion_task(note_content: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Ingest and process the following note/content into the second brain system:\n\n"
            f"{note_content}\n\n"
            "Extract: title, summary (2-3 sentences), key concepts, "
            "tags (5-10), related existing topics, action items, "
            "and a permanent note version (rewritten for clarity)."
        ),
        expected_output="Structured note object with title, summary, tags, concepts, linked topics, and action items",
        agent=agent,
    )


def create_knowledge_graph_task(notes: list[dict], agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze {len(notes)} notes and build a knowledge graph. "
            "Identify: topic clusters, frequently co-occurring concepts, "
            "orphan notes (no links), knowledge gaps, and suggested new connections. "
            "Output as a graph structure with nodes and edges."
        ),
        expected_output="Knowledge graph with topic clusters, connection suggestions, and gap analysis",
        agent=agent,
    )


def create_review_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Generate a weekly review report that includes: "
            "notes added this week, top 5 most connected concepts, "
            "revisitable notes (spaced repetition queue), "
            "action items from notes that haven't been addressed, "
            "and a 'note of the week' recommendation."
        ),
        expected_output="Weekly second brain review report with revisitable queue and action items",
        agent=agent,
    )


class SecondBrainWorkflow:
    """Second brain knowledge management workflow."""

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = storage_dir
        self.agent = create_second_brain()

    def ingest_note(self, content: str) -> str:
        task = create_note_ingestion_task(content, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def build_graph(self, notes: list[dict]) -> str:
        task = create_knowledge_graph_task(notes, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def weekly_review(self) -> str:
        task = create_review_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
