"""Second Brain Systems.

Knowledge base management, note linking, spaced repetition,
and context retrieval for personal knowledge management.

Implements a BASB + Zettelkasten-inspired system with:
- Note ingestion with auto-tagging and linking
- Knowledge graph with topic clusters and gap analysis
- Spaced repetition review queue (SM-2 inspired)
- Weekly review generation with analytics
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import write_json, read_json, ensure_dir


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Note Ingestion
# ---------------------------------------------------------------------------

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


class NoteIngestionPipeline:
    """Ingest raw content, extract structure, assign IDs, and persist notes.

    Each note gets:
    - Unique ID (UUID)
    - Title, summary, tags, concepts
    - Linked topic IDs (bidirectional)
    - Action items
    - Timestamps (created, updated)
    - Spaced repetition metadata
    """

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.notes_dir = self.storage_dir / "notes"
        self.index_path = self.storage_dir / "index.json"
        ensure_dir(str(self.storage_dir))
        ensure_dir(str(self.notes_dir))

    # -- public API ---------------------------------------------------------

    def ingest(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        concepts: list[str] | None = None,
        links: list[str] | None = None,
        action_items: list[str] | None = None,
    ) -> dict:
        """Ingest a new note into the brain.

        Args:
            title: Note title.
            content: Raw note body.
            tags: Optional tags (auto-extracted if not provided).
            concepts: Optional key concepts (auto-extracted if not provided).
            links: Optional IDs of related notes to link.
            action_items: Optional action items extracted from the note.

        Returns:
            The persisted note dict.
        """
        now = datetime.utcnow().isoformat()
        note = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "summary": _summarize(content),
            "tags": tags or [],
            "concepts": concepts or [],
            "links": links or [],
            "action_items": action_items or [],
            "created_at": now,
            "updated_at": now,
            "sr_metadata": _init_sr_metadata(now),
        }
        self._save_note(note)
        self._add_to_index(note)
        return note

    def bulk_ingest(self, notes: list[dict]) -> list[dict]:
        """Ingest multiple notes at once.

        Each dict must have 'title' and 'content' keys.
        Optional: 'tags', 'concepts', 'links', 'action_items'.
        """
        return [self.ingest(**n) for n in notes]

    def get_note(self, note_id: str) -> dict | None:
        """Retrieve a single note by ID."""
        path = self.notes_dir / f"{note_id}.json"
        if not path.exists():
            return None
        return read_json(str(path))

    def all_notes(self) -> list[dict]:
        """Return all stored notes."""
        notes = []
        for f in sorted(self.notes_dir.glob("*.json")):
            try:
                notes.append(read_json(str(f)))
            except (json.JSONDecodeError, OSError):
                continue
        return notes

    def search(self, query: str) -> list[dict]:
        """Search notes by title, summary, tags, and content (case-insensitive)."""
        q = query.lower()
        results = []
        for note in self.all_notes():
            searchable = " ".join([
                note.get("title", ""),
                note.get("summary", ""),
                note.get("content", ""),
                " ".join(note.get("tags", [])),
            ]).lower()
            if q in searchable:
                results.append(note)
        return results

    # -- internal -----------------------------------------------------------

    def _save_note(self, note: dict) -> None:
        path = self.notes_dir / f"{note['id']}.json"
        write_json(str(path), note)

    def _add_to_index(self, note: dict) -> None:
        index = self._load_index()
        index[note["id"]] = {
            "title": note["title"],
            "tags": note["tags"],
            "concepts": note["concepts"],
            "created_at": note["created_at"],
        }
        write_json(str(self.index_path), index)

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return read_json(str(self.index_path))
        return {}


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

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


class KnowledgeGraph:
    """Build and query a knowledge graph from stored notes.

    The graph is stored as an adjacency structure:
    {
        "nodes": [{"id", "label", "type": "note"|"concept"|"tag"}, ...],
        "edges": [{"source", "target", "relation", "weight"}, ...],
        "clusters": [{"label", "nodes": [...], "density": float}, ...],
        "orphans": [note_id, ...],
        "gaps": [{"concept": str, "note_count": int}, ...],
    }
    """

    GRAPH_PATH = "graph.json"

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.graph_path = self.storage_dir / self.GRAPH_PATH
        ensure_dir(str(self.storage_dir))

    # -- public API ---------------------------------------------------------

    def build(self, notes: list[dict]) -> dict:
        """Build a knowledge graph from a list of note dicts."""
        nodes: list[dict] = []
        edges: list[dict] = []
        concept_notes: dict[str, list[str]] = {}
        tag_notes: dict[str, list[str]] = {}

        # Create note nodes
        for note in notes:
            nid = note["id"]
            nodes.append({"id": nid, "label": note.get("title", "Untitled"), "type": "note"})

        # Index concepts and tags per note
        all_concepts: set[str] = set()
        all_tags: set[str] = set()
        linked_notes: set[str] = set()

        for note in notes:
            nid = note["id"]
            for concept in note.get("concepts", []):
                concept = concept.strip().lower()
                if concept:
                    all_concepts.add(concept)
                    concept_notes.setdefault(concept, []).append(nid)
            for tag in note.get("tags", []):
                tag = tag.strip().lower()
                if tag:
                    all_tags.add(tag)
                    tag_notes.setdefault(tag, []).append(nid)
            for link_id in note.get("links", []):
                linked_notes.add(nid)
                linked_notes.add(link_id)

        # Create concept nodes and edges
        for concept in all_concepts:
            nodes.append({"id": concept, "label": concept, "type": "concept"})
            note_ids = concept_notes[concept]
            for nid in note_ids:
                edges.append({
                    "source": nid,
                    "target": concept,
                    "relation": "has_concept",
                    "weight": 1,
                })
            # Connect notes sharing the same concept
            if len(note_ids) > 1:
                for i, a in enumerate(note_ids):
                    for b in note_ids[i + 1:]:
                        edges.append({
                            "source": a,
                            "target": b,
                            "relation": "shares_concept",
                            "weight": 1,
                        })

        # Create tag edges
        for tag in all_tags:
            for nid in tag_notes[tag]:
                edges.append({
                    "source": nid,
                    "target": tag,
                    "relation": "has_tag",
                    "weight": 1,
                })

        # Bi-directional link edges
        for note in notes:
            for lid in note.get("links", []):
                edges.append({
                    "source": note["id"],
                    "target": lid,
                    "relation": "links_to",
                    "weight": 2,
                })

        # Derive clusters by concept co-occurrence
        clusters = _cluster_by_cooccurrence(concept_notes)

        # Find orphans (no links or shared concepts)
        orphans: list[str] = []
        for note in notes:
            nid = note["id"]
            has_links = bool(note.get("links"))
            has_concepts = bool(note.get("concepts"))
            is_linked = nid in linked_notes
            if not has_links and not has_concepts and not is_linked:
                orphans.append(nid)

        # Gap analysis: concepts with very few notes
        gaps = sorted(
            [{"concept": c, "note_count": len(ids)} for c, ids in concept_notes.items()],
            key=lambda g: g["note_count"],
        )
        gaps = [g for g in gaps if g["note_count"] <= 1]

        graph = {
            "nodes": nodes,
            "edges": _deduplicate_edges(edges),
            "clusters": clusters,
            "orphans": orphans,
            "gaps": gaps,
            "stats": {
                "total_notes": len(notes),
                "total_concepts": len(all_concepts),
                "total_tags": len(all_tags),
                "total_edges": len(edges),
                "orphan_count": len(orphans),
            },
            "built_at": datetime.utcnow().isoformat(),
        }
        write_json(str(self.graph_path), graph)
        return graph

    def load(self) -> dict | None:
        """Load the persisted graph, or None if not built yet."""
        if not self.graph_path.exists():
            return None
        return read_json(str(self.graph_path))

    def get_related_notes(self, note_id: str, depth: int = 1) -> list[str]:
        """Find notes related to a given note via graph traversal."""
        graph = self.load()
        if not graph:
            return []
        adjacency: dict[str, set[str]] = {}
        for edge in graph["edges"]:
            s, t = edge["source"], edge["target"]
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

        visited: set[str] = {note_id}
        frontier: set[str] = {note_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier |= adjacency.get(n, set())
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
        visited.discard(note_id)
        return list(visited)

    def suggest_connections(self) -> list[dict]:
        """Suggest new connections between orphaned and clustered notes."""
        graph = self.load()
        if not graph:
            return []
        orphan_ids = set(graph.get("orphans", []))
        suggestions: list[dict] = []
        for edge in graph["edges"]:
            src, tgt = edge["source"], edge["target"]
            if src in orphan_ids and tgt not in orphan_ids:
                suggestions.append({"from": src, "to": tgt, "reason": "orphan_connection"})
            elif tgt in orphan_ids and src not in orphan_ids:
                suggestions.append({"from": tgt, "to": src, "reason": "orphan_connection"})
        return suggestions


# ---------------------------------------------------------------------------
# Spaced Repetition (SM-2 inspired)
# ---------------------------------------------------------------------------

class SpacedRepetitionQueue:
    """Spaced repetition review queue using an SM-2 inspired algorithm.

    Each note carries SR metadata:
    {
        "repetitions": int,
        "ease_factor": float,   # starts at 2.5
        "interval": float,      # days until next review
        "next_review": str,     # ISO datetime
        "last_reviewed": str,   # ISO datetime or null
    }

    On review, the user rates recall 0-5:
    - < 3: reset repetitions, interval → 1 day
    - ≥ 3: interval *= ease_factor updated by quality
    """

    SR_QUEUE_PATH = "sr_queue.json"

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.queue_path = self.storage_dir / self.SR_QUEUE_PATH
        ensure_dir(str(self.storage_dir))

    # -- public API ---------------------------------------------------------

    def get_due_notes(self, notes: list[dict]) -> list[dict]:
        """Return notes that are due for review, sorted by next_review ascending."""
        now = datetime.utcnow()
        due: list[dict] = []
        for note in notes:
            sr = note.get("sr_metadata")
            if not sr:
                continue
            next_review = sr.get("next_review")
            if next_review and datetime.fromisoformat(next_review) <= now:
                due.append(note)
        due.sort(key=lambda n: n.get("sr_metadata", {}).get("next_review", ""))
        return due

    def review_note(self, note: dict, quality: int) -> dict:
        """Process a review for a note.

        Args:
            note: The note dict (will be updated in-place).
            quality: Recall quality 0 (complete blackout) to 5 (perfect).

        Returns:
            Updated note dict with new SR metadata.
        """
        sr = note.get("sr_metadata", _init_sr_metadata(datetime.utcnow().isoformat()))
        repetitions = sr.get("repetitions", 0)
        ease_factor = sr.get("ease_factor", 2.5)
        interval = sr.get("interval", 1.0)

        if quality < 3:
            repetitions = 0
            interval = 1.0
        else:
            repetitions += 1
            if repetitions == 1:
                interval = 1.0
            elif repetitions == 2:
                interval = 3.0
            else:
                interval = round(interval * ease_factor, 2)

        # Adjust ease factor (SM-2 formula)
        ease_factor = max(1.3, ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

        now = datetime.utcnow()
        next_review = (now + timedelta(days=interval)).isoformat()

        sr = {
            "repetitions": repetitions,
            "ease_factor": round(ease_factor, 2),
            "interval": interval,
            "next_review": next_review,
            "last_reviewed": now.isoformat(),
        }
        note["sr_metadata"] = sr
        return note

    def enqueue(self, note_id: str) -> dict:
        """Add a note to the review queue. Returns the queue entry."""
        queue = self._load_queue()
        entry = {
            "note_id": note_id,
            "repetitions": 0,
            "ease_factor": 2.5,
            "interval": 1.0,
            "next_review": datetime.utcnow().isoformat(),
            "last_reviewed": None,
        }
        queue[note_id] = entry
        self._save_queue(queue)
        return entry

    def queue_stats(self, notes: list[dict]) -> dict:
        """Return stats about the review queue."""
        now = datetime.utcnow()
        due: list[dict] = []
        upcoming: list[dict] = []
        for note in notes:
            sr = note.get("sr_metadata")
            if not sr:
                continue
            nr = sr.get("next_review")
            if nr:
                if datetime.fromisoformat(nr) <= now:
                    due.append(note)
                else:
                    upcoming.append(note)
        return {
            "due_now": len(due),
            "upcoming": len(upcoming),
            "total_tracked": len(due) + len(upcoming),
        }

    # -- internal -----------------------------------------------------------

    def _load_queue(self) -> dict:
        if self.queue_path.exists():
            return read_json(str(self.queue_path))
        return {}

    def _save_queue(self, queue: dict) -> None:
        write_json(str(self.queue_path), queue)


# ---------------------------------------------------------------------------
# Weekly Review
# ---------------------------------------------------------------------------

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


class WeeklyReviewGenerator:
    """Generate weekly review reports from stored notes and graph data."""

    def __init__(
        self,
        ingestion: NoteIngestionPipeline | None = None,
        graph: KnowledgeGraph | None = None,
        sr_queue: SpacedRepetitionQueue | None = None,
        storage_dir: str = "~/.agentsuite/brain",
    ):
        self.ingestion = ingestion or NoteIngestionPipeline(storage_dir)
        self.graph = graph or KnowledgeGraph(storage_dir)
        self.sr_queue = sr_queue or SpacedRepetitionQueue(storage_dir)

    def generate(self) -> dict:
        """Generate the full weekly review report."""
        notes = self.ingestion.all_notes()
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        # Notes added this week
        recent_notes = [
            n for n in notes
            if n.get("created_at") and datetime.fromisoformat(n["created_at"]) >= week_ago
        ]

        # Top concepts by note count
        concept_counts: dict[str, int] = {}
        for note in notes:
            for concept in note.get("concepts", []):
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Spaced repetition due notes
        due_notes = self.sr_queue.get_due_notes(notes)

        # Pending action items
        pending_actions: list[dict] = []
        for note in notes:
            for item in note.get("action_items", []):
                pending_actions.append({
                    "action": item,
                    "note_id": note["id"],
                    "note_title": note.get("title", "Untitled"),
                })

        # Note of the week: highest connectivity
        note_connectivity = self._compute_connectivity(notes)
        note_of_week = max(note_connectivity, key=lambda x: x["score"]) if note_connectivity else None

        # Queue stats
        stats = self.sr_queue.queue_stats(notes)

        return {
            "period": {
                "from": week_ago.isoformat(),
                "to": now.isoformat(),
            },
            "summary": {
                "total_notes": len(notes),
                "notes_added_this_week": len(recent_notes),
                "notes_due_for_review": len(due_notes),
                "pending_action_items": len(pending_actions),
                "sr_queue": stats,
            },
            "recent_notes": [
                {"id": n["id"], "title": n["title"], "tags": n.get("tags", [])}
                for n in recent_notes
            ],
            "top_concepts": [{"concept": c, "note_count": cnt} for c, cnt in top_concepts],
            "notes_due_for_review": [
                {"id": n["id"], "title": n["title"]} for n in due_notes
            ],
            "pending_action_items": pending_actions,
            "note_of_the_week": note_of_week,
            "generated_at": now.isoformat(),
        }

    # -- internal -----------------------------------------------------------

    def _compute_connectivity(self, notes: list[dict]) -> list[dict]:
        scores: dict[str, int] = {}
        for note in notes:
            score = len(note.get("links", []))
            score += len(note.get("concepts", []))
            scores[note["id"]] = score

        note_map = {n["id"]: n for n in notes}
        return [
            {
                "id": nid,
                "title": note_map.get(nid, {}).get("title", "Unknown"),
                "score": score,
            }
            for nid, score in scores.items()
        ]


# ---------------------------------------------------------------------------
# Workflow orchestrator
# ---------------------------------------------------------------------------

class SecondBrainWorkflow:
    """Second brain knowledge management workflow."""

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = storage_dir
        self.agent = create_second_brain()
        self.ingestion = NoteIngestionPipeline(storage_dir)
        self.graph = KnowledgeGraph(storage_dir)
        self.sr_queue = SpacedRepetitionQueue(storage_dir)
        self.review = WeeklyReviewGenerator(
            self.ingestion, self.graph, self.sr_queue, storage_dir,
        )

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def ingest_note(title: str, content: str, storage_dir: str = "~/.agentsuite/brain", **kwargs) -> dict:
    """Convenience wrapper: ingest a single note."""
    return NoteIngestionPipeline(storage_dir).ingest(title, content, **kwargs)


def build_graph(storage_dir: str = "~/.agentsuite/brain") -> dict:
    """Convenience wrapper: build graph from all stored notes."""
    ingestion = NoteIngestionPipeline(storage_dir)
    graph = KnowledgeGraph(storage_dir)
    return graph.build(ingestion.all_notes())


def get_due_notes(storage_dir: str = "~/.agentsuite/brain") -> list[dict]:
    """Convenience wrapper: get all notes due for SR review."""
    ingestion = NoteIngestionPipeline(storage_dir)
    sr = SpacedRepetitionQueue(storage_dir)
    return sr.get_due_notes(ingestion.all_notes())


def generate_weekly_review(storage_dir: str = "~/.agentsuite/brain") -> dict:
    """Convenience wrapper: generate weekly review report."""
    return WeeklyReviewGenerator(storage_dir=storage_dir).generate()


def _init_sr_metadata(created_at: str) -> dict:
    return {
        "repetitions": 0,
        "ease_factor": 2.5,
        "interval": 1.0,
        "next_review": created_at,
        "last_reviewed": None,
    }


def _summarize(content: str) -> str:
    """Create a simple extractive summary (first ~3 sentences)."""
    sentences = content.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:3]).strip()
    if not summary.endswith("."):
        summary += "."
    return summary


def _cluster_by_cooccurrence(concept_notes: dict[str, list[str]]) -> list[dict]:
    """Group concepts that share notes into clusters."""
    if not concept_notes:
        return []
    concept_list = list(concept_notes.keys())
    clusters: list[dict] = []
    seen: set[str] = set()
    for i, c1 in enumerate(concept_list):
        if c1 in seen:
            continue
        cluster_nodes: set[str] = {c1}
        notes1 = set(concept_notes[c1])
        for c2 in concept_list[i + 1:]:
            notes2 = set(concept_notes[c2])
            if notes1 & notes2:
                cluster_nodes.add(c2)
        for n in cluster_nodes:
            seen.add(n)
        shared: set[str] = set()
        for n in cluster_nodes:
            shared.update(concept_notes[n])
        density = len(shared) / max(len(cluster_nodes), 1)
        clusters.append({
            "concepts": sorted(cluster_nodes),
            "shared_notes": sorted(shared),
            "density": round(density, 2),
        })
    return clusters


def _deduplicate_edges(edges: list[dict]) -> list[dict]:
    """Remove duplicate edges (same source, target, relation)."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
