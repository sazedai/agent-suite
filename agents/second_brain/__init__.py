"""Second Brain Systems.

Knowledge base management, note linking, spaced repetition,
and context retrieval for personal knowledge management.

This module provides both programmatic systems (NoteStore, KnowledgeGraph,
SpacedRepetition, WeeklyReview) and CrewAI agent/task/workflow wrappers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import write_json, read_json, ensure_dir


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Note:
    """A single note in the second brain."""

    def __init__(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        concepts: list[str] | None = None,
        links: list[str] | None = None,
        actions: list[str] | None = None,
        source: str = "",
    ):
        self.id = hashlib.sha256(f"{title}{content}".encode()).hexdigest()[:12]
        self.title = title
        self.content = content
        self.summary = ""
        self.tags = tags or []
        self.concepts = concepts or []
        self.links = links or []          # IDs of related notes
        self.actions = actions or []
        self.source = source
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        # Spaced repetition fields
        self.sr_interval = 0              # days until next review
        self.sr_easiness = 2.5            # SM-2 easiness factor
        self.sr_repetitions = 0           # successful review count
        self.sr_next_review: str | None = None  # ISO date

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "tags": self.tags,
            "concepts": self.concepts,
            "links": self.links,
            "actions": self.actions,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sr_interval": self.sr_interval,
            "sr_easiness": self.sr_easiness,
            "sr_repetitions": self.sr_repetitions,
            "sr_next_review": self.sr_next_review,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        note = cls(
            title=d["title"],
            content=d["content"],
            tags=d.get("tags", []),
            concepts=d.get("concepts", []),
            links=d.get("links", []),
            actions=d.get("actions", []),
            source=d.get("source", ""),
        )
        note.id = d.get("id", note.id)
        note.summary = d.get("summary", "")
        note.created_at = d.get("created_at", note.created_at)
        note.updated_at = d.get("updated_at", note.updated_at)
        note.sr_interval = d.get("sr_interval", 0)
        note.sr_easiness = d.get("sr_easiness", 2.5)
        note.sr_repetitions = d.get("sr_repetitions", 0)
        note.sr_next_review = d.get("sr_next_review")
        return note


# ---------------------------------------------------------------------------
# Note Store — persistence layer
# ---------------------------------------------------------------------------

class NoteStore:
    """File-backed storage for notes."""

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.notes_dir = self.notes_dir_path()
        ensure_dir(str(self.notes_dir))
        self.index_path = self.storage_dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def notes_dir_path(self) -> Path:
        return self.storage_dir / "notes"

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            try:
                data = read_json(str(self.index_path))
                return data.get("notes", {})
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        write_json(str(self.index_path), {"notes": self._index})

    def _note_path(self, note_id: str) -> Path:
        return self.notes_dir / f"{note_id}.json"

    def save(self, note: Note) -> None:
        """Persist a note and update the index."""
        note.updated_at = datetime.utcnow().isoformat()
        write_json(str(self._note_path(note.id)), note.to_dict())
        self._index[note.id] = {
            "title": note.title,
            "tags": note.tags,
            "concepts": note.concepts,
            "created_at": note.created_at,
            "sr_next_review": note.sr_next_review,
        }
        self._save_index()

    def load(self, note_id: str) -> Note | None:
        """Load a note by ID."""
        path = self._note_path(note_id)
        if not path.exists():
            return None
        try:
            data = read_json(str(path))
            return Note.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    def load_all(self) -> list[Note]:
        """Return every note in the store."""
        notes = []
        for note_id in list(self._index.keys()):
            note = self.load(note_id)
            if note:
                notes.append(note)
        return notes

    def delete(self, note_id: str) -> bool:
        """Remove a note. Returns True if it existed."""
        path = self._note_path(note_id)
        if path.exists():
            path.unlink()
        if note_id in self._index:
            del self._index[note_id]
            self._save_index()
            return True
        return False

    def search(self, query: str) -> list[Note]:
        """Search notes whose title, content, tags, or concepts contain the query (case-insensitive)."""
        q = query.lower()
        results: list[Note] = []
        for note_id, meta in self._index.items():
            if (
                q in meta.get("title", "").lower()
                or any(q in t.lower() for t in meta.get("tags", []))
                or any(q in c.lower() for c in meta.get("concepts", []))
            ):
                note = self.load(note_id)
                if note:
                    results.append(note)
        # Also search content for candidates matched by metadata
        for note_id in self._index:
            if note_id in {n.id for n in results}:
                continue
            note = self.load(note_id)
            if note and q in note.content.lower():
                results.append(note)
        return results

    @property
    def count(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Note Ingestion — parsing, extraction, storage
# ---------------------------------------------------------------------------

class NoteIngestion:
    """Parse raw content into structured notes and persist them."""

    def __init__(self, store: NoteStore):
        self.store = store
        self.stats = {"added": 0, "updated": 0}

    def ingest(self, content: str, title: str = "", source: str = "", tags: list[str] | None = None) -> Note:
        """Ingest raw text into a new Note."""
        parsed_title = title or self._extract_title(content)
        note = Note(
            title=parsed_title,
            content=content,
            tags=tags or [],
            source=source,
        )
        # Extract structured fields
        note.concepts = self._extract_concepts(content)
        note.tags = list(set(note.tags + self._extract_tags(content)))
        note.summary = self._generate_summary(content)
        note.actions = self._extract_actions(content)
        # Link to existing notes by concept overlap
        note.links = self._find_links(note)
        self.store.save(note)
        self.stats["added"] += 1
        return note

    def ingest_batch(self, items: list[dict[str, Any]]) -> list[Note]:
        """Ingest multiple notes. Each item can have keys: content, title, source, tags."""
        return [self.ingest(**item) for item in items]

    # -- internal helpers --

    @staticmethod
    def _extract_title(content: str) -> str:
        for line in content.strip().splitlines():
            stripped = line.strip()
            if stripped:
                # Markdown heading
                if stripped.startswith("#"):
                    return stripped.lstrip("#").strip()[:80]
                return stripped[:80]
        return "Untitled Note"

    @staticmethod
    def _extract_concepts(content: str) -> list[str]:
        """Extract capitalized multi-word phrases and #hashtags as concepts."""
        concepts: set[str] = set()
        # Hashtags
        for tag in re.findall(r"#(\w+)", content):
            concepts.add(tag)
        # Capitalized phrases (2-3 words)
        for match in re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", content
        ):
            concepts.add(match)
        return sorted(concepts)[:20]

    @staticmethod
    def _extract_tags(content: str) -> list[str]:
        """Derive lowercase keyword tags from the text."""
        tags: set[str] = set()
        for tag in re.findall(r"#(\w+)", content):
            tags.add(tag.lower())
        return sorted(tags)[:10]

    @staticmethod
    def _generate_summary(content: str) -> str:
        """Take the first non-empty sentence as a summary."""
        sentences = re.split(r"[.!?\n]+", content.strip())
        for s in sentences:
            s = s.strip()
            if len(s) > 15:
                return s[:200]
        return content[:200].strip()

    @staticmethod
    def _extract_actions(content: str) -> list[str]:
        """Find lines that look like action items."""
        actions: list[str] = []
        # Checklist items
        for m in re.findall(r"[-*]\s+\[([ xX])\]\s+(.+)", content):
            status, text = m
            if status == " ":
                actions.append(text.strip())
        # TODO / ACTION lines
        for m in re.findall(r"(?:TODO|ACTION|DO):\s*(.+)", content, re.IGNORECASE):
            actions.append(m.strip())
        return actions[:10]

    def _find_links(self, note: Note) -> list[str]:
        """Link to existing notes that share concepts."""
        linked: set[str] = set()
        existing = self.store.load_all()
        concept_set = set(n.lower() for n in note.concepts)
        for other in existing:
            other_concepts = set(c.lower() for c in other.concepts)
            if concept_set & other_concepts:
                linked.add(other.id)
                # Bidirectional: add this note's ID to the other if not present
                if note.id not in other.links:
                    other.links.append(note.id)
                    self.store.save(other)
        return sorted(linked)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """In-memory graph built from notes: nodes = notes + concepts, edges = co-occurrence."""

    def __init__(self, store: NoteStore):
        self.store = store
        self.nodes: dict[str, dict] = {}    # id -> {type, label, ...}
        self.edges: list[dict] = []         # [{source, target, weight, type}]
        self._built = False

    def build(self) -> "KnowledgeGraph":
        """Rebuild the graph from all notes in the store."""
        notes = self.store.load_all()
        self.nodes.clear()
        self.edges.clear()

        # Add note nodes
        for note in notes:
            self.nodes[f"note:{note.id}"] = {
                "type": "note",
                "label": note.title,
                "tags": note.tags,
                "concepts": note.concepts,
            }

        # Add concept nodes and note-concept edges
        concept_notes: dict[str, list[str]] = defaultdict(list)
        for note in notes:
            for concept in note.concepts:
                cid = f"concept:{concept.lower()}"
                concept_notes[concept.lower()].append(note.id)
                if cid not in self.nodes:
                    self.nodes[cid] = {"type": "concept", "label": concept}
                self.edges.append({
                    "source": f"note:{note.id}",
                    "target": cid,
                    "weight": 1,
                    "type": "has_concept",
                })

        # Note-note edges via concept co-occurrence
        for concept, note_ids in concept_notes.items():
            for i, a in enumerate(note_ids):
                for b in note_ids[i + 1:]:
                    edge = self._find_or_create_edge(f"note:{a}", f"note:{b}", "related")
                    edge["weight"] += 1

        # Explicit link edges
        for note in notes:
            for linked_id in note.links:
                edge = self._find_or_create_edge(
                    f"note:{note.id}", f"note:{linked_id}", "links_to"
                )
                edge["weight"] += 2

        self._built = True
        return self

    def _find_or_create_edge(self, source: str, target: str, edge_type: str) -> dict:
        for e in self.edges:
            if e["source"] == source and e["target"] == target and e["type"] == edge_type:
                return e
        edge = {"source": source, "target": target, "weight": 0, "type": edge_type}
        self.edges.append(edge)
        return edge

    def clusters(self) -> list[dict]:
        """Group notes into clusters by shared concepts (simple connected-components)."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in self.edges:
            if e["type"] in ("related", "links_to"):
                union(e["source"], e["target"])

        groups: dict[str, list[str]] = defaultdict(list)
        for node_id in self.nodes:
            if node_id.startswith("note:"):
                groups[find(node_id)].append(node_id)

        result = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            labels = [self.nodes[m]["label"] for m in members if m in self.nodes]
            shared_concepts: Counter[str] = Counter()
            for m in members:
                nid = m.split(":", 1)[1]
                note = self.store.load(nid)
                if note:
                    shared_concepts.update(c.lower() for c in note.concepts)
            result.append({
                "members": labels,
                "size": len(members),
                "top_concepts": [c for c, _ in shared_concepts.most_common(5)],
            })
        return sorted(result, key=lambda c: c["size"], reverse=True)

    def orphans(self) -> list[str]:
        """Return titles of notes with no links."""
        linked_ids: set[str] = set()
        for e in self.edges:
            if e["type"] in ("related", "links_to"):
                linked_ids.add(e["source"])
                linked_ids.add(e["target"])
        orphans = []
        for nid, node in self.nodes.items():
            if node["type"] == "note" and nid not in linked_ids:
                orphans.append(node["label"])
        return orphans

    def gaps(self) -> list[dict]:
        """Identify concept pairs that co-occur but aren't directly linked."""
        gaps = []
        concept_notes: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            if e["type"] == "has_concept":
                concept_notes[e["target"]].add(e["source"])
        concepts = list(concept_notes.keys())
        for i, ca in enumerate(concepts):
            for cb in concepts[i + 1:]:
                shared = concept_notes[ca] & concept_notes[cb]
                if len(shared) >= 2:
                    gaps.append({
                        "concepts": [
                            self.nodes.get(ca, {}).get("label", ca),
                            self.nodes.get(cb, {}).get("label", cb),
                        ],
                        "shared_notes": len(shared),
                    })
        return sorted(gaps, key=lambda g: g["shared_notes"], reverse=True)[:10]

    def top_concepts(self, n: int = 10) -> list[dict]:
        """Return the most connected concepts."""
        concept_degree: Counter[str] = Counter()
        for e in self.edges:
            if e["type"] == "has_concept":
                concept_degree[e["target"]] += 1
        return [
            {"concept": self.nodes.get(cid, {}).get("label", cid), "connections": count}
            for cid, count in concept_degree.most_common(n)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "clusters": self.clusters(),
            "orphans": self.orphans(),
            "gaps": self.gaps(),
            "top_concepts": self.top_concepts(),
        }


# ---------------------------------------------------------------------------
# Spaced Repetition (SM-2 inspired)
# ---------------------------------------------------------------------------

class SpacedRepetition:
    """SM-2 based spaced repetition scheduler for notes."""

    def __init__(self, store: NoteStore):
        self.store = store

    def review(self, note_id: str, quality: int) -> Note:
        """Process a review for a note.

        Args:
            note_id: The note to review.
            quality: 0-5 recall quality (0=complete blackout, 5=perfect).

        Returns:
            The updated Note.
        """
        note = self.store.load(note_id)
        if note is None:
            raise ValueError(f"Note {note_id} not found")

        if quality < 0 or quality > 5:
            raise ValueError("Quality must be between 0 and 5")

        # SM-2 algorithm
        if quality < 3:
            note.sr_repetitions = 0
            note.sr_interval = 1
        else:
            if note.sr_repetitions == 0:
                note.sr_interval = 1
            elif note.sr_repetitions == 1:
                note.sr_interval = 6
            else:
                note.sr_interval = round(note.sr_interval * note.sr_easiness)
            note.sr_repetitions += 1

        # Update easiness factor
        note.sr_easiness = max(
            1.3,
            note.sr_easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
        )

        # Schedule next review
        next_date = datetime.utcnow() + timedelta(days=note.sr_interval)
        note.sr_next_review = next_date.isoformat()
        note.updated_at = datetime.utcnow().isoformat()

        self.store.save(note)
        return note

    def due_notes(self) -> list[Note]:
        """Return all notes that are due for review."""
        now = datetime.utcnow().isoformat()
        due = []
        for note in self.store.load_all():
            if note.sr_next_review is None:
                # Never reviewed — consider it due
                due.append(note)
            elif note.sr_next_review <= now:
                due.append(note)
        return due

    def schedule(self) -> list[dict]:
        """Return the full review schedule."""
        schedule = []
        for note in self.store.load_all():
            schedule.append({
                "id": note.id,
                "title": note.title,
                "interval": note.sr_interval,
                "easiness": round(note.sr_easiness, 2),
                "repetitions": note.sr_repetitions,
                "next_review": note.sr_next_review,
            })
        return sorted(schedule, key=lambda s: s["next_review"] or "")

    def stats(self) -> dict[str, Any]:
        """Return aggregate spaced repetition statistics."""
        notes = self.store.load_all()
        reviewed = [n for n in notes if n.sr_repetitions > 0]
        due = self.due_notes()
        return {
            "total_notes": len(notes),
            "reviewed_notes": len(reviewed),
            "due_now": len(due),
            "avg_easiness": round(
                sum(n.sr_easiness for n in reviewed) / len(reviewed), 2
            ) if reviewed else 0,
            "avg_interval": round(
                sum(n.sr_interval for n in reviewed) / len(reviewed), 1
            ) if reviewed else 0,
        }


# ---------------------------------------------------------------------------
# Weekly Review Generator
# ---------------------------------------------------------------------------

class WeeklyReview:
    """Generate a weekly review report from the second brain."""

    def __init__(self, store: NoteStore, graph: KnowledgeGraph, sr: SpacedRepetition):
        self.store = store
        self.graph = graph
        self.sr = sr

    def generate(self) -> dict[str, Any]:
        """Build the weekly review report."""
        now = datetime.utcnow()
        week_ago = (now - timedelta(days=7)).isoformat()

        all_notes = self.store.load_all()
        this_week = [n for n in all_notes if n.created_at >= week_ago]

        # Rebuild graph for fresh analysis
        self.graph.build()

        # Top concepts
        top_concepts = self.graph.top_concepts(5)

        # Due for review
        due = self.sr.due_notes()

        # Open action items
        open_actions: list[dict] = []
        for note in all_notes:
            for action in note.actions:
                open_actions.append({"note": note.title, "action": action})

        # Note of the week: most connected note this week
        note_of_week = None
        if this_week:
            best = max(this_week, key=lambda n: len(n.links))
            note_of_week = {
                "title": best.title,
                "summary": best.summary,
                "links": len(best.links),
                "concepts": best.concepts[:5],
            }

        return {
            "period": {
                "from": week_ago,
                "to": now.isoformat(),
            },
            "notes_added": len(this_week),
            "note_titles": [n.title for n in this_week],
            "top_concepts": top_concepts,
            "revisitable_count": len(due),
            "revisitable": [{"title": n.title, "id": n.id} for n in due[:10]],
            "open_actions": open_actions[:20],
            "note_of_the_week": note_of_week,
            "clusters": self.graph.clusters()[:5],
            "orphans": self.graph.orphans()[:10],
            "sr_stats": self.sr.stats(),
        }

    def format_report(self) -> str:
        """Return a human-readable weekly review."""
        r = self.generate()
        lines = [
            "# 🧠 Second Brain Weekly Review",
            f"**Period:** {r['period']['from'][:10]} → {r['period']['to'][:10]}",
            "",
            f"## 📝 Notes Added: {r['notes_added']}",
        ]
        for title in r["note_titles"]:
            lines.append(f"  - {title}")

        lines += [
            "",
            "## 🔗 Top Concepts",
        ]
        for c in r["top_concepts"]:
            lines.append(f"  - **{c['concept']}** ({c['connections']} connections)")

        lines += [
            "",
            f"## 🔄 Due for Review: {r['revisitable_count']}",
        ]
        for item in r["revisitable"]:
            lines.append(f"  - {item['title']}")

        lines += [
            "",
            "## ✅ Open Action Items",
        ]
        for a in r["open_actions"]:
            lines.append(f"  - [{a['note']}] {a['action']}")

        if r["note_of_the_week"]:
            nw = r["note_of_the_week"]
            lines += [
                "",
                f"## ⭐ Note of the Week: {nw['title']}",
                f"  {nw['summary']}",
                f"  Links: {nw['links']} | Concepts: {', '.join(nw['concepts'])}",
            ]

        if r["orphans"]:
            lines += [
                "",
                "## 🏝️ Orphan Notes (no links)",
            ]
            for o in r["orphans"]:
                lines.append(f"  - {o}")

        lines += [
            "",
            "## 📊 Spaced Repetition Stats",
            f"  Total notes: {r['sr_stats']['total_notes']}",
            f"  Reviewed: {r['sr_stats']['reviewed_notes']}",
            f"  Due now: {r['sr_stats']['due_now']}",
            f"  Avg easiness: {r['sr_stats']['avg_easiness']}",
            f"  Avg interval: {r['sr_stats']['avg_interval']} days",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CrewAI Agent / Task / Workflow wrappers
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
    """Second brain knowledge management workflow.

    Provides both programmatic API (via NoteStore, KnowledgeGraph,
    SpacedRepetition, WeeklyReview) and CrewAI LLM-powered workflow methods.
    """

    def __init__(self, storage_dir: str = "~/.agentsuite/brain"):
        self.storage_dir = storage_dir
        self._agent: Agent | None = None
        self.store = NoteStore(storage_dir)
        self.ingestion = NoteIngestion(self.store)
        self.graph = KnowledgeGraph(self.store)
        self.sr = SpacedRepetition(self.store)
        self.reviewer = WeeklyReview(self.store, self.graph, self.sr)

    @property
    def agent(self) -> Agent:
        """Lazily create the CrewAI agent (requires valid LLM credentials)."""
        if self._agent is None:
            self._agent = create_second_brain()
        return self._agent

    # -- Programmatic API --

    def ingest_note(self, content: str, title: str = "", source: str = "", tags: list[str] | None = None) -> Note:
        """Ingest a note programmatically (no LLM call)."""
        return self.ingestion.ingest(content, title=title, source=source, tags=tags)

    def ingest_batch(self, items: list[dict[str, Any]]) -> list[Note]:
        """Ingest multiple notes programmatically."""
        return self.ingestion.ingest_batch(items)

    def build_graph(self) -> KnowledgeGraph:
        """Build and return the knowledge graph from stored notes."""
        return self.graph.build()

    def review_note(self, note_id: str, quality: int) -> Note:
        """Process a spaced repetition review for a note."""
        return self.sr.review(note_id, quality)

    def due_notes(self) -> list[Note]:
        """Return notes due for review."""
        return self.sr.due_notes()

    def weekly_review(self) -> str:
        """Generate a formatted weekly review report."""
        return self.reviewer.format_report()

    def weekly_review_data(self) -> dict[str, Any]:
        """Generate the weekly review as structured data."""
        return self.reviewer.generate()

    def search(self, query: str) -> list[Note]:
        """Search notes by query string."""
        return self.store.search(query)

    # -- LLM-powered workflow methods (CrewAI) --

    def llm_ingest_note(self, content: str) -> str:
        """Ingest a note using the LLM agent."""
        task = create_note_ingestion_task(content, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def llm_build_graph(self, notes: list[dict]) -> str:
        """Build knowledge graph using the LLM agent."""
        task = create_knowledge_graph_task(notes, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def llm_weekly_review(self) -> str:
        """Generate weekly review using the LLM agent."""
        task = create_review_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
