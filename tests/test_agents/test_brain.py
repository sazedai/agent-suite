"""Tests for Second Brain agent systems."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.second_brain import (
    NoteIngestionPipeline,
    KnowledgeGraph,
    SpacedRepetitionQueue,
    WeeklyReviewGenerator,
    SecondBrainWorkflow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_brain_dir():
    """Provide a temporary storage directory for all brain components."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_notes():
    """Return a list of sample note dicts for testing."""
    return [
        {
            "id": "note-1",
            "title": "Python Design Patterns",
            "content": "Python supports multiple design patterns. The Factory pattern creates objects without specifying exact classes. The Observer pattern lets objects subscribe to events.",
            "summary": "Python supports multiple design patterns.",
            "tags": ["python", "design-patterns", "oop"],
            "concepts": ["factory pattern", "observer pattern", "polymorphism"],
            "links": ["note-2"],
            "action_items": ["Implement factory example in project"],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "sr_metadata": {
                "repetitions": 0,
                "ease_factor": 2.5,
                "interval": 1.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        },
        {
            "id": "note-2",
            "title": "Event-Driven Architecture",
            "content": "Event-driven systems use events to trigger decoupled services. This pattern aligns well with message queues like RabbitMQ and Kafka.",
            "summary": "Event-driven systems use events to trigger decoupled services.",
            "tags": ["architecture", "events", "messaging"],
            "concepts": ["observer pattern", "message queues", "decoupling"],
            "links": ["note-1"],
            "action_items": [
                "Evaluate Kafka vs RabbitMQ for new service",
                "Set up event bus prototype",
            ],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "sr_metadata": {
                "repetitions": 2,
                "ease_factor": 2.3,
                "interval": 6.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        },
        {
            "id": "note-3",
            "title": "Team Meeting Notes",
            "content": "Discussed the Q3 roadmap. Key decisions: migrate to microservices, adopt Kubernetes. Action items assigned to all team leads.",
            "summary": "Discussed the Q3 roadmap.",
            "tags": ["meetings", "planning"],
            "concepts": [],
            "links": [],
            "action_items": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "sr_metadata": {
                "repetitions": 0,
                "ease_factor": 2.5,
                "interval": 1.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        },
    ]


@pytest.fixture
def populated_storage(tmp_brain_dir, sample_notes):
    """Write sample notes into a temporary brain directory."""
    notes_dir = Path(tmp_brain_dir) / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    for note in sample_notes:
        path = notes_dir / f"{note['id']}.json"
        with open(path, "w") as f:
            json.dump(note, f, indent=2, default=str)
    index = {
        n["id"]: {
            "title": n["title"],
            "tags": n["tags"],
            "concepts": n["concepts"],
            "created_at": n["created_at"],
        }
        for n in sample_notes
    }
    idx_path = Path(tmp_brain_dir) / "index.json"
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    return tmp_brain_dir


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

def test_second_brain_module_imports():
    """All second brain components should be importable."""
    from agents.second_brain import (
        create_second_brain,
        create_note_ingestion_task,
        create_knowledge_graph_task,
        create_review_task,
        NoteIngestionPipeline,
        KnowledgeGraph,
        SpacedRepetitionQueue,
        WeeklyReviewGenerator,
        SecondBrainWorkflow,
        ingest_note,
        build_graph,
        get_due_notes,
        generate_weekly_review,
    )
    assert callable(create_second_brain)
    assert callable(create_note_ingestion_task)
    assert callable(create_knowledge_graph_task)
    assert callable(create_review_task)
    assert NoteIngestionPipeline is not None
    assert KnowledgeGraph is not None
    assert SpacedRepetitionQueue is not None
    assert WeeklyReviewGenerator is not None
    assert SecondBrainWorkflow is not None


# ---------------------------------------------------------------------------
# Note Ingestion Pipeline
# ---------------------------------------------------------------------------

class TestNoteIngestionPipeline:

    def test_ingest_creates_note_with_required_fields(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(
            title="Test Note",
            content="This is a test note. It has multiple sentences. Enough for a summary.",
        )
        assert note["id"]
        assert note["title"] == "Test Note"
        assert note["content"]
        assert note["summary"]
        assert note["created_at"]
        assert note["updated_at"]
        assert "sr_metadata" in note

    def test_ingest_with_explicit_tags_and_concepts(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(
            title="ML Basics",
            content="Machine learning is a subset of AI. Deep learning uses neural networks.",
            tags=["ml", "ai"],
            concepts=["machine learning", "deep learning"],
            links=["other-note-id"],
            action_items=["Read Hinton paper"],
        )
        assert note["tags"] == ["ml", "ai"]
        assert note["concepts"] == ["machine learning", "deep learning"]
        assert note["links"] == ["other-note-id"]
        assert note["action_items"] == ["Read Hinton paper"]

    def test_ingest_persists_to_disk(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="Persistent", content="This note should be saved.")
        path = Path(tmp_brain_dir) / "notes" / f"{note['id']}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["title"] == "Persistent"

    def test_ingest_creates_and_updates_index(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="Indexed", content="This note goes into the index.")
        idx_path = Path(tmp_brain_dir) / "index.json"
        assert idx_path.exists()
        index = json.loads(idx_path.read_text())
        assert note["id"] in index
        assert index[note["id"]]["title"] == "Indexed"

    def test_bulk_ingest(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        notes_data = [
            {"title": "Note A", "content": "Content A."},
            {"title": "Note B", "content": "Content B."},
            {"title": "Note C", "content": "Content C."},
        ]
        results = pipeline.bulk_ingest(notes_data)
        assert len(results) == 3
        assert all(r["id"] for r in results)

    def test_get_note(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="Retrievable", content="Find me.")
        fetched = pipeline.get_note(note["id"])
        assert fetched is not None
        assert fetched["title"] == "Retrievable"

    def test_get_note_missing_returns_none(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        assert pipeline.get_note("nonexistent-id") is None

    def test_all_notes(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="One", content="First.")
        pipeline.ingest(title="Two", content="Second.")
        all_notes = pipeline.all_notes()
        assert len(all_notes) == 2

    def test_search_by_title(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="Python Guide", content="Learn Python.")
        pipeline.ingest(title="Rust Guide", content="Learn Rust.")
        results = pipeline.search("Python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Guide"

    def test_search_by_content(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="A", content="Kubernetes orchestrates containers.")
        pipeline.ingest(title="B", content="Docker builds containers.")
        results = pipeline.search("Kubernetes")
        assert len(results) == 1

    def test_search_by_tag(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="Tagged", content="Has tags.", tags=["kubernetes", "devops"])
        results = pipeline.search("devops")
        assert len(results) == 1

    def test_search_no_results(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="Something", content="Content.")
        results = pipeline.search("nonexistent")
        assert results == []

    def test_sr_metadata_initialized(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="SR Test", content="Check SR metadata.")
        sr = note["sr_metadata"]
        assert sr["repetitions"] == 0
        assert sr["ease_factor"] == 2.5
        assert sr["interval"] == 1.0
        assert sr["next_review"] is not None
        assert sr["last_reviewed"] is None


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:

    def test_build_creates_graph_structure(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        assert "nodes" in result
        assert "edges" in result
        assert "clusters" in result
        assert "orphans" in result
        assert "gaps" in result
        assert "stats" in result

    def test_build_creates_note_nodes(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        note_nodes = [n for n in result["nodes"] if n["type"] == "note"]
        assert len(note_nodes) == 3

    def test_build_creates_concept_nodes(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        concept_nodes = [n for n in result["nodes"] if n["type"] == "concept"]
        assert len(concept_nodes) > 0

    def test_build_detects_orphans(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        # note-3 has no concepts, no links, and is not linked to
        assert "note-3" in result["orphans"]

    def test_build_detects_clusters(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        # observer pattern is shared between note-1 and note-2
        assert len(result["clusters"]) > 0

    def test_build_stats(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build(sample_notes)
        stats = result["stats"]
        assert stats["total_notes"] == 3
        assert stats["orphan_count"] >= 1

    def test_build_persists_graph(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        graph.build(sample_notes)
        loaded = graph.load()
        assert loaded is not None
        assert loaded["stats"]["total_notes"] == 3

    def test_load_returns_none_when_not_built(self, tmp_brain_dir):
        graph = KnowledgeGraph(tmp_brain_dir)
        assert graph.load() is None

    def test_get_related_notes(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        graph.build(sample_notes)
        related = graph.get_related_notes("note-1", depth=1)
        assert "note-2" in related

    def test_get_related_notes_depth_2(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        graph.build(sample_notes)
        related = graph.get_related_notes("note-1", depth=2)
        assert isinstance(related, list)

    def test_suggest_connections(self, tmp_brain_dir, sample_notes):
        graph = KnowledgeGraph(tmp_brain_dir)
        graph.build(sample_notes)
        suggestions = graph.suggest_connections()
        assert isinstance(suggestions, list)

    def test_build_empty_notes(self, tmp_brain_dir):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build([])
        assert result["stats"]["total_notes"] == 0
        assert result["orphans"] == []


# ---------------------------------------------------------------------------
# Spaced Repetition Queue
# ---------------------------------------------------------------------------

class TestSpacedRepetitionQueue:

    def test_get_due_notes(self, tmp_brain_dir, sample_notes):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        due = sr.get_due_notes(sample_notes)
        # All notes have next_review set to now, so all should be due
        assert len(due) == 3

    def test_get_due_notes_empty_when_future(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        future = (datetime.utcnow() + timedelta(days=30)).isoformat()
        notes = [
            {
                "id": "n1",
                "title": "Future Note",
                "sr_metadata": {
                    "repetitions": 0,
                    "ease_factor": 2.5,
                    "interval": 30.0,
                    "next_review": future,
                    "last_reviewed": None,
                },
            }
        ]
        due = sr.get_due_notes(notes)
        assert len(due) == 0

    def test_review_note_quality_high(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        note = {
            "id": "n1",
            "title": "Review Test",
            "sr_metadata": {
                "repetitions": 0,
                "ease_factor": 2.5,
                "interval": 1.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        }
        updated = sr.review_note(note, quality=5)
        assert updated["sr_metadata"]["repetitions"] == 1
        assert updated["sr_metadata"]["last_reviewed"] is not None

    def test_review_note_quality_low_resets(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        note = {
            "id": "n1",
            "title": "Reset Test",
            "sr_metadata": {
                "repetitions": 3,
                "ease_factor": 2.5,
                "interval": 10.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        }
        updated = sr.review_note(note, quality=1)
        assert updated["sr_metadata"]["repetitions"] == 0
        assert updated["sr_metadata"]["interval"] == 1.0

    def test_review_note_interval_increases(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        note = {
            "id": "n1",
            "title": "Interval Test",
            "sr_metadata": {
                "repetitions": 2,
                "ease_factor": 2.5,
                "interval": 3.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        }
        updated = sr.review_note(note, quality=4)
        assert updated["sr_metadata"]["repetitions"] == 3
        assert updated["sr_metadata"]["interval"] > 3.0

    def test_review_note_ease_factor_minimum(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        note = {
            "id": "n1",
            "title": "EF Test",
            "sr_metadata": {
                "repetitions": 0,
                "ease_factor": 1.3,
                "interval": 1.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        }
        # Repeated low quality should not push EF below 1.3
        for _ in range(10):
            note = sr.review_note(note, quality=0)
        assert note["sr_metadata"]["ease_factor"] >= 1.3

    def test_enqueue(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        entry = sr.enqueue("note-123")
        assert entry["note_id"] == "note-123"
        assert entry["repetitions"] == 0

    def test_queue_stats(self, tmp_brain_dir, sample_notes):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        stats = sr.queue_stats(sample_notes)
        assert stats["due_now"] == 3
        assert stats["total_tracked"] == 3

    def test_queue_stats_empty(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        stats = sr.queue_stats([])
        assert stats["due_now"] == 0
        assert stats["total_tracked"] == 0


# ---------------------------------------------------------------------------
# Weekly Review Generator
# ---------------------------------------------------------------------------

class TestWeeklyReviewGenerator:

    def test_generate_returns_full_report(self, tmp_brain_dir, sample_notes):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        # Pre-populate storage
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = gen.generate()
        assert "period" in report
        assert "summary" in report
        assert "recent_notes" in report
        assert "top_concepts" in report
        assert "notes_due_for_review" in report
        assert "pending_action_items" in report
        assert "note_of_the_week" in report
        assert "generated_at" in report

    def test_generate_summary_counts(self, tmp_brain_dir, sample_notes):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = gen.generate()
        summary = report["summary"]
        assert summary["total_notes"] == 3
        assert summary["notes_added_this_week"] == 3

    def test_generate_top_concepts(self, tmp_brain_dir, sample_notes):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = gen.generate()
        top = report["top_concepts"]
        assert len(top) > 0
        # observer pattern appears in 2 notes
        observer = [c for c in top if c["concept"] == "observer pattern"]
        assert len(observer) == 1
        assert observer[0]["note_count"] == 2

    def test_generate_pending_action_items(self, tmp_brain_dir, sample_notes):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = gen.generate()
        actions = report["pending_action_items"]
        # note-1 has 1 action, note-2 has 2 actions, note-3 has 0
        assert len(actions) == 3

    def test_generate_note_of_the_week(self, tmp_brain_dir, sample_notes):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = gen.generate()
        notw = report["note_of_the_week"]
        assert notw is not None
        assert "id" in notw
        assert "score" in notw

    def test_generate_empty_brain(self, tmp_brain_dir):
        gen = WeeklyReviewGenerator(storage_dir=tmp_brain_dir)
        report = gen.generate()
        assert report["summary"]["total_notes"] == 0
        assert report["recent_notes"] == []
        assert report["top_concepts"] == []


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:

    def test_ingest_note_function(self, tmp_brain_dir):
        from agents.second_brain import ingest_note
        note = ingest_note(
            title="Quick Note",
            content="Quick content.",
            storage_dir=tmp_brain_dir,
        )
        assert note["title"] == "Quick Note"
        assert note["id"]

    def test_build_graph_function(self, tmp_brain_dir, sample_notes):
        from agents.second_brain import build_graph
        # Pre-populate
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        graph = build_graph(tmp_brain_dir)
        assert graph["stats"]["total_notes"] == 3

    def test_get_due_notes_function(self, tmp_brain_dir, sample_notes):
        from agents.second_brain import get_due_notes
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        due = get_due_notes(tmp_brain_dir)
        assert len(due) == 3

    def test_generate_weekly_review_function(self, tmp_brain_dir, sample_notes):
        from agents.second_brain import generate_weekly_review
        notes_dir = Path(tmp_brain_dir) / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for note in sample_notes:
            path = notes_dir / f"{note['id']}.json"
            with open(path, "w") as f:
                json.dump(note, f, indent=2, default=str)

        report = generate_weekly_review(tmp_brain_dir)
        assert report["summary"]["total_notes"] == 3


# ---------------------------------------------------------------------------
# Agent creation (mock get_llm to avoid API key issues)
# ---------------------------------------------------------------------------

class TestAgentCreation:

    def test_create_second_brain(self):
        with patch("agents.second_brain.get_llm", return_value="fake-model"):
            from agents.second_brain import create_second_brain
            agent = create_second_brain()
            assert agent.role == "Knowledge Management Specialist"

    def test_create_note_ingestion_task(self):
        with patch("agents.second_brain.get_llm", return_value="fake-model"):
            from agents.second_brain import create_second_brain, create_note_ingestion_task
            agent = create_second_brain()
            task = create_note_ingestion_task("Test content", agent)
            assert task is not None

    def test_create_knowledge_graph_task(self):
        with patch("agents.second_brain.get_llm", return_value="fake-model"):
            from agents.second_brain import create_second_brain, create_knowledge_graph_task
            agent = create_second_brain()
            task = create_knowledge_graph_task([{"id": "1"}], agent)
            assert task is not None

    def test_create_review_task(self):
        with patch("agents.second_brain.get_llm", return_value="fake-model"):
            from agents.second_brain import create_second_brain, create_review_task
            agent = create_second_brain()
            task = create_review_task(agent)
            assert task is not None


# ---------------------------------------------------------------------------
# SecondBrainWorkflow
# ---------------------------------------------------------------------------

class TestSecondBrainWorkflow:

    def test_workflow_initialization(self, tmp_brain_dir):
        with patch("agents.second_brain.get_llm", return_value="fake-model"):
            wf = SecondBrainWorkflow(storage_dir=tmp_brain_dir)
            assert wf.agent is not None
            assert wf.ingestion is not None
            assert wf.graph is not None
            assert wf.sr_queue is not None
            assert wf.review is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_ingest_empty_content(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="Empty", content="")
        assert note["id"]
        assert note["title"] == "Empty"

    def test_ingest_unicode_content(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        note = pipeline.ingest(title="Unicode: 你好", content="Content with emojis 🚀 and ünïcödé.")
        assert "你好" in note["title"]

    def test_search_case_insensitive(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="UPPERCASE TITLE", content="lowercase content.")
        results_upper = pipeline.search("UPPERCASE")
        results_lower = pipeline.search("uppercase")
        assert len(results_upper) == 1
        assert len(results_lower) == 1

    def test_graph_with_single_note(self, tmp_brain_dir):
        graph = KnowledgeGraph(tmp_brain_dir)
        result = graph.build([{
            "id": "solo",
            "title": "Solo Note",
            "concepts": ["only-concept"],
            "tags": [],
            "links": [],
        }])
        assert result["stats"]["total_notes"] == 1
        assert result["orphans"] == []

    def test_sr_review_quality_boundary(self, tmp_brain_dir):
        sr = SpacedRepetitionQueue(tmp_brain_dir)
        note = {
            "id": "b1",
            "title": "Boundary",
            "sr_metadata": {
                "repetitions": 1,
                "ease_factor": 2.5,
                "interval": 1.0,
                "next_review": datetime.utcnow().isoformat(),
                "last_reviewed": None,
            },
        }
        # quality=3 should NOT reset
        updated = sr.review_note(note, quality=3)
        assert updated["sr_metadata"]["repetitions"] == 2

    def test_all_notes_skips_corrupt_files(self, tmp_brain_dir):
        pipeline = NoteIngestionPipeline(tmp_brain_dir)
        pipeline.ingest(title="Good", content="Valid note.")
        # Write a corrupt file
        corrupt_path = Path(tmp_brain_dir) / "notes" / "corrupt.json"
        corrupt_path.write_text("not valid json{{{")
        notes = pipeline.all_notes()
        # Should return only the valid note, skipping corrupt
        assert len(notes) == 1
        assert notes[0]["title"] == "Good"
