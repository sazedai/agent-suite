"""Tests for Second Brain agent systems."""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import pytest

from agents.second_brain import (
    Note,
    NoteStore,
    NoteIngestion,
    KnowledgeGraph,
    SpacedRepetition,
    WeeklyReview,
    SecondBrainWorkflow,
    create_second_brain,
    create_note_ingestion_task,
    create_knowledge_graph_task,
    create_review_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="brain_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    return NoteStore(storage_dir=tmp_dir)


@pytest.fixture
def note():
    return Note(
        title="Test Note",
        content="This is a test note about Machine Learning and Python. #ml #python",
        tags=["test"],
        concepts=["Machine Learning", "Python"],
        actions=[""],
    )


def _make_note(title, content, tags=None, concepts=None, actions=None):
    return Note(
        title=title,
        content=content,
        tags=tags or [],
        concepts=concepts or [],
        actions=actions or [],
    )


# ---------------------------------------------------------------------------
# Note model
# ---------------------------------------------------------------------------

class TestNote:
    def test_note_creation(self, note):
        assert note.title == "Test Note"
        assert note.summary == ""
        assert note.tags == ["test"]
        assert note.concepts == ["Machine Learning", "Python"]

    def test_note_to_dict_roundtrip(self, note):
        d = note.to_dict()
        restored = Note.from_dict(d)
        assert restored.title == note.title
        assert restored.content == note.content
        assert restored.id == note.id
        assert restored.tags == note.tags
        assert restored.concepts == note.concepts

    def test_note_from_dict_preserves_sr_fields(self):
        d = {
            "id": "abc123",
            "title": "SR Test",
            "content": "Spaced repetition test note.",
            "summary": "SR note",
            "tags": [],
            "concepts": [],
            "links": [],
            "actions": [],
            "source": "",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "sr_interval": 6,
            "sr_easiness": 2.7,
            "sr_repetitions": 3,
            "sr_next_review": "2026-02-01T00:00:00",
        }
        note = Note.from_dict(d)
        assert note.sr_interval == 6
        assert note.sr_easiness == 2.7
        assert note.sr_repetitions == 3
        assert note.sr_next_review == "2026-02-01T00:00:00"

    def test_note_default_sr_fields(self):
        note = Note(title="New", content="Fresh note")
        assert note.sr_interval == 0
        assert note.sr_easiness == 2.5
        assert note.sr_repetitions == 0
        assert note.sr_next_review is None


# ---------------------------------------------------------------------------
# NoteStore
# ---------------------------------------------------------------------------

class TestNoteStore:
    def test_save_and_load(self, store, note):
        store.save(note)
        loaded = store.load(note.id)
        assert loaded is not None
        assert loaded.title == "Test Note"
        assert loaded.content == "This is a test note about Machine Learning and Python. #ml #python"

    def test_load_nonexistent(self, store):
        assert store.load("nonexistent") is None

    def test_delete(self, store, note):
        store.save(note)
        assert store.delete(note.id) is True
        assert store.load(note.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("fake-id") is False

    def test_load_all(self, store):
        notes = [
            _make_note("Note 1", "Content 1"),
            _make_note("Note 2", "Content 2"),
            _make_note("Note 3", "Content 3"),
        ]
        for n in notes:
            store.save(n)
        all_notes = store.load_all()
        assert len(all_notes) == 3
        titles = {n.title for n in all_notes}
        assert titles == {"Note 1", "Note 2", "Note 3"}

    def test_search_by_title(self, store):
        store.save(_make_note("Python Basics", "Learn python"))
        store.save(_make_note("Rust Intro", "Learn rust"))
        results = store.search("Python")
        assert len(results) == 1
        assert results[0].title == "Python Basics"

    def test_search_by_content(self, store):
        store.save(_make_note("A note", "This contains kubernetes content"))
        results = store.search("kubernetes")
        assert len(results) == 1

    def test_search_by_tag(self, store):
        store.save(_make_note("Tagged", "Content", tags=["ml", "ai"]))
        results = store.search("ml")
        assert len(results) == 1

    def test_search_case_insensitive(self, store):
        store.save(_make_note("Title", "Content about Docker"))
        results = store.search("docker")
        assert len(results) == 1

    def test_count(self, store):
        assert store.count == 0
        store.save(_make_note("A", "B"))
        assert store.count == 1


# ---------------------------------------------------------------------------
# NoteIngestion
# ---------------------------------------------------------------------------

class TestNoteIngestion:
    def test_basic_ingest(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest(
            "Machine Learning is a subset of Artificial Intelligence. #ml #ai",
            title="ML Basics",
        )
        assert note.title == "ML Basics"
        assert "ml" in note.tags
        assert "ai" in note.tags
        assert isinstance(note.concepts, list)

    def test_ingest_auto_title(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest("# My Heading\nSome content here.")
        assert "My Heading" in note.title

    def test_ingest_auto_title_plain(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest("First line is the title.\nMore content.")
        assert "First line is the title" in note.title

    def test_ingest_extracts_concepts(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest(
            "Neural Networks and Deep Learning are key concepts in Artificial Intelligence."
        )
        concepts_lower = [c.lower() for c in note.concepts]
        assert any("neural" in c for c in concepts_lower)
        assert any("deep" in c or "learning" in c for c in concepts_lower)

    def test_ingest_extracts_actions(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest(
            "- [ ] Review the paper\n- [x] Done item\nTODO: Send email"
        )
        assert any("Review the paper" in a for a in note.actions)
        assert any("Send email" in a for a in note.actions)

    def test_ingest_completed_checklist_not_action(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest("- [x] Already done")
        assert note.actions == []

    def test_ingest_generates_summary(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest("This is a great summary sentence. Extra text.")
        assert "great summary" in note.summary

    def test_ingest_links_by_concept(self, store):
        ing = NoteIngestion(store)
        n1 = ing.ingest(
            "Neural Networks are great for Natural Language Processing.",
            title="Neural Networks",
        )
        # Second note shares the "Natural Language Processing" concept
        n2 = ing.ingest(
            "Natural Language Processing uses Transformer Architecture extensively.",
            title="NLP Transforms",
        )
        # At least one should link to the other if concepts overlap
        assert len(n1.links) > 0 or len(n2.links) > 0

    def test_ingest_batch(self, store):
        ing = NoteIngestion(store)
        items = [
            {"content": "Note about Docker", "title": "Docker 101"},
            {"content": "Note about Kubernetes", "title": "K8s Intro"},
        ]
        notes = ing.ingest_batch(items)
        assert len(notes) == 2
        assert ing.stats["added"] == 2
        assert store.count == 2

    def test_ingest_with_source(self, store):
        ing = NoteIngestion(store)
        note = ing.ingest("Content", source="https://example.com")
        assert note.source == "https://example.com"


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    def _populate_store(self, store):
        notes_data = [
            ("Python Basics", "Python is a programming language. #python #coding"),
            ("Data Science", "Data Science uses Python and Statistics. #data #python"),
            ("Statistics 101", "Statistics is math for data. #statistics #math"),
            ("Rust Intro", "Rust is a systems language. #rust"),
        ]
        ing = NoteIngestion(store)
        for title, content in notes_data:
            ing.ingest(content, title=title)

    def test_build_creates_nodes_and_edges(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_note_nodes_exist(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        note_nodes = [n for n in graph.nodes if n.startswith("note:")]
        assert len(note_nodes) == 4

    def test_concept_nodes_exist(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        concept_nodes = [n for n in graph.nodes if n.startswith("concept:")]
        assert len(concept_nodes) > 0

    def test_clusters_by_shared_concepts(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        clusters = graph.clusters()
        # Python Basics and Data Science share #python / Python concept
        assert len(clusters) >= 1
        # The largest cluster should contain Python-related notes
        largest = clusters[0]
        assert largest["size"] >= 2

    def test_orphans(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        orphans = graph.orphans()
        # "Rust Intro" has no shared concepts with other notes
        assert "Rust Intro" in orphans

    def test_top_concepts(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        top = graph.top_concepts(3)
        assert len(top) > 0
        concepts = [c["concept"].lower() for c in top]
        assert any("python" in c for c in concepts)

    def test_to_dict(self, store):
        self._populate_store(store)
        graph = KnowledgeGraph(store)
        graph.build()
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "clusters" in d
        assert "orphans" in d
        assert "gaps" in d
        assert "top_concepts" in d

    def test_empty_graph(self, store):
        graph = KnowledgeGraph(store)
        graph.build()
        assert graph.nodes == {}
        assert graph.edges == []
        assert graph.clusters() == []
        assert graph.orphans() == []
        assert graph.top_concepts() == []


# ---------------------------------------------------------------------------
# SpacedRepetition
# ---------------------------------------------------------------------------

class TestSpacedRepetition:
    def _make_and_save(self, store, title="SR Note"):
        note = _make_note(title, "Review this note")
        store.save(note)
        return note

    def test_review_first_success(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        result = sr.review(note.id, quality=4)
        assert result.sr_repetitions == 1
        assert result.sr_interval == 1

    def test_review_second_success(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        sr.review(note.id, quality=4)
        result = sr.review(note.id, quality=5)
        assert result.sr_repetitions == 2
        assert result.sr_interval == 6

    def test_review_third_success(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        sr.review(note.id, quality=4)
        sr.review(note.id, quality=5)
        result = sr.review(note.id, quality=4)
        assert result.sr_repetitions == 3
        # interval = round(6 * easiness) which should be >= 6
        assert result.sr_interval >= 6

    def test_review_failure_resets(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        sr.review(note.id, quality=5)
        sr.review(note.id, quality=5)
        result = sr.review(note.id, quality=2)
        assert result.sr_repetitions == 0
        assert result.sr_interval == 1

    def test_review_quality_bounds(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        with pytest.raises(ValueError, match="Quality must be between 0 and 5"):
            sr.review(note.id, quality=6)
        with pytest.raises(ValueError, match="Quality must be between 0 and 5"):
            sr.review(note.id, quality=-1)

    def test_review_nonexistent_note(self, store):
        sr = SpacedRepetition(store)
        with pytest.raises(ValueError, match="not found"):
            sr.review("fake-id", quality=4)

    def test_easiness_never_below_1_3(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        for _ in range(10):
            sr.review(note.id, quality=0)
            note = store.load(note.id)
        assert note.sr_easiness >= 1.3

    def test_next_review_set(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        result = sr.review(note.id, quality=4)
        assert result.sr_next_review is not None

    def test_due_notes(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        # Never reviewed notes are due
        due = sr.due_notes()
        assert len(due) == 1

    def test_schedule(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        sr.review(note.id, quality=4)
        schedule = sr.schedule()
        assert len(schedule) == 1
        assert schedule[0]["title"] == "SR Note"

    def test_stats(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        stats = sr.stats()
        assert stats["total_notes"] == 1
        assert stats["reviewed_notes"] == 0
        assert stats["due_now"] == 1

    def test_stats_after_review(self, store):
        note = self._make_and_save(store)
        sr = SpacedRepetition(store)
        sr.review(note.id, quality=4)
        stats = sr.stats()
        assert stats["reviewed_notes"] == 1
        assert stats["due_now"] == 0  # Next review is in the future


# ---------------------------------------------------------------------------
# WeeklyReview
# ---------------------------------------------------------------------------

class TestWeeklyReview:
    def _populate(self, store):
        ing = NoteIngestion(store)
        ing.ingest(
            "Python is great for Machine Learning and Data Science. #python #ml",
            title="Python & ML",
        )
        ing.ingest(
            "Deep Learning requires GPU computing. #dl #gpu\nACTION: Research GPU options",
            title="Deep Learning",
        )
        ing.ingest(
            "Statistics fundamentals for Data Science. #statistics #data",
            title="Statistics",
        )

    def test_generate_returns_dict(self, tmp_dir):
        store = NoteStore(tmp_dir)
        self._populate(store)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        data = review.generate()
        assert isinstance(data, dict)
        assert "notes_added" in data
        assert "top_concepts" in data
        assert "revisitable" in data
        assert "open_actions" in data
        assert "sr_stats" in data

    def test_notes_added_count(self, tmp_dir):
        store = NoteStore(tmp_dir)
        self._populate(store)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        data = review.generate()
        assert data["notes_added"] >= 3

    def test_open_actions(self, tmp_dir):
        store = NoteStore(tmp_dir)
        self._populate(store)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        data = review.generate()
        assert any("GPU options" in a["action"] for a in data["open_actions"])

    def test_note_of_the_week(self, tmp_dir):
        store = NoteStore(tmp_dir)
        self._populate(store)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        data = review.generate()
        # At least one note should be the "note of the week"
        assert data["note_of_the_week"] is not None

    def test_format_report(self, tmp_dir):
        store = NoteStore(tmp_dir)
        self._populate(store)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        report = review.format_report()
        assert "# 🧠 Second Brain Weekly Review" in report
        assert "Notes Added" in report
        assert "Top Concepts" in report
        assert "Due for Review" in report
        assert "Open Action Items" in report
        assert "Spaced Repetition Stats" in report

    def test_empty_review(self, tmp_dir):
        store = NoteStore(tmp_dir)
        graph = KnowledgeGraph(store)
        sr = SpacedRepetition(store)
        review = WeeklyReview(store, graph, sr)
        data = review.generate()
        assert data["notes_added"] == 0
        assert data["note_of_the_week"] is None


# ---------------------------------------------------------------------------
# SecondBrainWorkflow
# ---------------------------------------------------------------------------

class TestSecondBrainWorkflow:
    def test_ingest_note(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        note = wf.ingest_note("Test note about Python. #python", title="Test")
        assert note.title == "Test"
        assert "python" in note.tags

    def test_build_graph(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        wf.ingest_note("Python and Data Science", title="DS")
        wf.ingest_note("Python and Machine Learning", title="ML")
        graph = wf.build_graph()
        assert len(graph.nodes) > 0

    def test_review_note(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        note = wf.ingest_note("Review me", title="Review")
        result = wf.review_note(note.id, quality=4)
        assert result.sr_repetitions == 1

    def test_due_notes(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        note = wf.ingest_note("Due note", title="Due")
        due = wf.due_notes()
        # Never reviewed — should be due
        assert any(n.id == note.id for n in due)

    def test_weekly_review_data(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        wf.ingest_note("Note for review", title="Weekly")
        data = wf.weekly_review_data()
        assert isinstance(data, dict)

    def test_weekly_review_format(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        wf.ingest_note("Note for report", title="Report")
        report = wf.weekly_review()
        assert "Second Brain" in report

    def test_search(self, tmp_dir):
        wf = SecondBrainWorkflow(storage_dir=tmp_dir)
        wf.ingest_note("Content about Kubernetes", title="K8s")
        results = wf.search("Kubernetes")
        assert len(results) == 1

    def test_store_persists(self, tmp_dir):
        wf1 = SecondBrainWorkflow(storage_dir=tmp_dir)
        wf1.ingest_note("Persisted note", title="Persist")
        # New workflow instance on same dir
        wf2 = SecondBrainWorkflow(storage_dir=tmp_dir)
        assert wf2.store.count >= 1


# ---------------------------------------------------------------------------
# Module imports (agent + tasks)
# ---------------------------------------------------------------------------

def test_module_imports():
    """All public symbols should be importable."""
    from agents.second_brain import (
        Note,
        NoteStore,
        NoteIngestion,
        KnowledgeGraph,
        SpacedRepetition,
        WeeklyReview,
        SecondBrainWorkflow,
        create_second_brain,
        create_note_ingestion_task,
        create_knowledge_graph_task,
        create_review_task,
    )
    assert Note is not None
    assert NoteStore is not None
    assert NoteIngestion is not None
    assert KnowledgeGraph is not None
    assert SpacedRepetition is not None
    assert WeeklyReview is not None
    assert SecondBrainWorkflow is not None


def test_task_function_signatures():
    """Task factory functions should accept the expected parameters."""
    import inspect

    sig = inspect.signature(create_note_ingestion_task)
    assert "note_content" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_knowledge_graph_task)
    assert "notes" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_review_task)
    assert "agent" in sig.parameters
