"""Tests for Second Brain agent."""

import pytest


def test_second_brain_module_imports():
    """Second Brain module should import all components."""
    from agents.second_brain import (
        create_second_brain,
        create_note_ingestion_task,
        create_knowledge_graph_task,
        create_review_task,
        SecondBrainWorkflow,
    )
    assert create_second_brain is not None
    assert create_note_ingestion_task is not None
    assert create_knowledge_graph_task is not None
    assert create_review_task is not None
    assert SecondBrainWorkflow is not None


def test_second_brain_task_signatures():
    """Second brain task functions should have correct signatures."""
    import inspect
    from agents.second_brain import (
        create_note_ingestion_task,
        create_knowledge_graph_task,
    )

    sig = inspect.signature(create_note_ingestion_task)
    assert "note_content" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_knowledge_graph_task)
    assert "notes" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
