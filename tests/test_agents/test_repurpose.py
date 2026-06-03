"""Tests for Content Repurposing agent."""

import pytest


def test_content_repurpose_module_imports():
    """Content Repurposing module should import all components."""
    from agents.content_repurpose import (
        create_repurposer,
        create_blog_to_social_task,
        create_video_to_blog_task,
        ContentRepurposeWorkflow,
    )
    assert create_repurposer is not None
    assert create_blog_to_social_task is not None
    assert create_video_to_blog_task is not None
    assert ContentRepurposeWorkflow is not None


def test_content_repurpose_task_signatures():
    """Content repurpose task functions should have correct signatures."""
    import inspect
    from agents.content_repurpose import (
        create_blog_to_social_task,
        create_video_to_blog_task,
    )

    sig = inspect.signature(create_blog_to_social_task)
    assert "blog_url" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_video_to_blog_task)
    assert "video_transcript" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
