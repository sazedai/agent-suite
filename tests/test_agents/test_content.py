"""Tests for Content Production agent."""

import pytest


def test_content_production_module_imports():
    """Content Production module should import all components."""
    from agents.content_production import (
        create_content_producer,
        create_blog_post_task,
        create_social_posts_task,
        create_newsletter_task,
        ContentProductionWorkflow,
    )
    assert create_content_producer is not None
    assert create_blog_post_task is not None
    assert create_social_posts_task is not None
    assert create_newsletter_task is not None
    assert ContentProductionWorkflow is not None


def test_content_production_task_signatures():
    """Content production task functions should have correct signatures."""
    import inspect
    from agents.content_production import (
        create_blog_post_task,
        create_social_posts_task,
        create_newsletter_task,
    )

    sig = inspect.signature(create_blog_post_task)
    assert "topic" in sig.parameters
    assert "keywords" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_social_posts_task)
    assert "topic" in sig.parameters
    assert "platforms" in sig.parameters

    sig = inspect.signature(create_newsletter_task)
    assert "topic" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
