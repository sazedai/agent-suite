"""Tests for Content Repurposing agent."""

import inspect

import pytest


# ── Import tests ────────────────────────────────────────────────────────


def test_repurpose_module_imports():
    """Content repurpose module should import all public components."""
    from agents.content_repurpose import (
        create_repurposer,
        create_blog_to_social_task,
        create_blog_to_twitter_thread_task,
        create_blog_to_linkedin_task,
        create_video_to_blog_task,
        create_podcast_to_blog_task,
        create_webinar_to_guide_task,
        create_cross_platform_task,
        create_infographic_brief_task,
        create_newsletter_brief_task,
        create_full_repurpose_task,
        ContentRepurposeWorkflow,
    )
    assert create_repurposer is not None
    assert create_blog_to_social_task is not None
    assert create_blog_to_twitter_thread_task is not None
    assert create_blog_to_linkedin_task is not None
    assert create_video_to_blog_task is not None
    assert create_podcast_to_blog_task is not None
    assert create_webinar_to_guide_task is not None
    assert create_cross_platform_task is not None
    assert create_infographic_brief_task is not None
    assert create_newsletter_brief_task is not None
    assert create_full_repurpose_task is not None
    assert ContentRepurposeWorkflow is not None


# ── Signature tests ─────────────────────────────────────────────────────


class TestTaskSignatures:
    """Verify task factory functions accept the expected parameters."""

    def test_blog_to_social_signature(self):
        from agents.content_repurpose import create_blog_to_social_task
        sig = inspect.signature(create_blog_to_social_task)
        assert "blog_url" in sig.parameters
        assert "agent" in sig.parameters

    def test_blog_to_twitter_signature(self):
        from agents.content_repurpose import create_blog_to_twitter_thread_task
        sig = inspect.signature(create_blog_to_twitter_thread_task)
        assert "blog_url" in sig.parameters
        assert "agent" in sig.parameters

    def test_blog_to_linkedin_signature(self):
        from agents.content_repurpose import create_blog_to_linkedin_task
        sig = inspect.signature(create_blog_to_linkedin_task)
        assert "blog_url" in sig.parameters
        assert "agent" in sig.parameters

    def test_video_to_blog_signature(self):
        from agents.content_repurpose import create_video_to_blog_task
        sig = inspect.signature(create_video_to_blog_task)
        assert "video_transcript" in sig.parameters
        assert "agent" in sig.parameters

    def test_podcast_to_blog_signature(self):
        from agents.content_repurpose import create_podcast_to_blog_task
        sig = inspect.signature(create_podcast_to_blog_task)
        assert "transcript" in sig.parameters
        assert "agent" in sig.parameters

    def test_webinar_to_guide_signature(self):
        from agents.content_repurpose import create_webinar_to_guide_task
        sig = inspect.signature(create_webinar_to_guide_task)
        assert "transcript" in sig.parameters
        assert "agent" in sig.parameters

    def test_cross_platform_signature(self):
        from agents.content_repurpose import create_cross_platform_task
        sig = inspect.signature(create_cross_platform_task)
        assert "content" in sig.parameters
        assert "source_platform" in sig.parameters
        assert "target_platforms" in sig.parameters
        assert "agent" in sig.parameters

    def test_infographic_brief_signature(self):
        from agents.content_repurpose import create_infographic_brief_task
        sig = inspect.signature(create_infographic_brief_task)
        assert "content" in sig.parameters
        assert "agent" in sig.parameters

    def test_newsletter_brief_signature(self):
        from agents.content_repurpose import create_newsletter_brief_task
        sig = inspect.signature(create_newsletter_brief_task)
        assert "content" in sig.parameters
        assert "agent" in sig.parameters

    def test_full_repurpose_signature(self):
        from agents.content_repurpose import create_full_repurpose_task
        sig = inspect.signature(create_full_repurpose_task)
        assert "content" in sig.parameters
        assert "content_type" in sig.parameters
        assert "agent" in sig.parameters


# ── Task description content tests ──────────────────────────────────────


class TestTaskDescriptions:
    """Verify task descriptions contain key platform/format directives."""

    def test_blog_to_social_description_contains_five_platforms(self):
        from agents.content_repurpose import create_blog_to_social_task
        from crewai import Agent
        # We can't instantiate a real Agent without an LLM, so test by
        # inspecting the function source
        src = inspect.getsource(create_blog_to_social_task)
        assert "Twitter" in src or "tweet" in src
        assert "LinkedIn" in src
        assert "Instagram" in src
        assert "video script" in src or "video" in src
        assert "newsletter" in src

    def test_cross_platform_description_mentions_target_platforms(self):
        from agents.content_repurpose import create_cross_platform_task
        src = inspect.getsource(create_cross_platform_task)
        assert "target_platforms" in src
        assert "tone" in src
        assert "formatting" in src

    def test_full_repurpose_description_produces_eight_pieces(self):
        from agents.content_repurpose import create_full_repurpose_task
        src = inspect.getsource(create_full_repurpose_task)
        assert "Twitter" in src or "tweet" in src
        assert "LinkedIn" in src
        assert "Instagram" in src
        assert "Blog" in src or "blog" in src
        assert "Newsletter" in src or "newsletter" in src
        assert "Infographic" in src or "infographic" in src
        assert "video script" in src or "video" in src


# ── Workflow class tests ────────────────────────────────────────────────


def test_workflow_has_blog_to_social():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "blog_to_social")
    sig = inspect.signature(ContentRepurposeWorkflow.blog_to_social)
    assert "blog_url" in sig.parameters


def test_workflow_has_blog_to_twitter():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "blog_to_twitter")
    sig = inspect.signature(ContentRepurposeWorkflow.blog_to_twitter)
    assert "blog_url" in sig.parameters


def test_workflow_has_blog_to_linkedin():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "blog_to_linkedin")
    sig = inspect.signature(ContentRepurposeWorkflow.blog_to_linkedin)
    assert "blog_url" in sig.parameters


def test_workflow_has_video_to_blog():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "video_to_blog")
    sig = inspect.signature(ContentRepurposeWorkflow.video_to_blog)
    assert "transcript" in sig.parameters


def test_workflow_has_podcast_to_blog():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "podcast_to_blog")
    sig = inspect.signature(ContentRepurposeWorkflow.podcast_to_blog)
    assert "transcript" in sig.parameters


def test_workflow_has_webinar_to_guide():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "webinar_to_guide")
    sig = inspect.signature(ContentRepurposeWorkflow.webinar_to_guide)
    assert "transcript" in sig.parameters


def test_workflow_has_adapt_platforms():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "adapt_platforms")
    sig = inspect.signature(ContentRepurposeWorkflow.adapt_platforms)
    assert "content" in sig.parameters
    assert "source_platform" in sig.parameters
    assert "target_platforms" in sig.parameters


def test_workflow_adapt_platforms_defaults():
    from agents.content_repurpose import ContentRepurposeWorkflow
    sig = inspect.signature(ContentRepurposeWorkflow.adapt_platforms)
    params = sig.parameters
    assert params["source_platform"].default == "blog"
    # target_platforms default should be None (then resolved to the 3 defaults)
    assert params["target_platforms"].default is None


def test_workflow_has_create_infographic_brief():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "create_infographic_brief")
    sig = inspect.signature(ContentRepurposeWorkflow.create_infographic_brief)
    assert "content" in sig.parameters


def test_workflow_has_create_newsletter():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "create_newsletter")
    sig = inspect.signature(ContentRepurposeWorkflow.create_newsletter)
    assert "content" in sig.parameters


def test_workflow_has_full_repurpose():
    from agents.content_repurpose import ContentRepurposeWorkflow
    assert hasattr(ContentRepurposeWorkflow, "full_repurpose")
    sig = inspect.signature(ContentRepurposeWorkflow.full_repurpose)
    assert "content" in sig.parameters
    assert "content_type" in sig.parameters


def test_workflow_full_repurpose_defaults():
    from agents.content_repurpose import ContentRepurposeWorkflow
    sig = inspect.signature(ContentRepurposeWorkflow.full_repurpose)
    assert sig.parameters["content_type"].default == "article"


# ── Version / module checks ────────────────────────────────────────────


def test_package_version():
    from agents import __version__
    assert __version__ == "0.1.0"
