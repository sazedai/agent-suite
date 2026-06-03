"""Tests for Content Production agent."""

import inspect
import pytest
from unittest.mock import patch, MagicMock

from agents.content_production import (
    create_content_producer,
    create_blog_post_task,
    create_social_posts_task,
    create_newsletter_task,
    create_video_script_task,
    create_content_brief_task,
    ContentProductionWorkflow,
)


# ── Module-level import tests ───────────────────────────────────────

def test_module_imports():
    """All public symbols should be importable."""
    assert create_content_producer is not None
    assert create_blog_post_task is not None
    assert create_social_posts_task is not None
    assert create_newsletter_task is not None
    assert create_video_script_task is not None
    assert create_content_brief_task is not None
    assert ContentProductionWorkflow is not None


# ── Agent factory ──────────────────────────────────────────────────

def test_create_content_producer():
    """create_content_producer should return a CrewAI Agent."""
    with patch("agents.content_production.get_llm") as mock_get_llm:
        mock_get_llm.return_value = "gpt-4o"
        agent = create_content_producer()
    assert agent.role == "Senior Content Producer"
    assert "SEO-optimized" in agent.goal
    assert agent.verbose is True
    assert agent.allow_delegation is False


# ── Task factory signatures ────────────────────────────────────────

def test_blog_post_task_signature():
    """create_blog_post_task should accept topic, keywords, agent."""
    sig = inspect.signature(create_blog_post_task)
    assert "topic" in sig.parameters
    assert "keywords" in sig.parameters
    assert "agent" in sig.parameters


def test_social_posts_task_signature():
    """create_social_posts_task should accept topic, platforms, agent."""
    sig = inspect.signature(create_social_posts_task)
    assert "topic" in sig.parameters
    assert "platforms" in sig.parameters
    assert "agent" in sig.parameters


def test_newsletter_task_signature():
    """create_newsletter_task should accept topic, agent."""
    sig = inspect.signature(create_newsletter_task)
    assert "topic" in sig.parameters
    assert "agent" in sig.parameters


def test_video_script_task_signature():
    """create_video_script_task should accept topic, duration_seconds, style, agent."""
    sig = inspect.signature(create_video_script_task)
    assert "topic" in sig.parameters
    assert "duration_seconds" in sig.parameters
    assert "style" in sig.parameters
    assert "agent" in sig.parameters


def test_content_brief_task_signature():
    """create_content_brief_task should accept topic, target_audience, agent."""
    sig = inspect.signature(create_content_brief_task)
    assert "topic" in sig.parameters
    assert "target_audience" in sig.parameters
    assert "agent" in sig.parameters


# ── Task factory output ────────────────────────────────────────────

def _make_mock_agent():
    """Return a mock Agent with a .get() method for process_model_config."""
    agent = MagicMock()
    agent.get = MagicMock(return_value=None)
    return agent


def test_blog_post_task_returns_task():
    """create_blog_post_task should return a CrewAI Task with blog description."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        result = create_blog_post_task("AI trends", ["ai", "ml"], agent)
    assert result is mock_task
    call_kwargs = MockTask.call_args
    assert "blog post" in call_kwargs.kwargs["description"].lower()
    assert "AI trends" in call_kwargs.kwargs["description"]
    assert "ai" in call_kwargs.kwargs["description"]


def test_social_posts_task_returns_task():
    """create_social_posts_task should return a Task with social description."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        result = create_social_posts_task("AI trends", ["twitter", "linkedin"], agent)
    assert result is mock_task
    call_kwargs = MockTask.call_args
    assert "social media" in call_kwargs.kwargs["description"].lower()
    assert "twitter" in call_kwargs.kwargs["description"]


def test_newsletter_task_returns_task():
    """create_newsletter_task should return a Task with newsletter description."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        result = create_newsletter_task("AI trends", agent)
    assert result is mock_task
    call_kwargs = MockTask.call_args
    assert "newsletter" in call_kwargs.kwargs["description"].lower()
    assert "AI trends" in call_kwargs.kwargs["description"]


def test_video_script_task_returns_task():
    """create_video_script_task should return a Task with video script description."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        result = create_video_script_task("AI trends", 60, "educational", agent)
    assert result is mock_task
    call_kwargs = MockTask.call_args
    assert "video script" in call_kwargs.kwargs["description"].lower()
    assert "60" in call_kwargs.kwargs["description"]
    assert "educational" in call_kwargs.kwargs["description"]


def test_content_brief_task_returns_task():
    """create_content_brief_task should return a Task with brief description."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        result = create_content_brief_task("AI trends", "developers", agent)
    assert result is mock_task
    call_kwargs = MockTask.call_args
    assert "content brief" in call_kwargs.kwargs["description"].lower()
    assert "developers" in call_kwargs.kwargs["description"]


# ── Workflow class ─────────────────────────────────────────────────

def test_workflow_init():
    """ContentProductionWorkflow should create an agent on init."""
    with patch("agents.content_production.create_content_producer") as mock_create:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        wf = ContentProductionWorkflow(target_audience="developers")
    mock_create.assert_called_once()
    assert wf.agent is mock_agent
    assert wf.target_audience == "developers"


def test_workflow_init_default_audience():
    """Default target_audience should be 'general'."""
    with patch("agents.content_production.create_content_producer"):
        wf = ContentProductionWorkflow()
    assert wf.target_audience == "general"


def test_workflow_write_blog():
    """write_blog should create a Crew and call kickoff."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_blog_post_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "blog output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.write_blog("AI trends", ["ai", "ml"])

    assert result == "blog output"
    mock_task.assert_called_once_with("AI trends", ["ai", "ml"], mock_agent)
    MockCrew.assert_called_once()
    mock_crew.kickoff.assert_called_once()


def test_workflow_create_social():
    """create_social should default to twitter/linkedin/instagram."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_social_posts_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "social output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.create_social("AI trends")

    assert result == "social output"
    mock_task.assert_called_once_with(
        "AI trends", ["twitter", "linkedin", "instagram"], mock_agent
    )


def test_workflow_create_social_custom_platforms():
    """create_social should accept custom platform lists."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_social_posts_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "social output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.create_social("AI trends", platforms=["twitter", "tiktok"])

    mock_task.assert_called_once_with("AI trends", ["twitter", "tiktok"], mock_agent)


def test_workflow_write_newsletter():
    """write_newsletter should create a Crew and call kickoff."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_newsletter_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "newsletter output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.write_newsletter("AI trends")

    assert result == "newsletter output"
    mock_task.assert_called_once_with("AI trends", mock_agent)


def test_workflow_create_video_script():
    """create_video_script should pass duration and style to task factory."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_video_script_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "video script output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.create_video_script("AI trends", duration_seconds=120, style="entertaining")

    assert result == "video script output"
    mock_task.assert_called_once_with("AI trends", 120, "entertaining", mock_agent)


def test_workflow_create_video_script_defaults():
    """create_video_script should default to 60s educational."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_video_script_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "video output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.create_video_script("AI trends")

    mock_task.assert_called_once_with("AI trends", 60, "educational", mock_agent)


def test_workflow_plan_brief():
    """plan_brief should create a content brief task and run it."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_content_brief_task") as mock_task, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_task.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "brief output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        result = wf.plan_brief("AI trends", target_audience="developers")

    assert result == "brief output"
    mock_task.assert_called_once_with("AI trends", "developers", mock_agent)


def test_workflow_full_pipeline():
    """full_pipeline should chain all 5 tasks sequentially."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_content_brief_task") as mock_brief, \
         patch("agents.content_production.create_blog_post_task") as mock_blog, \
         patch("agents.content_production.create_social_posts_task") as mock_social, \
         patch("agents.content_production.create_newsletter_task") as mock_news, \
         patch("agents.content_production.create_video_script_task") as mock_video, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_brief.return_value = MagicMock()
        mock_blog.return_value = MagicMock()
        mock_social.return_value = MagicMock()
        mock_news.return_value = MagicMock()
        mock_video.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "full pipeline output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow(target_audience="developers")
        result = wf.full_pipeline(
            topic="AI trends",
            keywords=["ai", "ml"],
            platforms=["twitter", "linkedin"],
            video_duration=90,
            video_style="entertaining",
        )

    assert result == "full pipeline output"
    mock_brief.assert_called_once_with("AI trends", "developers", mock_agent)
    mock_blog.assert_called_once_with("AI trends", ["ai", "ml"], mock_agent)
    mock_social.assert_called_once_with("AI trends", ["twitter", "linkedin"], mock_agent)
    mock_news.assert_called_once_with("AI trends", mock_agent)
    mock_video.assert_called_once_with("AI trends", 90, "entertaining", mock_agent)

    # Verify Crew was created with all 5 tasks in order
    call_kwargs = MockCrew.call_args.kwargs
    assert len(call_kwargs["tasks"]) == 5
    # process kwarg should be set to a sequential process enum value
    assert call_kwargs["process"] is not None


def test_workflow_full_pipeline_defaults():
    """full_pipeline should use default platforms and video settings."""
    with patch("agents.content_production.create_content_producer") as mock_create, \
         patch("agents.content_production.create_content_brief_task"), \
         patch("agents.content_production.create_blog_post_task"), \
         patch("agents.content_production.create_social_posts_task") as mock_social, \
         patch("agents.content_production.create_newsletter_task"), \
         patch("agents.content_production.create_video_script_task") as mock_video, \
         patch("agents.content_production.Crew") as MockCrew:
        mock_create.return_value = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "output"
        MockCrew.return_value = mock_crew

        wf = ContentProductionWorkflow()
        wf.full_pipeline(topic="AI trends", keywords=["ai"])

    mock_social.assert_called_once_with(
        "AI trends", ["twitter", "linkedin", "instagram"], mock_create.return_value
    )
    mock_video.assert_called_once_with(
        "AI trends", 60, "educational", mock_create.return_value
    )


# ── Task description content checks ────────────────────────────────

def test_blog_task_description_contains_seo():
    """Blog task description should mention SEO elements."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        MockTask.return_value = MagicMock()
        create_blog_post_task("test topic", ["kw1", "kw2"], agent)
    desc = MockTask.call_args.kwargs["description"]
    assert "SEO-optimized title" in desc
    assert "meta description" in desc
    assert "1500-2000 words" in desc


def test_video_task_description_contains_duration():
    """Video task description should include the duration and style."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        MockTask.return_value = MagicMock()
        create_video_script_task("test topic", 120, "documentary", agent)
    desc = MockTask.call_args.kwargs["description"]
    assert "120" in desc
    assert "documentary" in desc
    assert "VISUAL" in desc
    assert "AUDIO" in desc


def test_newsletter_task_description_contains_ab_variants():
    """Newsletter task description should mention A/B subject lines."""
    agent = _make_mock_agent()
    with patch("agents.content_production.Task") as MockTask:
        MockTask.return_value = MagicMock()
        create_newsletter_task("test topic", agent)
    desc = MockTask.call_args.kwargs["description"]
    assert "A/B" in desc
    assert "subject line" in desc.lower()
