"""Content Production Workflows.

Generates blog posts, social media content, newsletters,
video scripts, and SEO-optimized copy. Supports multi-format
content production with sequential pipeline execution.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search, fetch_page


def create_content_producer() -> Agent:
    return Agent(
        role="Senior Content Producer",
        goal="Create high-quality, engaging, and SEO-optimized content across formats",
        backstory=(
            "You are a multi-format content expert who writes blog posts, tweets, "
            "LinkedIn posts, newsletters, and video scripts. You understand SEO, "
            "audience engagement, brand voice, and content strategy. Everything you "
            "produce is polished, on-brand, and optimized for distribution."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ── Blog Post ──────────────────────────────────────────────────────

def create_blog_post_task(
    topic: str,
    keywords: list[str],
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Write a comprehensive blog post about '{topic}'. "
            f"Target keywords: {', '.join(keywords)}. "
            "Include: SEO-optimized title, meta description, H2/H3 structure, "
            "introduction hook, 3-5 body sections with subheadings, "
            "data/statistics (search for real numbers), internal linking suggestions, "
            "call-to-action, and a conclusion. Target 1500-2000 words."
        ),
        expected_output=(
            "Full blog post draft with SEO metadata (title, meta description), "
            "heading structure (H2/H3), body content (1500-2000 words), internal "
            "linking suggestions, and call-to-action"
        ),
        agent=agent,
    )


# ── Social Media ───────────────────────────────────────────────────

def create_social_posts_task(
    topic: str,
    platforms: list[str],
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Create social media posts about '{topic}' for: {', '.join(platforms)}. "
            "For each platform tailor the format, tone, character limits, and hashtags. "
            "Include: Twitter/X thread (5-8 tweets), LinkedIn post (1300 chars), "
            "Instagram caption with hashtags, and a short video script hook."
        ),
        expected_output=(
            "Platform-optimized social media copy for each specified platform, "
            "including Twitter/X thread, LinkedIn post, Instagram caption, and "
            "video script hook"
        ),
        agent=agent,
    )


# ── Newsletter ─────────────────────────────────────────────────────

def create_newsletter_task(
    topic: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Create an email newsletter edition focused on '{topic}'. "
            "Include: subject line (A/B variants), preview text, 3-5 article summaries "
            "with 'read more' links, a personal note section, and call-to-action buttons. "
            "Keep total length under 600 words with clear visual hierarchy."
        ),
        expected_output=(
            "Complete email newsletter with subject line A/B variants, preview text, "
            "article summaries, personal notes, and CTA layout suggestions"
        ),
        agent=agent,
    )


# ── Video Script ───────────────────────────────────────────────────

def create_video_script_task(
    topic: str,
    duration_seconds: int,
    style: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Write a {duration_seconds}-second video script about '{topic}'. "
            f"Style: {style}. "
            "Include: hook (first 5 seconds), scene-by-scene breakdown with "
            "visual directions, on-screen text overlays, voiceover/narration copy, "
            "background music suggestions, call-to-action, and end screen. "
            "Format the script in a two-column table: VISUAL | AUDIO/TEXT.\n\n"
            f"Estimated word count: ~{duration_seconds * 2} words "
            "(average speaking pace is ~2 words/s)."
        ),
        expected_output=(
            f"Complete {duration_seconds}s video script with scene-by-scene breakdown, "
            "visual directions, on-screen text, narration copy, music suggestions, "
            "CTA, and end screen. Formatted as VISUAL | AUDIO table."
        ),
        agent=agent,
    )


# ── Content Brief ──────────────────────────────────────────────────

def create_content_brief_task(
    topic: str,
    target_audience: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            f"Create a content brief for '{topic}' targeting '{target_audience}'. "
            "Include: content goals, target audience persona, key messages, "
            "SEO keywords to research, competitor content analysis, "
            "content format recommendations (blog/social/newsletter/video), "
            "distribution channels, and KPIs to track."
        ),
        expected_output=(
            "Content brief document covering goals, audience persona, key messages, "
            "SEO keyword candidates, competitor analysis, format recommendations, "
            "distribution channels, and success KPIs"
        ),
        agent=agent,
    )


# ── Workflows ──────────────────────────────────────────────────────

class ContentProductionWorkflow:
    """Multi-format content production workflow.

    Provides individual task creators plus a full multi-format
    pipeline that chains tasks sequentially via CrewAI.
    """

    def __init__(self, target_audience: str = "general"):
        self.agent = create_content_producer()
        self.target_audience = target_audience

    def write_blog(self, topic: str, keywords: list[str]) -> str:
        """Produce a full SEO-optimized blog post."""
        task = create_blog_post_task(topic, keywords, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def create_social(self, topic: str, platforms: list[str] | None = None) -> str:
        """Generate platform-tailored social media posts."""
        platforms = platforms or ["twitter", "linkedin", "instagram"]
        task = create_social_posts_task(topic, platforms, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def write_newsletter(self, topic: str) -> str:
        """Create a complete email newsletter edition."""
        task = create_newsletter_task(topic, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def create_video_script(
        self,
        topic: str,
        duration_seconds: int = 60,
        style: str = "educational",
    ) -> str:
        """Write a scene-by-scene script for a short-form video."""
        task = create_video_script_task(topic, duration_seconds, style, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def plan_brief(self, topic: str, target_audience: str = "general") -> str:
        """Generate a content brief for a topic and audience segment."""
        task = create_content_brief_task(topic, target_audience, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def full_pipeline(
        self,
        topic: str,
        keywords: list[str],
        platforms: list[str] | None = None,
        video_duration: int = 60,
        video_style: str = "educational",
    ) -> str:
        """Run the complete multi-format content pipeline sequentially.

        Produces: content brief → blog post → social posts → newsletter → video script.
        Each task feeds context to the next via CrewAI sequential processing.
        """
        platforms = platforms or ["twitter", "linkedin", "instagram"]

        brief_task = create_content_brief_task(topic, self.target_audience, self.agent)
        blog_task = create_blog_post_task(topic, keywords, self.agent)
        social_task = create_social_posts_task(topic, platforms, self.agent)
        newsletter_task = create_newsletter_task(topic, self.agent)
        video_task = create_video_script_task(
            topic, video_duration, video_style, self.agent
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[brief_task, blog_task, social_task, newsletter_task, video_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
