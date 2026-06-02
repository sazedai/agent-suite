"""Content Production Workflows.

Generates blog posts, social media content, newsletters,
video scripts, and SEO-optimized copy.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search


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


def create_blog_post_task(topic: str, keywords: list[str], agent: Agent) -> Task:
    return Task(
        description=(
            f"Write a comprehensive blog post about '{topic}'. "
            f"Target keywords: {', '.join(keywords)}. "
            "Include: SEO-optimized title, meta description, H2/H3 structure, "
            "introduction hook, 3-5 body sections with subheadings, "
            "data/statistics (search for real numbers), internal linking suggestions, "
            "call-to-action, and a conclusion. Target 1500-2000 words."
        ),
        expected_output="Full blog post draft with SEO metadata, structure, and internal linking suggestions",
        agent=agent,
    )


def create_social_posts_task(topic: str, platforms: list[str], agent: Agent) -> Task:
    return Task(
        description=(
            f"Create social media posts about '{topic}' for: {', '.join(platforms)}. "
            "For each platform tailor the format, tone, character limits, and hashtags. "
            "Include: Twitter/X thread (5-8 tweets), LinkedIn post (1300 chars), "
            "Instagram caption with hashtags, and a short video script hook."
        ),
        expected_output="Platform-optimized social media copy for each specified platform",
        agent=agent,
    )


def create_newsletter_task(topic: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Create an email newsletter edition focused on '{topic}'. "
            "Include: subject line (A/B variants), preview text, 3-5 article summaries "
            "with 'read more' links, a personal note section, and call-to-action buttons. "
            "Keep total length under 600 words with clear visual hierarchy."
        ),
        expected_output="Complete email newsletter with subject lines, copy, and layout suggestions",
        agent=agent,
    )


class ContentProductionWorkflow:
    """Multi-format content production workflow."""

    def __init__(self):
        self.agent = create_content_producer()

    def write_blog(self, topic: str, keywords: list[str]) -> str:
        task = create_blog_post_task(topic, keywords, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def create_social(self, topic: str, platforms: list[str] | None = None) -> str:
        platforms = platforms or ["twitter", "linkedin", "instagram"]
        task = create_social_posts_task(topic, platforms, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def write_newsletter(self, topic: str) -> str:
        task = create_newsletter_task(topic, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
