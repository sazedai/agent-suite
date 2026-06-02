"""Content Repurposing Agent.

Transforms existing content across formats:
blog-to-social, video-to-text, podcast-to-blog, infographic briefs, etc.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import fetch_page


def create_repurposer() -> Agent:
    return Agent(
        role="Content Repurposing Specialist",
        goal="Transform any content into multiple formats for maximum reach and ROI",
        backstory=(
            "You are a content strategist who specializes in getting maximum value "
            "from every piece of content. You turn blog posts into Twitter threads, "
            "podcasts into blog articles, webinars into infographics, and videos "
            "into LinkedIn carousels. You identify the core message and adapt it "
            "to each platform's strengths."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_blog_to_social_task(blog_url: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Fetch and analyze the blog post at {blog_url}. Then create: "
            "1) A Twitter/X thread (7-12 tweets) highlighting key insights, "
            "2) A LinkedIn post (1300 chars) with professional framing, "
            "3) An Instagram carousel outline (10 slides with titles/text), "
            "4) A short video script (60 seconds) summarizing the key takeaway, "
            "and 5) A newsletter blurb (100 words) with a CTA."
        ),
        expected_output="5 repurposed content pieces optimized for each platform, extracted from the source blog post",
        agent=agent,
    )


def create_video_to_blog_task(video_transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Transform the following video/podcast transcript into a full blog post:\n\n"
            f"{video_transcript}\n\n"
            "Create: SEO title, meta description, structured headings, "
            "written version of all key points, pull quotes, "
            "summary bullets, and suggested header image description."
        ),
        expected_output="Complete blog post adapted from video transcript with SEO metadata",
        agent=agent,
    )


class ContentRepurposeWorkflow:
    """Content repurposing workflow."""

    def __init__(self):
        self.agent = create_repurposer()

    def blog_to_social(self, blog_url: str) -> str:
        task = create_blog_to_social_task(blog_url, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def video_to_blog(self, transcript: str) -> str:
        task = create_video_to_blog_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
