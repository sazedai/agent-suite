"""Content Repurposing Agent.

Transforms existing content across formats and platforms:
blog-to-social, video-to-text, podcast-to-blog, infographic briefs,
cross-platform adaptation, newsletter generation, and carousel outlines.
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


# ── Blog-to-Social tasks ──────────────────────────────────────────────

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
        expected_output=(
            "5 repurposed content pieces optimized for each platform, "
            "extracted from the source blog post"
        ),
        agent=agent,
    )


def create_blog_to_twitter_thread_task(blog_url: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Fetch and analyze the blog post at {blog_url}. "
            "Create a compelling Twitter/X thread of 7-12 tweets that: "
            "opens with a strong hook tweet, breaks down each key insight "
            "into its own tweet with specific data or quotes, uses numbered "
            "formatting for scannability, and ends with a CTA tweet linking "
            "back to the original post. Each tweet must be under 280 characters."
        ),
        expected_output="A complete Twitter/X thread (7-12 tweets) ready to post",
        agent=agent,
    )


def create_blog_to_linkedin_task(blog_url: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Fetch and analyze the blog post at {blog_url}. "
            "Write a LinkedIn post (max 1300 characters) with professional "
            "framing. Use a personal-story opener, 3-5 bullet points of key "
            "takeaways, and end with a question to drive engagement. "
            "Include 3-5 relevant hashtags."
        ),
        expected_output="A LinkedIn-formatted post under 1300 characters with hashtags",
        agent=agent,
    )


# ── Video-to-Blog tasks ───────────────────────────────────────────────

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


def create_podcast_to_blog_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Adapt the following podcast transcript into an engaging blog article:\n\n"
            f"{transcript}\n\n"
            "Structure: catchy headline, introduction that hooks the reader, "
            "section headings for each major topic discussed, cleaned-up prose "
            "(remove filler words, add transitions), 3 pull quotes from the host "
            "or guest, key takeaways section at the end, and suggested tags/categories."
        ),
        expected_output="A publication-ready blog article derived from the podcast transcript",
        agent=agent,
    )


def create_webinar_to_guide_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Convert the following webinar transcript into a step-by-step how-to guide:\n\n"
            f"{transcript}\n\n"
            "Structure: descriptive title, prerequisites section, numbered steps "
            "with clear actions, screenshots/diagrams callout notes, "
            "troubleshooting tips from Q&A, and a summary checklist."
        ),
        expected_output="A structured how-to guide with numbered steps, tips, and checklist",
        agent=agent,
    )


# ── Cross-platform adaptation tasks ────────────────────────────────────

def create_cross_platform_task(
    content: str,
    source_platform: str,
    target_platforms: list[str],
    agent: Agent,
) -> Task:
    return Task(
        description=(
            "Adapt the following content from "
            f"{source_platform} to {', '.join(target_platforms)}:\n\n"
            f"{content}\n\n"
            "For each target platform, adapt: tone, length, formatting, "
            "hashtags/CTA style, media suggestions, and optimal posting time "
            "recommendations. Preserve the core message while maximizing "
            "engagement on each platform."
        ),
        expected_output=(
            f"Platform-adapted versions for {', '.join(target_platforms)} "
            "with tone, length, and formatting guidelines per platform"
        ),
        agent=agent,
    )


def create_infographic_brief_task(content: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Analyze the following content and create an infographic design brief:\n\n"
            f"{content}\n\n"
            "Include: suggested title, 5-7 data points or statistics to visualize, "
            "section layout (flow from top to bottom), color palette suggestions "
            "based on topic mood, icon/illustration recommendations, and "
            "a caption for social sharing."
        ),
        expected_output="A detailed infographic design brief with layout and data visualization plan",
        agent=agent,
    )


def create_newsletter_brief_task(content: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Turn the following content into a newsletter edition brief:\n\n"
            f"{content}\n\n"
            "Include: 2 subject line variants (A/B test), preview text (90 chars), "
            "3 article summaries with 'read more' CTAs, a personal note paragraph, "
            "suggested images/headers, and unsubscribe-safe copy length."
        ),
        expected_output="A newsletter edition brief with subject lines, copy, and layout plan",
        agent=agent,
    )


# ── Orchestration tasks ────────────────────────────────────────────────

def create_full_repurpose_task(content: str, content_type: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Take this {content_type} and create a comprehensive multi-platform repurposing plan:\n\n"
            f"{content}\n\n"
            "Produce ALL of the following: "
            "1) Twitter/X thread (8 tweets), "
            "2) LinkedIn post (1300 chars), "
            "3) Instagram carousel (10 slides), "
            "4) Blog post outline (if not already a blog), "
            "5) Newsletter edition brief, "
            "6) Infographic design brief, "
            "7) Short video script (60 sec), "
            "8) 3 platform-specific hashtag sets."
        ),
        expected_output="8-piece multi-platform repurposing plan extracted from the source content",
        agent=agent,
    )


# ── Workflow class ─────────────────────────────────────────────────────

class ContentRepurposeWorkflow:
    """Content repurposing transformation workflows."""

    def __init__(self):
        self.agent = create_repurposer()

    # Blog-to-social entry points
    def blog_to_social(self, blog_url: str) -> str:
        task = create_blog_to_social_task(blog_url, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def blog_to_twitter(self, blog_url: str) -> str:
        task = create_blog_to_twitter_thread_task(blog_url, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def blog_to_linkedin(self, blog_url: str) -> str:
        task = create_blog_to_linkedin_task(blog_url, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Video/audio-to-text entry points
    def video_to_blog(self, transcript: str) -> str:
        task = create_video_to_blog_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def podcast_to_blog(self, transcript: str) -> str:
        task = create_podcast_to_blog_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def webinar_to_guide(self, transcript: str) -> str:
        task = create_webinar_to_guide_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Cross-platform adaptation
    def adapt_platforms(
        self,
        content: str,
        source_platform: str = "blog",
        target_platforms: list[str] | None = None,
    ) -> str:
        targets = target_platforms or ["twitter", "linkedin", "instagram"]
        task = create_cross_platform_task(content, source_platform, targets, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def create_infographic_brief(self, content: str) -> str:
        task = create_infographic_brief_task(content, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def create_newsletter(self, content: str) -> str:
        task = create_newsletter_brief_task(content, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    # Full orchestration
    def full_repurpose(self, content: str, content_type: str = "article") -> str:
        task = create_full_repurpose_task(content, content_type, self.agent)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
