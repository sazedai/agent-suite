"""Executive Assistant Workflows.

Calendar management, meeting preparation, travel planning,
priority triage, email drafting, and executive communications.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


def create_exec_assistant() -> Agent:
    return Agent(
        role="Executive Assistant",
        goal="Maximize executive productivity through intelligent scheduling, preparation, and communication",
        backstory=(
            "You are a world-class executive assistant who anticipates needs "
            "before they arise. You manage complex calendars, prepare briefing "
            "documents, plan travel logistics, triage communications, draft "
            "stakeholder correspondence, and ensure nothing falls through the "
            "cracks. You're proactive, discreet, and always three steps ahead."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_meeting_prep_task(meeting_details: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Prepare a comprehensive briefing for this meeting:\n\n"
            f"{meeting_details}\n\n"
            "Your briefing must include the following sections:\n"
            "1. **Meeting Context**: Purpose, desired outcomes, and strategic relevance\n"
            "2. **Attendee Research**: Key background on each participant — role, "
            "interests, recent work, and likely concerns\n"
            "3. **Proposed Agenda**: Time-boxed agenda with clear objectives per item\n"
            "4. **Talking Points**: Key messages the executive should communicate\n"
            "5. **Potential Objections & Responses**: Anticipated pushback and "
            "prepared counter-arguments\n"
            "6. **Pre-Meeting Reading**: Curated list of 3-5 documents/articles to review\n"
            "7. **One-Page Summary**: An executive summary (max 300 words) suitable "
            "for a quick glance\n\n"
            "Tailor the depth and tone of the briefing to the seniority of attendees "
            "and the strategic importance of the meeting. Flag any risks or sensitive topics."
        ),
        expected_output=(
            "A structured meeting briefing document with Context, Attendee Research, "
            "Proposed Agenda, Talking Points, Objections & Responses, "
            "Pre-Meeting Reading, and One-Page Summary sections"
        ),
        agent=agent,
    )


def create_priority_triage_task(items: str, agent: Agent) -> Task:
    return Task(
        description=(
            "Triage and prioritize these items from the executive's inbox:\n\n"
            f"{items}\n\n"
            "Process each item through the following framework:\n"
            "1. **Eisenhower Matrix**: Classify as Urgent/Important, Important/Not Urgent, "
            "Urgent/Not Important, or Neither\n"
            "2. **Time Blocking**: Assign each actionable item to a suggested time block "
            "(morning deep work, midday batch, afternoon review, or delegate)\n"
            "3. **Delegation Plan**: Identify items that can be delegated, with a "
            "recommended person/team and brief delegation message\n"
            "4. **Elimination Candidates**: Flag items that can be safely ignored, "
            "archived, or auto-responded to — with justification\n"
            "5. **Quick Responses**: For items requiring a reply but minimal thought, "
            "draft a 1-2 sentence response\n"
            "6. **Risk Flags**: Highlight any items with legal, financial, or "
            "reputational risk if not handled promptly\n\n"
            "Output a structured table/matrix, not a narrative essay."
        ),
        expected_output=(
            "A prioritized action matrix with Eisenhower classification, time blocks, "
            "delegation plans, elimination candidates, quick response drafts, "
            "and risk flags"
        ),
        agent=agent,
    )


def create_travel_plan_task(destination: str, dates: str, purpose: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Plan a comprehensive business trip to {destination} from {dates} "
            f"for the following purpose: {purpose}.\n\n"
            "Research and compile a complete travel package:\n"
            "1. **Flight Options**: 2-3 recommended itineraries with airline, "
            "departure/arrival times, layover details, and price estimate\n"
            "2. **Hotel Recommendations**: 3 options ranked by proximity to "
            "meeting venues, with price tier (budget/mid-range/luxury), "
            "and key amenities\n"
            "3. **Airport Transfer**: Ground transport from airport to hotel "
            "(taxi/rideshare/transit) with time estimate\n"
            "4. **Daily Itinerary**: Day-by-day schedule with time blocks for "
            "meetings, meals, transit, and buffer time\n"
            "5. **Visa & Entry Requirements**: Current visa requirements for the "
            "executive's passport (assume US passport if not specified)\n"
            "6. **Weather Forecast**: Expected weather during the trip dates "
            "with clothing recommendations\n"
            "7. **Dining**: 3 restaurant recommendations for business dinners — "
            "one casual, one upscale, one local experience\n"
            "8. **Cultural Notes**: Key business etiquette norms for {destination} "
            "(greetings, punctuality, gift-giving, tipping)\n"
            "9. **Travel Advisories**: Any current health, safety, or security alerts\n"
            "10. **Packing Checklist**: Tailored to weather, activities, and "
            "business dress requirements\n\n"
            "Use web search where needed to provide current, accurate information."
        ),
        expected_output=(
            "A complete travel itinerary document with flight options, hotels, "
            "daily schedule, visa info, weather, dining, cultural notes, "
            "advisories, and packing checklist"
        ),
        agent=agent,
    )


def create_email_draft_task(
    email_context: str,
    tone: str,
    key_points: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            "Draft a professional executive email based on the following brief:\n\n"
            f"**Context**: {email_context}\n"
            f"**Desired Tone**: {tone}\n"
            f"**Key Points to Cover**: {key_points}\n\n"
            "Produce a polished email draft that:\n"
            "1. Opens with an appropriate greeting and context-setting first line\n"
            "2. Covers all key points concisely and logically\n"
            "3. Matches the requested tone precisely "
            "(formal/diplomatic/assertive/empathetic/friendly)\n"
            "4. Includes a clear call-to-action or next step\n"
            "5. Closes with an appropriate sign-off\n"
            "6. Is free of jargon, clichés, and unnecessary filler\n"
            "7. Remains under 200 words unless the context demands otherwise\n\n"
            "Format your output as:\n"
            "**Subject:** [suggested subject line]\n\n"
            "**Draft:**\n"
            "[email body]\n\n"
            "**Notes:** [2-3 sentences on framing choices and any content "
            "the executive should review carefully before sending]"
        ),
        expected_output=(
            "A structured email draft with subject line, body copy, framing notes, "
            "and word count — ready for executive review"
        ),
        agent=agent,
    )


def create_follow_up_task(
    meeting_notes: str,
    attendees: str,
    agent: Agent,
) -> Task:
    return Task(
        description=(
            "Generate comprehensive meeting follow-ups from these notes:\n\n"
            f"{meeting_notes}\n\n"
            f"Attendees:\n{attendees}\n\n"
            "Produce the following:\n"
            "1. **Decisions Made**: Bullet list of all confirmed decisions with "
            "rationale\n"
            "2. **Action Items**: Table with columns: Owner, Task, Deadline, Priority "
            "(P0/P1/P2)\n"
            "3. **Follow-Up Email to All Attendees**: Professional summary email "
            "covering decisions and action items\n"
            "4. **Individual Follow-Ups**: Separate brief messages for attendees "
            "with P0 action items, with specific requests and deadlines\n"
            "5. **Parking Lot**: Items discussed but deferred — with suggested "
            "follow-up dates\n"
            "6. **Risks & Blockers**: Any concerns raised that need escalation "
            "or monitoring"
        ),
        expected_output=(
            "Meeting follow-up package with decisions, action table, group email, "
            "individual follow-ups, parking lot, and risks/blockers"
        ),
        agent=agent,
    )


class ExecutiveAssistantWorkflow:
    """Executive assistant workflow orchestrator."""

    def __init__(self):
        self.agent = create_exec_assistant()

    def prep_meeting(self, meeting_details: str) -> str:
        """Generate a comprehensive meeting briefing."""
        task = create_meeting_prep_task(meeting_details, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def triage(self, items: str) -> str:
        """Triage and prioritize inbox items."""
        task = create_priority_triage_task(items, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def plan_travel(self, destination: str, dates: str, purpose: str) -> str:
        """Plan a comprehensive business trip."""
        task = create_travel_plan_task(destination, dates, purpose, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def draft_email(self, email_context: str, tone: str, key_points: str) -> str:
        """Draft a professional executive email."""
        task = create_email_draft_task(email_context, tone, key_points, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def follow_up(self, meeting_notes: str, attendees: str) -> str:
        """Generate meeting follow-ups and action items."""
        task = create_follow_up_task(meeting_notes, attendees, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        return crew.kickoff()

    def full_briefing(self, meeting_details: str, email_items: str) -> dict:
        """Run meeting prep + priority triage in parallel."""
        meeting_task = create_meeting_prep_task(meeting_details, self.agent)
        triage_task = create_priority_triage_task(email_items, self.agent)
        crew = Crew(
            agents=[self.agent],
            tasks=[meeting_task, triage_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
