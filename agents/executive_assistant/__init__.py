"""Executive Assistant Workflows.

Calendar management, meeting preparation, travel planning,
priority triage, and executive communications.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.tools import web_search


def create_exec_assistant() -> Agent:
    return Agent(
        role="Executive Assistant",
        goal="Maximize executive productivity through intelligent scheduling, preparation, and communication",
        backstory=(
            "You are a world-class executive assistant who anticipates needs "
            "before they arise. You manage complex calendars, prepare briefing "
            "documents, plan travel logistics, triage communications, and "
            "ensure nothing falls through the cracks. You're proactive, "
            "discreet, and always three steps ahead."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_meeting_prep_task(meeting_details: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Prepare a comprehensive briefing for this meeting:\n\n{meeting_details}\n\n"
            "Create: attendee background research, agenda suggestions, "
            "key talking points, potential objections and responses, "
            "pre-meeting reading list, and a one-page briefing document."
        ),
        expected_output="Meeting briefing document with attendee research, agenda, talking points, and pre-reading",
        agent=agent,
    )


def create_priority_triage_task(items: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Triage and prioritize these items from the executive's inbox:\n\n{items}\n\n"
            "Categorize by: urgent/important matrix (Eisenhower matrix), "
            "suggested time blocks for each, items that can be delegated, "
            "items that can be eliminated, and recommended response templates "
            "for quick replies."
        ),
        expected_output="Prioritized action matrix with time blocks, delegation plan, and quick response templates",
        agent=agent,
    )


def create_travel_plan_task(destination: str, dates: str, purpose: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Plan a business trip to {destination} during {dates} for {purpose}. "
            "Research: flight options, hotel recommendations near meeting venues, "
            "local transportation, visa requirements, weather forecast, "
            "restaurant options for business dinners, and a day-by-day itinerary. "
            "Search for current travel advisories and local business etiquette."
        ),
        expected_output="Complete travel itinerary with flights, hotels, transport, dining, and cultural notes",
        agent=agent,
    )


class ExecutiveAssistantWorkflow:
    """Executive assistant workflow."""

    def __init__(self):
        self.agent = create_exec_assistant()

    def prep_meeting(self, meeting_details: str) -> str:
        task = create_meeting_prep_task(meeting_details, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def triage(self, items: str) -> str:
        task = create_priority_triage_task(items, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def plan_travel(self, destination: str, dates: str, purpose: str) -> str:
        task = create_travel_plan_task(destination, dates, purpose, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
