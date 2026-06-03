"""Tests for Executive Assistant workflows."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

# Module under test — imported once at module level for signature tests
import agents.executive_assistant as ea_mod
from agents.executive_assistant import (
    create_exec_assistant,
    create_meeting_prep_task,
    create_priority_triage_task,
    create_travel_plan_task,
    create_email_draft_task,
    create_follow_up_task,
    ExecutiveAssistantWorkflow,
)


# ---------------------------------------------------------------------------
# Lightweight stand-in for crewai.Task to avoid pydantic agent validation
# ---------------------------------------------------------------------------


class _MockTask:
    """Captures Task constructor args without pydantic validation."""

    def __init__(self, description, expected_output, agent, **kwargs):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_exec_assistant_module_imports():
    """Executive assistant module should import all public components."""
    assert create_exec_assistant is not None
    assert create_meeting_prep_task is not None
    assert create_priority_triage_task is not None
    assert create_travel_plan_task is not None
    assert create_email_draft_task is not None
    assert create_follow_up_task is not None
    assert ExecutiveAssistantWorkflow is not None


# ---------------------------------------------------------------------------
# Factory signature tests
# ---------------------------------------------------------------------------


class TestFactorySignatures:
    """Task factory functions expose the expected call signatures."""

    def test_meeting_prep_task_signature(self):
        sig = inspect.signature(create_meeting_prep_task)
        assert set(sig.parameters) == {"meeting_details", "agent"}

    def test_priority_triage_task_signature(self):
        sig = inspect.signature(create_priority_triage_task)
        assert set(sig.parameters) == {"items", "agent"}

    def test_travel_plan_task_signature(self):
        sig = inspect.signature(create_travel_plan_task)
        assert set(sig.parameters) == {"destination", "dates", "purpose", "agent"}

    def test_email_draft_task_signature(self):
        sig = inspect.signature(create_email_draft_task)
        assert set(sig.parameters) == {"email_context", "tone", "key_points", "agent"}

    def test_follow_up_task_signature(self):
        sig = inspect.signature(create_follow_up_task)
        assert set(sig.parameters) == {"meeting_notes", "attendees", "agent"}


# ---------------------------------------------------------------------------
# Workflow method signatures
# ---------------------------------------------------------------------------


class TestWorkflowSignatures:
    """ExecutiveAssistantWorkflow methods expose expected call signatures."""

    def test_prep_meeting_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.prep_meeting)
        assert "meeting_details" in sig.parameters

    def test_triage_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.triage)
        assert "items" in sig.parameters

    def test_plan_travel_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.plan_travel)
        assert set(sig.parameters) == {"self", "destination", "dates", "purpose"}

    def test_draft_email_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.draft_email)
        assert set(sig.parameters) == {"self", "email_context", "tone", "key_points"}

    def test_follow_up_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.follow_up)
        assert set(sig.parameters) == {"self", "meeting_notes", "attendees"}

    def test_full_briefing_signature(self):
        sig = inspect.signature(ExecutiveAssistantWorkflow.full_briefing)
        assert "meeting_details" in sig.parameters
        assert "email_items" in sig.parameters


# ---------------------------------------------------------------------------
# Agent creation tests — patch get_llm at module level
# ---------------------------------------------------------------------------


class TestAgentCreation:
    """create_exec_assistant produces a properly configured CrewAI Agent."""

    def test_agent_role(self):
        with patch("agents.executive_assistant.get_llm", return_value="gpt-4o"):
            agent = create_exec_assistant()
        assert agent.role == "Executive Assistant"

    def test_agent_goal_contains_productivity(self):
        with patch("agents.executive_assistant.get_llm", return_value="gpt-4o"):
            agent = create_exec_assistant()
        assert "productivity" in agent.goal.lower()

    def test_agent_no_delegation(self):
        with patch("agents.executive_assistant.get_llm", return_value="gpt-4o"):
            agent = create_exec_assistant()
        assert agent.allow_delegation is False


# ---------------------------------------------------------------------------
# Task production tests — patch Task to avoid pydantic agent validation
# ---------------------------------------------------------------------------


class TestTaskCreation:
    """Task factories produce Task objects with populated descriptions."""

    @pytest.fixture(autouse=True)
    def _patch_task(self):
        """Replace crewai.Task with a lightweight mock in the module namespace."""
        with patch("agents.executive_assistant.Task", _MockTask):
            yield

    @pytest.fixture
    def mock_agent(self):
        return MagicMock()

    def test_meeting_prep_task_description(self, mock_agent):
        task = create_meeting_prep_task("Board meeting Q4 strategy", mock_agent)
        desc = task.description
        assert "Board meeting Q4 strategy" in desc
        assert "Attendee Research" in desc
        assert "Proposed Agenda" in desc
        assert "Meeting Context" in desc

    def test_meeting_prep_task_expected_output(self, mock_agent):
        task = create_meeting_prep_task("test", mock_agent)
        assert "structured meeting briefing" in task.expected_output.lower()
        assert task.agent is mock_agent

    def test_priority_triage_task_description(self, mock_agent):
        task = create_priority_triage_task("3 emails, 2 Slack DMs, 1 voicemail", mock_agent)
        desc = task.description
        assert "3 emails, 2 Slack DMs, 1 voicemail" in desc
        assert "Eisenhower Matrix" in desc
        assert "Time Blocking" in desc
        assert "Delegation Plan" in desc

    def test_priority_triage_task_expected_output(self, mock_agent):
        task = create_priority_triage_task("test", mock_agent)
        assert "prioritized action matrix" in task.expected_output.lower()
        assert task.agent is mock_agent

    def test_travel_plan_task_description(self, mock_agent):
        task = create_travel_plan_task("Tokyo", "2026-07-15 to 2026-07-20", "investor meetings", mock_agent)
        desc = task.description
        assert "Tokyo" in desc
        assert "investor meetings" in desc
        assert "Flight Options" in desc
        assert "Visa & Entry" in desc
        assert "Cultural Notes" in desc

    def test_travel_plan_task_expected_output(self, mock_agent):
        task = create_travel_plan_task("London", "next week", "conference", mock_agent)
        assert "travel itinerary" in task.expected_output.lower()
        assert task.agent is mock_agent

    def test_email_draft_task_description(self, mock_agent):
        task = create_email_draft_task(
            "Follow up with investor after pitch meeting",
            "diplomatic",
            "Thank them, reiterate key metrics, propose next call",
            mock_agent,
        )
        desc = task.description
        assert "Follow up with investor" in desc
        assert "diplomatic" in desc
        assert "reiterate key metrics" in desc
        assert "**Subject:**" in desc
        assert "**Draft:**" in desc

    def test_email_draft_task_expected_output(self, mock_agent):
        task = create_email_draft_task("ctx", "formal", "points", mock_agent)
        assert "email draft" in task.expected_output.lower()
        assert task.agent is mock_agent

    def test_follow_up_task_description(self, mock_agent):
        task = create_follow_up_task(
            "Discussed Q3 targets, agreed on 15% growth goal",
            "Alice (CEO), Bob (CFO), Carol (VP Sales)",
            mock_agent,
        )
        desc = task.description
        assert "Q3 targets" in desc
        assert "Alice (CEO)" in desc
        assert "Decisions Made" in desc
        assert "Action Items" in desc
        assert "Parking Lot" in desc

    def test_follow_up_task_expected_output(self, mock_agent):
        task = create_follow_up_task("notes", "attendees", mock_agent)
        assert "follow-up package" in task.expected_output.lower()
        assert task.agent is mock_agent


# ---------------------------------------------------------------------------
# Workflow instantiation tests
# ---------------------------------------------------------------------------


class TestWorkflowInstantiation:
    """ExecutiveAssistantWorkflow initialises correctly."""

    def test_workflow_creates_agent(self):
        with patch("agents.executive_assistant.get_llm", return_value="gpt-4o"):
            wf = ExecutiveAssistantWorkflow()
        assert wf.agent is not None
        assert wf.agent.role == "Executive Assistant"


# ---------------------------------------------------------------------------
# Workflow method delegation tests — patch Agent, Task, and Crew
# ---------------------------------------------------------------------------


class TestWorkflowMethods:
    """Each workflow method delegates to Crew.kickoff correctly."""

    @pytest.fixture(autouse=True)
    def _patch_all(self):
        with patch("agents.executive_assistant.get_llm", return_value="gpt-4o"), \
             patch("agents.executive_assistant.Task", _MockTask):
            yield

    @pytest.fixture
    def workflow(self):
        return ExecutiveAssistantWorkflow()

    @patch("agents.executive_assistant.Crew")
    def test_prep_meeting_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "briefing result"
        MockCrew.return_value = mock_crew

        result = workflow.prep_meeting("Board meeting")
        mock_crew.kickoff.assert_called_once()
        assert result == "briefing result"

    @patch("agents.executive_assistant.Crew")
    def test_triage_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "triage result"
        MockCrew.return_value = mock_crew

        result = workflow.triage("inbox items")
        mock_crew.kickoff.assert_called_once()
        assert result == "triage result"

    @patch("agents.executive_assistant.Crew")
    def test_plan_travel_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "travel plan"
        MockCrew.return_value = mock_crew

        result = workflow.plan_travel("Tokyo", "July 15-20", "investor meetings")
        mock_crew.kickoff.assert_called_once()
        assert result == "travel plan"

    @patch("agents.executive_assistant.Crew")
    def test_draft_email_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "email draft"
        MockCrew.return_value = mock_crew

        result = workflow.draft_email("investor follow-up", "diplomatic", "key metrics")
        mock_crew.kickoff.assert_called_once()
        assert result == "email draft"

    @patch("agents.executive_assistant.Crew")
    def test_follow_up_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "follow-up package"
        MockCrew.return_value = mock_crew

        result = workflow.follow_up("meeting notes", "Alice, Bob")
        mock_crew.kickoff.assert_called_once()
        assert result == "follow-up package"

    @patch("agents.executive_assistant.Crew")
    def test_full_briefing_calls_kickoff(self, MockCrew, workflow):
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = {"meeting": "brief", "triage": "matrix"}
        MockCrew.return_value = mock_crew

        result = workflow.full_briefing("board meeting", "inbox items")
        mock_crew.kickoff.assert_called_once()
        assert result == {"meeting": "brief", "triage": "matrix"}


# ---------------------------------------------------------------------------
# Description content quality tests
# ---------------------------------------------------------------------------


class TestDescriptionQuality:
    """Task descriptions contain the structured guidance expected by the spec."""

    @pytest.fixture(autouse=True)
    def _patch_task(self):
        with patch("agents.executive_assistant.Task", _MockTask):
            yield

    @pytest.fixture
    def mock_agent(self):
        return MagicMock()

    def test_meeting_prep_has_seven_sections(self, mock_agent):
        task = create_meeting_prep_task("test", mock_agent)
        desc = task.description
        for section in [
            "Meeting Context",
            "Attendee Research",
            "Proposed Agenda",
            "Talking Points",
            "Potential Objections",
            "Pre-Meeting Reading",
            "One-Page Summary",
        ]:
            assert section in desc, f"Missing section: {section}"

    def test_triage_has_six_sections(self, mock_agent):
        task = create_priority_triage_task("test", mock_agent)
        desc = task.description
        for section in [
            "Eisenhower Matrix",
            "Time Blocking",
            "Delegation Plan",
            "Elimination Candidates",
            "Quick Responses",
            "Risk Flags",
        ]:
            assert section in desc, f"Missing section: {section}"

    def test_travel_has_ten_sections(self, mock_agent):
        task = create_travel_plan_task("Paris", "Aug 1-5", "summit", mock_agent)
        desc = task.description
        for section in [
            "Flight Options",
            "Hotel Recommendations",
            "Airport Transfer",
            "Daily Itinerary",
            "Visa & Entry",
            "Weather Forecast",
            "Dining",
            "Cultural Notes",
            "Travel Advisories",
            "Packing Checklist",
        ]:
            assert section in desc, f"Missing section: {section}"

    def test_email_draft_has_format_instructions(self, mock_agent):
        task = create_email_draft_task("ctx", "formal", "points", mock_agent)
        desc = task.description
        assert "**Subject:**" in desc
        assert "**Draft:**" in desc
        assert "**Notes:**" in desc
        assert "under 200 words" in desc

    def test_follow_up_has_six_sections(self, mock_agent):
        task = create_follow_up_task("notes", "people", mock_agent)
        desc = task.description
        for section in [
            "Decisions Made",
            "Action Items",
            "Follow-Up Email",
            "Individual Follow-Ups",
            "Parking Lot",
            "Risks & Blockers",
        ]:
            assert section in desc, f"Missing section: {section}"
