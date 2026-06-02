"""Tests for the CRM + Inbox Management agent and HubSpot integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# ── HubSpot integration tests ────────────────────────────────────────────────

from core.integrations.hubspot import (
    search_contacts,
    create_contact,
    update_contact,
    get_contact,
    create_deal,
    update_deal_stage,
    list_deals,
    get_deal,
    list_pipelines,
    _headers,
    BASE_URL,
)


FAKE_HUBSPOT_KEY = "test-hubspot-key-123"


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch):
    """Patch settings to provide a fake HubSpot key for all tests."""
    from core import config
    monkeypatch.setattr(config.settings, "hubspot_api_key", FAKE_HUBSPOT_KEY)


class TestHubspotHeaders:
    """Tests for the _headers helper."""

    def test_headers_contains_bearer_token(self):
        headers = _headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {FAKE_HUBSPOT_KEY}"


class TestSearchContacts:
    """Tests for search_contacts."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await search_contacts("alice@example.com")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_contacts_from_response(self):
        mock_results = {
            "results": [
                {"properties": {"email": "alice@example.com", "firstname": "Alice"}},
                {"properties": {"email": "bob@example.com", "firstname": "Bob"}},
            ]
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = mock_results
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await search_contacts("alice")

        assert len(result) == 2
        assert result[0]["email"] == "alice@example.com"
        assert result[1]["firstname"] == "Bob"

    @pytest.mark.asyncio
    async def test_sends_correct_search_body(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            await search_contacts("test@co.com", limit=5)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["limit"] == 5
        assert "test@co.com" in str(body)


class TestCreateContact:
    """Tests for create_contact."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await create_contact("test@example.com")
        assert result == {}

    @pytest.mark.asyncio
    async def test_creates_contact_with_all_fields(self):
        fake_response = {"id": "123", "properties": {"email": "a@b.com"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake_response

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await create_contact(
                email="a@b.com",
                firstname="A",
                lastname="B",
                company="Co",
                phone="+123",
                lifecyclestage="lead",
            )

        assert result == fake_response
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        props = body["properties"]
        assert props["email"] == "a@b.com"
        assert props["firstname"] == "A"
        assert props["lastname"] == "B"
        assert props["company"] == "Co"
        assert props["phone"] == "+123"
        assert props["lifecyclestage"] == "lead"

    @pytest.mark.asyncio
    async def test_omits_empty_optional_fields(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {"id": "1"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            await create_contact(email="x@y.com")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        props = body["properties"]
        assert set(props.keys()) == {"email"}


class TestUpdateContact:
    """Tests for update_contact."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await update_contact("123", {"firstname": "New"})
        assert result == {}

    @pytest.mark.asyncio
    async def test_patches_contact(self):
        fake_response = {"id": "123", "properties": {"firstname": "New"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake_response

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.patch = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await update_contact("123", {"firstname": "New"})

        assert result == fake_response
        mock_client.patch.assert_called_once()


class TestGetContact:
    """Tests for get_contact."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await get_contact("123")
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_properties(self):
        fake = {"properties": {"email": "a@b.com", "firstname": "A"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await get_contact("123")

        assert result["email"] == "a@b.com"

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await get_contact("nonexistent")

        assert result == {}


class TestCreateDeal:
    """Tests for create_deal."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await create_deal("Test Deal")
        assert result == {}

    @pytest.mark.asyncio
    async def test_creates_deal_with_defaults(self):
        fake = {"id": "deal-1", "properties": {"dealname": "Test"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.put = AsyncMock(return_value=MagicMock(spec=httpx.Response))

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await create_deal("Test Deal")

        assert result["id"] == "deal-1"
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        props = body["properties"]
        assert props["dealname"] == "Test Deal"
        assert props["pipeline"] == "default"
        assert props["dealstage"] == "appointmentscheduled"

    @pytest.mark.asyncio
    async def test_associates_contact_and_company(self):
        fake = {"id": "deal-1"}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_assoc = MagicMock(spec=httpx.Response)
        mock_client.put = AsyncMock(return_value=mock_assoc)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            await create_deal("Deal", contact_id="c1", company_id="comp1")

        # Should have 2 PUT calls for associations
        assert mock_client.put.call_count == 2


class TestUpdateDealStage:
    """Tests for update_deal_stage."""

    @pytest.mark.asyncio
    async def test_patches_deal_stage(self):
        fake = {"id": "deal-1", "properties": {"dealstage": "qualified"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.patch = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await update_deal_stage("deal-1", "qualified")

        assert result["id"] == "deal-1"


class TestListDeals:
    """Tests for list_deals."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await list_deals()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_deals(self):
        fake = {
            "results": [
                {"properties": {"dealname": "D1", "amount": "1000"}},
                {"properties": {"dealname": "D2", "amount": "2000"}},
            ]
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await list_deals(limit=10)

        assert len(result) == 2
        assert result[0]["dealname"] == "D1"

    @pytest.mark.asyncio
    async def test_filters_by_pipeline(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {"results": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            await list_deals(pipeline="custom-pipeline")

        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "filterGroups" in body


class TestGetDeal:
    """Tests for get_deal."""

    @pytest.mark.asyncio
    async def test_returns_deal_properties(self):
        fake = {"properties": {"dealname": "D1", "amount": "500"}}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await get_deal("deal-1")

        assert result["dealname"] == "D1"

    @pytest.mark.asyncio
    async def test_returns_empty_on_404(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await get_deal("nonexistent")

        assert result == {}


class TestListPipelines:
    """Tests for list_pipelines."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self, monkeypatch):
        from core import config
        monkeypatch.setattr(config.settings, "hubspot_api_key", "")
        result = await list_pipelines()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_pipelines_with_stages(self):
        fake = {
            "results": [
                {
                    "id": "default",
                    "label": "Sales Pipeline",
                    "stages": [
                        {"id": "appointmentscheduled", "label": "Appointment"},
                        {"id": "qualifiedtobuy", "label": "Qualified"},
                    ],
                }
            ]
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = fake

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("core.integrations.hubspot.httpx.AsyncClient", return_value=mock_client):
            result = await list_pipelines()

        assert len(result) == 1
        assert result[0]["id"] == "default"
        assert result[0]["label"] == "Sales Pipeline"
        assert len(result[0]["stages"]) == 2
        assert result[0]["stages"][0]["label"] == "Appointment"


# ── CRM + Inbox Agent tests ─────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

import pytest

from core import config as _config_module
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation


class _FakeLLM(BaseLLM):
    """A minimal LLM mock that passes CrewAI Agent validation."""

    model_name: str = "mock-model"
    temperature: float = 0.7
    stop: list = []

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def _llm_type(self):
        return "mock"

    @property
    def _identifying_params(self):
        return {}

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        return LLMResult(generations=[[Generation(text="mock response")]])


_fake_llm_instance = _FakeLLM()


_fake_llm_instance = _FakeLLM()

# We need to mock the Agent class to bypass strict Pydantic llm validation
# which rejects even valid BaseLLM subclasses.  The mock stores constructor
# args so tests can inspect role, goal, backstory, etc.
_mock_agents: list = []


class _MockAgent:
    """Drop-in replacement for CrewAI Agent that skips Pydantic validation."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        _mock_agents.append(self)

    def get(self, key, default=None):
        """Dict-like access for CrewAI Task compatibility."""
        return getattr(self, key, default)


class _MockTask:
    """Drop-in replacement for CrewAI Task."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


with patch("core.llm.get_llm", return_value=_fake_llm_instance), \
     patch("agents.crm_inbox.Agent", _MockAgent), \
     patch("agents.crm_inbox.Task", _MockTask):
    from agents.crm_inbox import (
        create_crm_agent,
        create_contact_sync_task,
        create_email_triage_task,
        create_pipeline_report_task,
        create_follow_up_task,
        CRMInboxWorkflow,
    )


@pytest.fixture(autouse=True)
def _mock_agent():
    """Ensure Agent, Task, and get_llm stay mocked throughout the test session."""
    with patch("core.llm.get_llm", return_value=_fake_llm_instance), \
         patch("agents.crm_inbox.Agent", _MockAgent), \
         patch("agents.crm_inbox.Task", _MockTask):
        yield


class TestCreateCrmAgent:
    """Tests for the CRM agent factory."""

    def test_agent_role(self):
        agent = create_crm_agent()
        assert agent.role == "CRM + Inbox Manager"

    def test_agent_goal_contains_pipeline(self):
        agent = create_crm_agent()
        assert "pipeline" in agent.goal.lower()

    def test_agent_backstory_mentions_hubspot(self):
        agent = create_crm_agent()
        assert "hubspot" in agent.backstory.lower()

    def test_agent_backstory_mentions_email_triage(self):
        agent = create_crm_agent()
        assert "email" in agent.backstory.lower() or "triage" in agent.backstory.lower()

    def test_agent_allows_no_delegation(self):
        agent = create_crm_agent()
        assert agent.allow_delegation is False


class TestContactSyncTask:
    """Tests for create_contact_sync_task."""

    def test_task_description_includes_count(self):
        agent = create_crm_agent()
        contacts = [{"email": "a@b.com"}, {"email": "c@d.com"}]
        task = create_contact_sync_task(contacts, agent)
        assert "2 contacts" in task.description

    def test_task_mentions_deduplication(self):
        agent = create_crm_agent()
        task = create_contact_sync_task([{"email": "a@b.com"}], agent)
        assert "deduplicate" in task.description.lower() or "duplicate" in task.description.lower()

    def test_task_mentions_search_contacts(self):
        agent = create_crm_agent()
        task = create_contact_sync_task([{"email": "a@b.com"}], agent)
        assert "search_contacts" in task.description

    def test_task_expected_output_mentions_conflicts(self):
        agent = create_crm_agent()
        task = create_contact_sync_task([{"email": "a@b.com"}], agent)
        assert "conflict" in task.expected_output.lower()


class TestEmailTriageTask:
    """Tests for create_email_triage_task."""

    def test_task_includes_email_content(self):
        agent = create_crm_agent()
        emails = "From: bob@example.com\nSubject: Urgent issue"
        task = create_email_triage_task(emails, agent)
        assert "bob@example.com" in task.description

    def test_task_classifies_urgency(self):
        agent = create_crm_agent()
        task = create_email_triage_task("test email", agent)
        assert "urgency" in task.description.lower()

    def test_task_classifies_category(self):
        agent = create_crm_agent()
        task = create_email_triage_task("test email", agent)
        assert "category" in task.description.lower()

    def test_task_expected_output_has_table(self):
        agent = create_crm_agent()
        task = create_email_triage_task("test email", agent)
        assert "triage" in task.expected_output.lower()

    def test_task_mentions_crm_actions(self):
        agent = create_crm_agent()
        task = create_email_triage_task("test email", agent)
        assert "crm" in task.description.lower() or "deal" in task.description.lower()


class TestPipelineReportTask:
    """Tests for create_pipeline_report_task."""

    def test_task_mentions_list_pipelines(self):
        agent = create_crm_agent()
        task = create_pipeline_report_task(agent)
        assert "list_pipelines" in task.description

    def test_task_mentions_list_deals(self):
        agent = create_crm_agent()
        task = create_pipeline_report_task(agent)
        assert "list_deals" in task.description

    def test_task_expected_output_has_metrics(self):
        agent = create_crm_agent()
        task = create_pipeline_report_task(agent)
        assert "value" in task.expected_output.lower() or "metric" in task.expected_output.lower()


class TestFollowUpTask:
    """Tests for create_follow_up_task."""

    def test_task_includes_contact_count(self):
        agent = create_crm_agent()
        task = create_follow_up_task(["c1", "c2", "c3"], agent)
        assert "3" in task.description

    def test_task_mentions_get_contact(self):
        agent = create_crm_agent()
        task = create_follow_up_task(["c1"], agent)
        assert "get_contact" in task.description

    def test_task_expected_output_has_schedule(self):
        agent = create_crm_agent()
        task = create_follow_up_task(["c1"], agent)
        assert "schedule" in task.expected_output.lower()


class TestCRMInboxWorkflow:
    """Tests for the CRMInboxWorkflow class."""

    def test_workflow_creates_agent(self):
        workflow = CRMInboxWorkflow()
        assert workflow.agent is not None
        assert workflow.agent.role == "CRM + Inbox Manager"

    def test_workflow_has_sync_contacts(self):
        workflow = CRMInboxWorkflow()
        assert hasattr(workflow, "sync_contacts")

    def test_workflow_has_triage_emails(self):
        workflow = CRMInboxWorkflow()
        assert hasattr(workflow, "triage_emails")

    def test_workflow_has_pipeline_report(self):
        workflow = CRMInboxWorkflow()
        assert hasattr(workflow, "pipeline_report")

    def test_workflow_has_schedule_follow_ups(self):
        workflow = CRMInboxWorkflow()
        assert hasattr(workflow, "schedule_follow_ups")


# ── Integration __init__ exports ─────────────────────────────────────────────

class TestIntegrationExports:
    """Verify all new functions are exported from core.integrations."""

    def test_hubspot_functions_exported(self):
        from core import integrations
        assert hasattr(integrations, "update_contact")
        assert hasattr(integrations, "get_contact")
        assert hasattr(integrations, "create_deal")
        assert hasattr(integrations, "update_deal_stage")
        assert hasattr(integrations, "list_deals")
        assert hasattr(integrations, "get_deal")
        assert hasattr(integrations, "list_pipelines")

    def test_all_in_all_list(self):
        from core.integrations import __all__
        expected = [
            "search_contacts", "create_contact", "update_contact", "get_contact",
            "create_deal", "update_deal_stage", "list_deals", "get_deal",
            "associate_deal_contact", "associate_deal_company", "list_pipelines",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"
