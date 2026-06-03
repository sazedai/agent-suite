"""Tests for Customer Support agent."""

from unittest.mock import patch, MagicMock

import pytest


# ── Module-level imports ────────────────────────────────────────────────────────

def test_customer_support_module_imports():
    """Customer support module should import all components."""
    from agents.customer_support import (
        create_support_agent,
        create_ticket_triage_task,
        create_response_draft_task,
        create_escalation_handling_task,
        create_escalation_classification_task,
        create_faq_management_task,
        create_faq_answer_task,
        create_support_analytics_task,
        create_satisfaction_prediction_task,
        CustomerSupportWorkflow,
    )
    assert create_support_agent is not None
    assert create_ticket_triage_task is not None
    assert create_response_draft_task is not None
    assert create_escalation_handling_task is not None
    assert create_escalation_classification_task is not None
    assert create_faq_management_task is not None
    assert create_faq_answer_task is not None
    assert create_support_analytics_task is not None
    assert create_satisfaction_prediction_task is not None
    assert CustomerSupportWorkflow is not None


# ── Function signatures ────────────────────────────────────────────────────────

class TestFunctionSignatures:
    """Verify all task factory functions have the expected parameters."""

    def test_ticket_triage_signature(self):
        import inspect
        from agents.customer_support import create_ticket_triage_task
        sig = inspect.signature(create_ticket_triage_task)
        assert "tickets" in sig.parameters
        assert "agent" in sig.parameters

    def test_response_draft_signature(self):
        import inspect
        from agents.customer_support import create_response_draft_task
        sig = inspect.signature(create_response_draft_task)
        assert "ticket" in sig.parameters
        assert "company_info" in sig.parameters
        assert "agent" in sig.parameters

    def test_escalation_handling_signature(self):
        import inspect
        from agents.customer_support import create_escalation_handling_task
        sig = inspect.signature(create_escalation_handling_task)
        assert "ticket" in sig.parameters
        assert "escalation_policy" in sig.parameters
        assert "agent" in sig.parameters

    def test_escalation_classification_signature(self):
        import inspect
        from agents.customer_support import create_escalation_classification_task
        sig = inspect.signature(create_escalation_classification_task)
        assert "ticket" in sig.parameters
        assert "agent" in sig.parameters

    def test_faq_management_signature(self):
        import inspect
        from agents.customer_support import create_faq_management_task
        sig = inspect.signature(create_faq_management_task)
        assert "faq_entries" in sig.parameters
        assert "action" in sig.parameters
        assert "agent" in sig.parameters

    def test_faq_answer_signature(self):
        import inspect
        from agents.customer_support import create_faq_answer_task
        sig = inspect.signature(create_faq_answer_task)
        assert "question" in sig.parameters
        assert "faq_database" in sig.parameters
        assert "agent" in sig.parameters

    def test_support_analytics_signature(self):
        import inspect
        from agents.customer_support import create_support_analytics_task
        sig = inspect.signature(create_support_analytics_task)
        assert "ticket_log" in sig.parameters
        assert "period" in sig.parameters
        assert "agent" in sig.parameters

    def test_satisfaction_prediction_signature(self):
        import inspect
        from agents.customer_support import create_satisfaction_prediction_task
        sig = inspect.signature(create_satisfaction_prediction_task)
        assert "ticket_thread" in sig.parameters
        assert "agent" in sig.parameters


# ── Test descriptions contain expected keywords ────────────────────────────────

class TestTaskDescriptionContent:
    """Verify task descriptions are comprehensive and well-structured."""

    @pytest.fixture
    def agent(self):
        """Minimal mock agent for task creation."""
        return None

    def test_triage_description_mentions_severity(self, agent):
        from agents.customer_support import create_ticket_triage_task
        task = create_ticket_triage_task("test ticket", agent)
        assert "severity" in task.description.lower()

    def test_triage_description_mentions_category(self, agent):
        from agents.customer_support import create_ticket_triage_task
        task = create_ticket_triage_task("test ticket\nid: T-001\nsubject: login issue", agent)
        assert "category" in task.description.lower()

    def test_triage_description_mentions_sentiment(self, agent):
        from agents.customer_support import create_ticket_triage_task
        task = create_ticket_triage_task("test ticket", agent)
        assert "sentiment" in task.description.lower()

    def test_triage_description_mentions_escalation(self, agent):
        from agents.customer_support import create_ticket_triage_task
        task = create_ticket_triage_task("test ticket", agent)
        assert "escalat" in task.description.lower()

    def test_response_description_mentions_empathy(self, agent):
        from agents.customer_support import create_response_draft_task
        task = create_response_draft_task("test", "company info", agent)
        assert "empath" in task.description.lower()

    def test_response_description_mentions_resolution(self, agent):
        from agents.customer_support import create_response_draft_task
        task = create_response_draft_task("test", "company info", agent)
        assert "resolution" in task.description.lower()

    def test_escalation_handling_mentions_root_cause(self, agent):
        from agents.customer_support import create_escalation_handling_task
        task = create_escalation_handling_task("test ticket", "policy: tier2", agent)
        assert "root cause" in task.description.lower()

    def test_escalation_handling_mentions_prevention(self, agent):
        from agents.customer_support import create_escalation_handling_task
        task = create_escalation_handling_task("test ticket", "policy: tier2", agent)
        assert "prevention" in task.description.lower()

    def test_classification_mentions_tier_levels(self, agent):
        from agents.customer_support import create_escalation_classification_task
        task = create_escalation_classification_task("test ticket", agent)
        assert "tier" in task.description.lower()

    def test_faq_management_supports_actions(self, agent):
        from agents.customer_support import create_faq_management_task
        task = create_faq_management_task("Q: How? A: Like this.", "audit", agent)
        desc_lower = task.description.lower()
        assert "audit" in desc_lower
        assert "organize" in desc_lower or "update" in desc_lower

    def test_faq_answer_returns_no_match_guidance(self, agent):
        from agents.customer_support import create_faq_answer_task
        task = create_faq_answer_task("random question", "Q: Foo? A: Bar.", agent)
        assert "NO_MATCH" in task.description

    def test_analytics_description_covers_key_metrics(self, agent):
        from agents.customer_support import create_support_analytics_task
        task = create_support_analytics_task("tickets log", "last_30_days", agent)
        desc_lower = task.description.lower()
        assert "response time" in desc_lower or "resolution" in desc_lower

    def test_satisfaction_predicts_csat(self, agent):
        from agents.customer_support import create_satisfaction_prediction_task
        task = create_satisfaction_prediction_task("thread content", agent)
        assert "CSAT" in task.description or "satisfaction" in task.description.lower()


# ── Workflow class tests ──────────────────────────────────────────────────────

class TestCustomerSupportWorkflow:
    """Test the CustomerSupportWorkflow class structure."""

    @patch("agents.customer_support.get_llm")
    def test_workflow_instantiation(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import CustomerSupportWorkflow
        workflow = CustomerSupportWorkflow()
        assert workflow.agent is not None

    def test_workflow_has_triage_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "triage_tickets")
        assert callable(getattr(CustomerSupportWorkflow, "triage_tickets"))

    def test_workflow_has_draft_response_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "draft_response")

    def test_workflow_has_escalation_handling_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "handle_escalation")

    def test_workflow_has_escalation_classification_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "classify_escalation")

    def test_workflow_has_faq_management_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "manage_faq")

    def test_workflow_has_faq_answer_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "answer_faq")

    def test_workflow_has_analytics_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "analyze_support")

    def test_workflow_has_satisfaction_prediction_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "predict_satisfaction")

    def test_workflow_has_pipeline_method(self):
        from agents.customer_support import CustomerSupportWorkflow
        assert hasattr(CustomerSupportWorkflow, "run_pipeline")

    def test_draft_response_default_company_info(self):
        """draft_response should accept empty company_info by default."""
        from agents.customer_support import CustomerSupportWorkflow
        import inspect
        sig = inspect.signature(CustomerSupportWorkflow.draft_response)
        params = sig.parameters
        assert "company_info" in params
        assert params["company_info"].default == ""

    def test_manage_faq_default_action(self):
        """manage_faq should default to 'audit' action."""
        from agents.customer_support import CustomerSupportWorkflow
        import inspect
        sig = inspect.signature(CustomerSupportWorkflow.manage_faq)
        params = sig.parameters
        assert "action" in params
        assert params["action"].default == "audit"

    def test_analyze_support_default_period(self):
        """analyze_support should default to 'last_30_days'."""
        from agents.customer_support import CustomerSupportWorkflow
        import inspect
        sig = inspect.signature(CustomerSupportWorkflow.analyze_support)
        params = sig.parameters
        assert "period" in params
        assert params["period"].default == "last_30_days"

    def test_run_pipeline_signature(self):
        import inspect
        from agents.customer_support import CustomerSupportWorkflow
        sig = inspect.signature(CustomerSupportWorkflow.run_pipeline)
        expected_params = [
            "tickets", "company_info", "escalation_policy",
            "faq_entries", "ticket_log", "period",
        ]
        for p in expected_params:
            assert p in sig.parameters

    def test_workflow_default_period_in_pipeline(self):
        import inspect
        from agents.customer_support import CustomerSupportWorkflow
        sig = inspect.signature(CustomerSupportWorkflow.run_pipeline)
        assert sig.parameters["period"].default == "last_30_days"


# ── Expected outputs ───────────────────────────────────────────────────────────

class TestExpectedOutputs:
    """Verify expected outputs are descriptive and non-empty."""

    @pytest.fixture
    def agent(self):
        return None

    def test_triage_expected_output(self, agent):
        from agents.customer_support import create_ticket_triage_task
        task = create_ticket_triage_task("test", agent)
        assert len(task.expected_output) > 10
        assert "triage" in task.expected_output.lower()

    def test_escalation_handling_expected_output(self, agent):
        from agents.customer_support import create_escalation_handling_task
        task = create_escalation_handling_task("test", "policy", agent)
        assert "root cause" in task.expected_output.lower()
        assert "handoff" in task.expected_output.lower()

    def test_faq_audit_expected_output(self, agent):
        from agents.customer_support import create_faq_management_task
        task = create_faq_management_task("entries", "audit", agent)
        assert "faq" in task.expected_output.lower()

    def test_analytics_expected_output(self, agent):
        from agents.customer_support import create_support_analytics_task
        task = create_support_analytics_task("log", "week", agent)
        assert "metric" in task.expected_output.lower() or "analytics" in task.expected_output.lower()

    def test_satisfaction_expected_output(self, agent):
        from agents.customer_support import create_satisfaction_prediction_task
        task = create_satisfaction_prediction_task("thread", agent)
        output_lower = task.expected_output.lower()
        assert "csat" in output_lower or "satisfaction" in output_lower
        assert "churn" in output_lower


# ── Agent creation ────────────────────────────────────────────────────────────

class TestAgentCreation:
    """Test that a support agent can be created."""

    @patch("agents.customer_support.get_llm")
    def test_create_support_agent_returns_agent(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        from crewai import Agent
        agent = create_support_agent()
        assert isinstance(agent, Agent)

    @patch("agents.customer_support.get_llm")
    def test_agent_role(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        agent = create_support_agent()
        assert agent.role == "Customer Support Specialist"

    @patch("agents.customer_support.get_llm")
    def test_agent_has_llm(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        agent = create_support_agent()
        assert agent.llm is not None

    @patch("agents.customer_support.get_llm")
    def test_agent_backstory_contains_escalation_knowledge(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        agent = create_support_agent()
        assert "escalat" in agent.backstory.lower()

    @patch("agents.customer_support.get_llm")
    def test_agent_backstory_contains_faq_knowledge(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        agent = create_support_agent()
        assert "faq" in agent.backstory.lower() or "databas" in agent.backstory.lower()

    @patch("agents.customer_support.get_llm")
    def test_agent_allows_no_delegation(self, mock_llm):
        mock_llm.return_value = "gpt-4o"
        from agents.customer_support import create_support_agent
        agent = create_support_agent()
        assert agent.allow_delegation is False
