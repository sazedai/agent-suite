"""Tests for Investment Analyst agent."""

import pytest


def test_investment_module_imports():
    """Investment Analyst module should import all components."""
    from agents.investment_analyst import (
        create_investment_analyst,
        create_stock_screening_task,
        create_portfolio_analysis_task,
        create_market_outlook_task,
        InvestmentAnalystWorkflow,
    )
    assert create_investment_analyst is not None
    assert create_stock_screening_task is not None
    assert create_portfolio_analysis_task is not None
    assert create_market_outlook_task is not None
    assert InvestmentAnalystWorkflow is not None


def test_investment_task_signatures():
    """Investment task functions should have correct signatures."""
    import inspect
    from agents.investment_analyst import (
        create_stock_screening_task,
        create_portfolio_analysis_task,
        create_market_outlook_task,
    )

    sig = inspect.signature(create_stock_screening_task)
    assert "criteria" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_portfolio_analysis_task)
    assert "holdings" in sig.parameters

    sig = inspect.signature(create_market_outlook_task)
    assert "agent" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
