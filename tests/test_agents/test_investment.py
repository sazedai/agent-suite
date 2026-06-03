"""Tests for Investment Analyst agent."""

import inspect

import pytest

from agents.investment_analyst import (
    DSE_BLUE_CHIPS,
    DSE_DATA_SOURCES,
    DSE_SECTORS,
    InvestmentAnalystWorkflow,
    _build_dse_context,
    create_investment_analyst,
    create_investment_thesis_task,
    create_market_outlook_task,
    create_portfolio_analysis_task,
    create_risk_assessment_task,
    create_stock_screening_task,
    create_dse_scanner_task,
)


# ---------------------------------------------------------------------------
# Module-level import tests
# ---------------------------------------------------------------------------


def test_investment_module_imports():
    """Investment analyst module should export all expected components."""
    assert create_investment_analyst is not None
    assert create_stock_screening_task is not None
    assert create_portfolio_analysis_task is not None
    assert create_market_outlook_task is not None
    assert create_risk_assessment_task is not None
    assert create_investment_thesis_task is not None
    assert create_dse_scanner_task is not None
    assert InvestmentAnalystWorkflow is not None


def test_investment_all_exports_present():
    """Module should have all public factory attributes."""
    from agents import investment_analyst as mod
    public_names = [
        "create_investment_analyst",
        "create_stock_screening_task",
        "create_portfolio_analysis_task",
        "create_market_outlook_task",
        "create_risk_assessment_task",
        "create_investment_thesis_task",
        "create_dse_scanner_task",
        "InvestmentAnalystWorkflow",
    ]
    for name in public_names:
        assert hasattr(mod, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# DSE constants and helpers
# ---------------------------------------------------------------------------


class TestDSEConstants:
    """Validate DSE market data constants."""

    def test_blue_chips_is_non_empty_list(self):
        assert isinstance(DSE_BLUE_CHIPS, list)
        assert len(DSE_BLUE_CHIPS) > 0

    def test_blue_chip_entries_have_required_fields(self):
        required = {"symbol", "name", "sector"}
        for stock in DSE_BLUE_CHIPS:
            assert required.issubset(stock.keys()), f"Missing fields in {stock}"
            assert isinstance(stock["symbol"], str) and stock["symbol"]
            assert isinstance(stock["name"], str) and stock["name"]

    def test_blue_chips_include_known_tickers(self):
        symbols = {s["symbol"] for s in DSE_BLUE_CHIPS}
        for expected in ("BRACBANK", "SQPHARMA", "EBL", "BSRMSTEEL"):
            assert expected in symbols, f"Missing blue-chip: {expected}"

    def test_sectors_is_dict_with_lists(self):
        assert isinstance(DSE_SECTORS, dict)
        assert len(DSE_SECTORS) > 0
        for sector, tickers in DSE_SECTORS.items():
            assert isinstance(sector, str)
            assert isinstance(tickers, list)
            assert len(tickers) > 0

    def test_sectors_cover_known_names(self):
        assert "Financial" in DSE_SECTORS
        assert "Pharmaceuticals" in DSE_SECTORS
        assert "BRACBANK" in DSE_SECTORS["Financial"]
        assert "SQPHARMA" in DSE_SECTORS["Pharmaceuticals"]

    def test_data_sources_present(self):
        assert isinstance(DSE_DATA_SOURCES, dict)
        assert "amarstock" in DSE_DATA_SOURCES
        assert "dsebd" in DSE_DATA_SOURCES


def test_build_dse_context_returns_string():
    """_build_dse_context should return a non-empty string."""
    ctx = _build_dse_context()
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_build_dse_context_contains_blue_chips():
    ctx = _build_dse_context()
    assert "BRACBANK" in ctx
    assert "SQPHARMA" in ctx


def test_build_dse_context_contains_sectors():
    ctx = _build_dse_context()
    assert "Financial" in ctx
    assert "Pharmaceuticals" in ctx


# ---------------------------------------------------------------------------
# Factory function signatures
# ---------------------------------------------------------------------------


class TestFactorySignatures:
    """Verify task factory function signatures match expectations."""

    def test_stock_screening_sig(self):
        sig = inspect.signature(create_stock_screening_task)
        params = set(sig.parameters.keys())
        assert "criteria" in params
        assert "agent" in params
        assert "market" in params
        assert "focus_sectors" in params
        assert "exclude_z_category" in params
        assert "max_pe" in params
        assert "min_dividend_yield" in params

    def test_portfolio_analysis_sig(self):
        sig = inspect.signature(create_portfolio_analysis_task)
        params = set(sig.parameters.keys())
        assert "holdings" in params
        assert "agent" in params
        assert "benchmark" in params
        assert "risk_free_rate" in params

    def test_market_outlook_sig(self):
        sig = inspect.signature(create_market_outlook_task)
        params = set(sig.parameters.keys())
        assert "agent" in params
        assert "scope" in params
        assert "period" in params
        assert "include_dse" in params

    def test_risk_assessment_sig(self):
        sig = inspect.signature(create_risk_assessment_task)
        params = set(sig.parameters.keys())
        assert "holdings" in params
        assert "agent" in params
        assert "market_data" in params
        assert "scenario" in params

    def test_investment_thesis_sig(self):
        sig = inspect.signature(create_investment_thesis_task)
        params = set(sig.parameters.keys())
        assert "symbol" in params
        assert "agent" in params
        assert "conviction" in params

    def test_dse_scanner_sig(self):
        sig = inspect.signature(create_dse_scanner_task)
        params = set(sig.parameters.keys())
        assert "agent" in params
        assert "scan_type" in params


# ---------------------------------------------------------------------------
# Workflow class tests
# ---------------------------------------------------------------------------


class TestInvestmentAnalystWorkflow:
    """Test the InvestmentAnalystWorkflow class structure."""

    def test_workflow_has_expected_methods(self):
        methods = {
            "screen_stocks",
            "analyze_portfolio",
            "market_outlook",
            "assess_risk",
            "investment_thesis",
            "scan_dse",
        }
        for method in methods:
            assert hasattr(InvestmentAnalystWorkflow, method), f"Missing method: {method}"

    def test_workflow_screen_stocks_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.screen_stocks)
        params = set(sig.parameters.keys())
        assert "criteria" in params
        assert "market" in params
        assert "focus_sectors" in params
        assert "exclude_z_category" in params
        assert "max_pe" in params
        assert "min_dividend_yield" in params

    def test_workflow_analyze_portfolio_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.analyze_portfolio)
        params = set(sig.parameters.keys())
        assert "holdings" in params
        assert "benchmark" in params
        assert "risk_free_rate" in params

    def test_workflow_market_outlook_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.market_outlook)
        params = set(sig.parameters.keys())
        assert "scope" in params
        assert "period" in params
        assert "include_dse" in params

    def test_workflow_assess_risk_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.assess_risk)
        params = set(sig.parameters.keys())
        assert "holdings" in params
        assert "market_data" in params
        assert "scenario" in params

    def test_workflow_investment_thesis_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.investment_thesis)
        params = set(sig.parameters.keys())
        assert "symbol" in params
        assert "conviction" in params

    def test_workflow_scan_dse_sig(self):
        sig = inspect.signature(InvestmentAnalystWorkflow.scan_dse)
        params = set(sig.parameters.keys())
        assert "scan_type" in params


# ---------------------------------------------------------------------------
# DSE Scanner scan_type coverage
# ---------------------------------------------------------------------------


class TestDSEScannerTypes:
    """Verify DSE scanner task function accepts scan_type parameter."""

    def test_scan_type_parameter_defaults_to_full(self):
        sig = inspect.signature(create_dse_scanner_task)
        assert "scan_type" in sig.parameters
        default = sig.parameters["scan_type"].default
        assert default == "full"

    def test_scan_type_parameter_exists(self):
        sig = inspect.signature(create_dse_scanner_task)
        param = sig.parameters["scan_type"]
        assert str(param.annotation) == "str"


# ---------------------------------------------------------------------------
# Risk Assessment scenarios
# ---------------------------------------------------------------------------


class TestRiskScenarios:
    """Ensure risk assessment covers documented scenarios."""

    def test_scenario_parameter_defaults_to_standard(self):
        sig = inspect.signature(create_risk_assessment_task)
        assert "scenario" in sig.parameters
        default = sig.parameters["scenario"].default
        assert default == "standard"

    def test_holdings_optional(self):
        """holdings parameter should accept None for market-level risk."""
        sig = inspect.signature(create_risk_assessment_task)
        param = sig.parameters["holdings"]
        annotation_str = str(param.annotation)
        assert "None" in annotation_str or "dict" in annotation_str


# ---------------------------------------------------------------------------
# Stock Screening filters
# ---------------------------------------------------------------------------


class TestStockScreening:
    """Test stock screening filter parameters."""

    def test_exclude_z_category_default_true(self):
        sig = inspect.signature(create_stock_screening_task)
        param = sig.parameters["exclude_z_category"]
        assert param.default is True

    def test_focus_sectors_optional(self):
        sig = inspect.signature(create_stock_screening_task)
        param = sig.parameters["focus_sectors"]
        annotation_str = str(param.annotation)
        assert "None" in annotation_str

    def test_market_default_dse(self):
        sig = inspect.signature(create_stock_screening_task)
        param = sig.parameters["market"]
        assert param.default == "DSE"


# ---------------------------------------------------------------------------
# Portfolio Analysis parameters
# ---------------------------------------------------------------------------


class TestPortfolioAnalysis:
    """Test portfolio analysis defaults and parameters."""

    def test_benchmark_default_dsex(self):
        sig = inspect.signature(create_portfolio_analysis_task)
        assert sig.parameters["benchmark"].default == "DSEX"

    def test_risk_free_rate_default(self):
        sig = inspect.signature(create_portfolio_analysis_task)
        assert sig.parameters["risk_free_rate"].default == 0.08


# ---------------------------------------------------------------------------
# Market Outlook parameters
# ---------------------------------------------------------------------------


class TestMarketOutlook:
    """Test market outlook generation parameters."""

    def test_scope_default_global(self):
        sig = inspect.signature(create_market_outlook_task)
        assert sig.parameters["scope"].default == "GLOBAL"

    def test_period_default_weekly(self):
        sig = inspect.signature(create_market_outlook_task)
        assert sig.parameters["period"].default == "weekly"

    def test_include_dse_default_true(self):
        sig = inspect.signature(create_market_outlook_task)
        assert sig.parameters["include_dse"].default is True
