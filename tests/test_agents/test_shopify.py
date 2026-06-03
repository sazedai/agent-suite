"""Tests for Shopify Assistant agent and Shopify integration client.

Covers:
- Module imports for the enhanced shopify_assistant agent
- Agent creation and configuration
- Task factory functions (product, order, inventory, customer)
- Analysis helper functions (analyze_product_quality, analyze_orders,
  analyze_inventory)
- Integration client graceful behavior when credentials are missing
- Integration client URL/header construction
"""
from __future__ import annotations

import asyncio
import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the repo root is on sys.path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

def test_shopify_assistant_imports():
    """Shopify assistant module should export all enhanced functions."""
    from agents.shopify_assistant import (
        create_shopify_assistant,
        create_product_audit_task,
        create_product_pricing_analysis_task,
        create_product_seo_optimization_task,
        create_order_tracking_task,
        create_order_fulfillment_task,
        create_refund_analysis_task,
        create_inventory_alert_task,
        create_inventory_reorder_task,
        create_customer_insights_task,
        analyze_product_quality,
        analyze_orders,
        analyze_inventory,
        ShopifyAssistantWorkflow,
    )
    assert callable(create_shopify_assistant)
    assert callable(create_product_audit_task)
    assert callable(create_product_pricing_analysis_task)
    assert callable(create_product_seo_optimization_task)
    assert callable(create_order_tracking_task)
    assert callable(create_order_fulfillment_task)
    assert callable(create_refund_analysis_task)
    assert callable(create_inventory_alert_task)
    assert callable(create_inventory_reorder_task)
    assert callable(create_customer_insights_task)
    assert callable(analyze_product_quality)
    assert callable(analyze_orders)
    assert callable(analyze_inventory)
    assert ShopifyAssistantWorkflow is not None


def test_shopify_assistant_in_all_agents():
    """Shopify assistant should be importable from the agents package."""
    from agents import shopify_assistant
    assert shopify_assistant is not None


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

def test_create_shopify_assistant():
    """Agent should have the correct role, goal, and backstory."""
    with patch("agents.shopify_assistant.get_llm", return_value="gpt-4o"):
        from agents.shopify_assistant import create_shopify_assistant
        agent = create_shopify_assistant()

    assert agent.role == "Shopify Store Manager"
    assert "Shopify" in agent.goal
    assert "product" in agent.goal.lower()
    assert "inventory" in agent.goal.lower()
    assert "Shopify" in agent.backstory
    assert agent.verbose is True
    assert agent.allow_delegation is False


# ---------------------------------------------------------------------------
# Task factories — verify signatures and return types
# ---------------------------------------------------------------------------

def test_product_audit_task_signature():
    """Product audit task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_product_audit_task
    sig = inspect.signature(create_product_audit_task)
    assert "agent" in sig.parameters


def test_product_pricing_task_signature():
    """Pricing analysis task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_product_pricing_analysis_task
    sig = inspect.signature(create_product_pricing_analysis_task)
    assert "agent" in sig.parameters


def test_product_seo_task_signature():
    """SEO optimization task should accept agent and optional focus."""
    import inspect
    from agents.shopify_assistant import create_product_seo_optimization_task
    sig = inspect.signature(create_product_seo_optimization_task)
    assert "agent" in sig.parameters
    assert "focus" in sig.parameters


def test_order_tracking_task_signature():
    """Order tracking task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_order_tracking_task
    sig = inspect.signature(create_order_tracking_task)
    assert "agent" in sig.parameters


def test_order_fulfillment_task_signature():
    """Fulfillment task factory should accept agent and optional order_id."""
    import inspect
    from agents.shopify_assistant import create_order_fulfillment_task
    sig = inspect.signature(create_order_fulfillment_task)
    assert "agent" in sig.parameters
    assert "order_id" in sig.parameters


def test_refund_analysis_task_signature():
    """Refund analysis task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_refund_analysis_task
    sig = inspect.signature(create_refund_analysis_task)
    assert "agent" in sig.parameters


def test_inventory_alert_task_signature():
    """Inventory alert task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_inventory_alert_task
    sig = inspect.signature(create_inventory_alert_task)
    assert "agent" in sig.parameters


def test_inventory_reorder_task_signature():
    """Inventory reorder task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_inventory_reorder_task
    sig = inspect.signature(create_inventory_reorder_task)
    assert "agent" in sig.parameters


def test_customer_insights_task_signature():
    """Customer insights task factory should accept an agent parameter."""
    import inspect
    from agents.shopify_assistant import create_customer_insights_task
    sig = inspect.signature(create_customer_insights_task)
    assert "agent" in sig.parameters


# ---------------------------------------------------------------------------
# Task factories — produce Task objects (mocked agent)
# ---------------------------------------------------------------------------

class _MockAgent:
    """Minimal mock that satisfies Task construction."""
    def get(self, _key, _default=None):
        return _default

    def process_model_config(self, _cfg, _logger):
        return {}


@pytest.fixture
def mock_agent():
    return _MockAgent()


@patch("agents.shopify_assistant.Task")
def test_product_audit_task_returns_task(mock_task_cls, mock_agent):
    """Product audit task should return a Task."""
    from agents.shopify_assistant import create_product_audit_task
    mock_task_cls.return_value = MagicMock()
    result = create_product_audit_task(mock_agent)
    assert result is not None
    mock_task_cls.assert_called_once()
    call_kwargs = mock_task_cls.call_args[1] or {}
    call_args = mock_task_cls.call_args
    # Check that the description mentions audit concepts
    desc = ""
    if call_kwargs.get("description"):
        desc = call_kwargs["description"]
    elif len(call_args) > 1:
        desc = call_args[1]
    assert "audit" in desc.lower() or "catalog" in desc.lower()


@patch("agents.shopify_assistant.Task")
def test_product_seo_task_with_focus(mock_task_cls, mock_agent):
    """SEO task should include the focus area in its description."""
    from agents.shopify_assistant import create_product_seo_optimization_task
    mock_task_cls.return_value = MagicMock()
    result = create_product_seo_optimization_task(mock_agent, focus="electronics")
    assert result is not None
    call_kwargs = mock_task_cls.call_args[1] or {}
    desc = call_kwargs.get("description", "")
    assert "electronics" in desc.lower()


# ---------------------------------------------------------------------------
# Analysis helpers — analyze_product_quality
# ---------------------------------------------------------------------------

def test_analyze_product_quality_empty():
    """Empty product list should return zero counts."""
    from agents.shopify_assistant import analyze_product_quality
    result = analyze_product_quality([])
    assert result["total_products"] == 0
    assert result["missing_images"] == []
    assert result["empty_descriptions"] == []
    assert result["zero_inventory"] == []
    assert result["duplicate_names"] == []
    assert result["missing_seo"] == []


def test_analyze_product_quality_missing_images():
    """Products without images should be flagged."""
    from agents.shopify_assistant import analyze_product_quality
    products = [
        {"id": 1, "title": "Shirt", "images": [], "body_html": "<p>ok</p>", "variants": [{"inventory_quantity": 5}]},
        {"id": 2, "title": "Hat", "images": [{"src": "a.jpg"}], "body_html": "<p>ok</p>", "variants": [{"inventory_quantity": 5}]},
    ]
    result = analyze_product_quality(products)
    assert 1 in result["missing_images"]
    assert 2 not in result["missing_images"]


def test_analyze_product_quality_empty_descriptions():
    """Products with empty body_html should be flagged."""
    from agents.shopify_assistant import analyze_product_quality
    products = [
        {"id": 1, "title": "Shirt", "body_html": "", "images": [{}], "variants": [{"inventory_quantity": 5}]},
        {"id": 2, "title": "Hat", "body_html": "   ", "images": [{}], "variants": [{"inventory_quantity": 5}]},
        {"id": 3, "title": "Bag", "body_html": "<p>good</p>", "images": [{}], "variants": [{"inventory_quantity": 5}]},
    ]
    result = analyze_product_quality(products)
    assert 1 in result["empty_descriptions"]
    assert 2 in result["empty_descriptions"]
    assert 3 not in result["empty_descriptions"]


def test_analyze_product_quality_zero_inventory():
    """Products with zero total inventory should be flagged."""
    from agents.shopify_assistant import analyze_product_quality
    products = [
        {"id": 1, "title": "Shirt", "body_html": "<p>ok</p>", "images": [{}], "variants": [{"inventory_quantity": 0}]},
        {"id": 2, "title": "Hat", "body_html": "<p>ok</p>", "images": [{}], "variants": [{"inventory_quantity": 5}]},
    ]
    result = analyze_product_quality(products)
    assert 1 in result["zero_inventory"]
    assert 2 not in result["zero_inventory"]


def test_analyze_product_quality_multi_variant_inventory():
    """Inventory should be summed across all variants."""
    from agents.shopify_assistant import analyze_product_quality
    products = [
        {"id": 1, "title": "Shirt", "body_html": "<p>ok</p>", "images": [{}],
         "variants": [{"inventory_quantity": 3}, {"inventory_quantity": 2}]},
    ]
    result = analyze_product_quality(products)
    assert 1 not in result["zero_inventory"]  # total = 5


def test_analyze_product_quality_duplicates():
    """Case-insensitive duplicate product names should be detected."""
    from agents.shopify_assistant import analyze_product_quality
    products = [
        {"id": 1, "title": "T-Shirt", "body_html": "<p>ok</p>", "images": [{}], "variants": [{"inventory_quantity": 5}]},
        {"id": 2, "title": "t-shirt", "body_html": "<p>ok</p>", "images": [{}], "variants": [{"inventory_quantity": 5}]},
        {"id": 3, "title": "T-Shirt", "body_html": "<p>ok</p>", "images": [{}], "variants": [{"inventory_quantity": 5}]},
    ]
    result = analyze_product_quality(products)
    assert len(result["duplicate_names"]) >= 1


# ---------------------------------------------------------------------------
# Analysis helpers — analyze_orders
# ---------------------------------------------------------------------------

def test_analyze_orders_empty():
    """Empty order list should return zero counts."""
    from agents.shopify_assistant import analyze_orders
    result = analyze_orders([])
    assert result["total_orders"] == 0
    assert result["unfulfilled_old"] == []
    assert result["high_value"] == []
    assert result["fraud_flags"] == []
    assert result["repeat_customers"] == {}
    assert result["refund_patterns"] == []


def test_analyze_orders_unfulfilled():
    """Non-fulfilled orders should be flagged."""
    from agents.shopify_assistant import analyze_orders
    orders = [
        {"id": 1, "total_price": "50.00", "fulfillment_status": None, "email": "a@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
        {"id": 2, "total_price": "50.00", "fulfillment_status": "fulfilled", "email": "b@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
    ]
    result = analyze_orders(orders)
    assert 1 in result["unfulfilled_old"]
    assert 2 not in result["unfulfilled_old"]


def test_analyze_orders_high_value():
    """Orders over $500 should be flagged as high-value."""
    from agents.shopify_assistant import analyze_orders
    orders = [
        {"id": 1, "total_price": "50.00", "fulfillment_status": "fulfilled", "email": "a@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
        {"id": 2, "total_price": "750.00", "fulfillment_status": "fulfilled", "email": "b@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
    ]
    result = analyze_orders(orders)
    assert len(result["high_value"]) == 1
    assert result["high_value"][0]["id"] == 2
    assert result["high_value"][0]["total"] == 750.0


def test_analyze_orders_fraud_flags():
    """Orders with billing/shipping mismatches should be flagged."""
    from agents.shopify_assistant import analyze_orders
    orders = [
        {"id": 1, "total_price": "100.00", "fulfillment_status": None, "email": "a@x.com", "financial_status": "paid",
         "billing_address": {"city": "NYC", "country": "US"}, "shipping_address": {"city": "LA", "country": "US"}},
        {"id": 2, "total_price": "100.00", "fulfillment_status": None, "email": "b@x.com", "financial_status": "paid",
         "billing_address": {"city": "NYC", "country": "US"}, "shipping_address": {"city": "NYC", "country": "US"}},
    ]
    result = analyze_orders(orders)
    assert len(result["fraud_flags"]) == 1
    assert result["fraud_flags"][0]["id"] == 1


def test_analyze_orders_repeat_customers():
    """Customers with multiple orders should be identified."""
    from agents.shopify_assistant import analyze_orders
    orders = [
        {"id": 1, "total_price": "100.00", "fulfillment_status": None, "email": "vip@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
        {"id": 2, "total_price": "200.00", "fulfillment_status": None, "email": "vip@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
        {"id": 3, "total_price": "50.00", "fulfillment_status": None, "email": "new@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None},
    ]
    result = analyze_orders(orders)
    assert result["repeat_customers"] == {"vip@x.com": 2}


def test_analyze_orders_refunds():
    """Refunded orders should be tracked in refund_patterns."""
    from agents.shopify_assistant import analyze_orders
    orders = [
        {"id": 1, "total_price": "100.00", "fulfillment_status": None, "email": "a@x.com", "financial_status": "refunded",
         "billing_address": None, "shipping_address": None, "created_at": "2024-01-01"},
        {"id": 2, "total_price": "200.00", "fulfillment_status": None, "email": "b@x.com", "financial_status": "partially_refunded",
         "billing_address": None, "shipping_address": None, "created_at": "2024-01-02"},
        {"id": 3, "total_price": "50.00", "fulfillment_status": None, "email": "c@x.com", "financial_status": "paid",
         "billing_address": None, "shipping_address": None, "created_at": "2024-01-03"},
    ]
    result = analyze_orders(orders)
    assert len(result["refund_patterns"]) == 2


# ---------------------------------------------------------------------------
# Analysis helpers — analyze_inventory
# ---------------------------------------------------------------------------

def test_analyze_inventory_empty():
    """Empty product list should return empty categories."""
    from agents.shopify_assistant import analyze_inventory
    result = analyze_inventory([])
    assert result["out_of_stock"] == []
    assert result["low_stock"] == []
    assert result["healthy_stock"] == []
    assert result["total_tracked_inventory"] == 0


def test_analyze_inventory_categorization():
    """Products should be categorized by inventory level."""
    from agents.shopify_assistant import analyze_inventory
    products = [
        {"id": 1, "title": "Sold Out", "variants": [{"inventory_quantity": 0}]},
        {"id": 2, "title": "Running Low", "variants": [{"inventory_quantity": 3}]},
        {"id": 3, "title": "Well Stocked", "variants": [{"inventory_quantity": 50}]},
    ]
    result = analyze_inventory(products)
    assert len(result["out_of_stock"]) == 1
    assert result["out_of_stock"][0]["id"] == 1
    assert len(result["low_stock"]) == 1
    assert result["low_stock"][0]["id"] == 2
    assert result["low_stock"][0]["quantity"] == 3
    assert len(result["healthy_stock"]) == 1
    assert result["healthy_stock"][0]["id"] == 3


def test_analyze_inventory_total():
    """Total inventory should sum across all products."""
    from agents.shopify_assistant import analyze_inventory
    products = [
        {"id": 1, "title": "A", "variants": [{"inventory_quantity": 5}]},
        {"id": 2, "title": "B", "variants": [{"inventory_quantity": 10}]},
    ]
    result = analyze_inventory(products)
    assert result["total_tracked_inventory"] == 15


def test_analyze_inventory_boundary():
    """Inventory of exactly 10 should be healthy (not low)."""
    from agents.shopify_assistant import analyze_inventory
    products = [
        {"id": 1, "title": "At Boundary", "variants": [{"inventory_quantity": 10}]},
    ]
    result = analyze_inventory(products)
    assert len(result["low_stock"]) == 0
    assert len(result["healthy_stock"]) == 1


# ---------------------------------------------------------------------------
# Integration client — no credentials
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shopify_no_credentials():
    """All shopify functions should return empty values without credentials."""
    from core.integrations import shopify as shop

    # Ensure no credentials are set
    with patch.object(shop.settings, "shopify_api_key", ""),          patch.object(shop.settings, "shopify_store_url", ""):
        assert await shop.get_products() == []
        assert await shop.get_product(1) == {}
        assert await shop.create_product("x") == {}
        assert await shop.update_product(1, title="x") == {}
        assert await shop.delete_product(1) is False
        assert await shop.count_products() == 0
        assert await shop.get_orders() == []
        assert await shop.get_order(1) == {}
        assert await shop.fulfill_order(1, 2) == {}
        assert await shop.count_orders() == 0
        assert await shop.get_inventory() == []
        assert await shop.get_inventory_item(1) == {}
        assert await shop.set_inventory_level(1, 2, 10) == {}
        assert await shop.get_customers() == []
        assert await shop.get_customer(1) == {}
        assert await shop.search_customers("x") == []


def test_shopify_has_credentials_false():
    """_has_credentials should return False without settings."""
    from core.integrations.shopify import _has_credentials
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_api_key = ""
        mock_settings.shopify_store_url = ""
        assert _has_credentials() is False


def test_shopify_has_credentials_true():
    """_has_credentials should return True with both values set."""
    from core.integrations.shopify import _has_credentials
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_api_key = "shpat_abc123"
        mock_settings.shopify_store_url = "my-store.myshopify.com"
        assert _has_credentials() is True


def test_shopify_base_url_plain_domain():
    """_base_url should work with plain myshopify.com domain."""
    from core.integrations.shopify import _base_url
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_store_url = "my-store.myshopify.com"
        assert _base_url() == "https://my-store.myshopify.com/admin/api/2024-01"


def test_shopify_base_url_https_prefix():
    """_base_url should strip https:// prefix."""
    from core.integrations.shopify import _base_url
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_store_url = "https://my-store.myshopify.com"
        assert _base_url() == "https://my-store.myshopify.com/admin/api/2024-01"


def test_shopify_base_url_trailing_path():
    """_base_url should strip trailing /admin/api/... path."""
    from core.integrations.shopify import _base_url
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_store_url = "https://my-store.myshopify.com/admin/api/2023-10"
        assert _base_url() == "https://my-store.myshopify.com/admin/api/2024-01"


def test_shopify_headers():
    """_headers should include the access token."""
    from core.integrations.shopify import _headers
    with patch("core.integrations.shopify.settings") as mock_settings:
        mock_settings.shopify_api_key = "shpat_test123"
        headers = _headers()
        assert headers["X-Shopify-Access-Token"] == "shpat_test123"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Integration __init__ exports
# ---------------------------------------------------------------------------

def test_integrations_init_exports():
    """core.integrations should export all shopify functions."""
    from core.integrations import (
        get_products, get_product, create_product, update_product,
        delete_product, count_products,
        get_orders, get_order, fulfill_order, count_orders,
        get_inventory, get_inventory_item, set_inventory_level,
        get_customers, get_customer, search_customers,
        search_contacts, create_contact,
    )
    assert callable(get_products)
    assert callable(get_product)
    assert callable(create_product)
    assert callable(update_product)
    assert callable(delete_product)
    assert callable(count_products)
    assert callable(get_orders)
    assert callable(get_order)
    assert callable(fulfill_order)
    assert callable(count_orders)
    assert callable(get_inventory)
    assert callable(get_inventory_item)
    assert callable(set_inventory_level)
    assert callable(get_customers)
    assert callable(get_customer)
    assert callable(search_customers)


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

def test_workflow_init():
    """ShopifyAssistantWorkflow should initialize and create agent."""
    with patch("agents.shopify_assistant.get_llm", return_value="gpt-4o"):
        from agents.shopify_assistant import ShopifyAssistantWorkflow
        wf = ShopifyAssistantWorkflow()
    assert wf.agent is not None
    assert wf.agent.role == "Shopify Store Manager"


# ---------------------------------------------------------------------------
# Task description checks (content validation)
# ---------------------------------------------------------------------------

def test_product_audit_description_content():
    """Product audit task description should mention key audit areas."""
    from agents.shopify_assistant import create_product_audit_task
    import inspect
    source = inspect.getsource(create_product_audit_task)
    assert "image" in source.lower()
    assert "inventory" in source.lower()
    assert "descriptions" in source.lower()


def test_order_tracking_description_content():
    """Order tracking task description should mention fraud and fulfillment."""
    from agents.shopify_assistant import create_order_tracking_task
    import inspect
    source = inspect.getsource(create_order_tracking_task)
    assert "fulfill" in source.lower()
    assert "fraud" in source.lower()


def test_inventory_alert_description_content():
    """Inventory alert task description should mention low-stock and reorder."""
    from agents.shopify_assistant import create_inventory_alert_task
    import inspect
    source = inspect.getsource(create_inventory_alert_task)
    assert "low" in source.lower() and "stock" in source.lower()
    assert "reorder" in source.lower()
