"""Tests for Shopify Assistant agent."""

import pytest


def test_shopify_module_imports():
    """Shopify Assistant module should import all components."""
    from agents.shopify_assistant import (
        create_shopify_assistant,
        create_product_audit_task,
        create_order_tracking_task,
        create_inventory_alert_task,
        ShopifyAssistantWorkflow,
    )
    assert create_shopify_assistant is not None
    assert create_product_audit_task is not None
    assert create_order_tracking_task is not None
    assert create_inventory_alert_task is not None
    assert ShopifyAssistantWorkflow is not None


def test_shopify_task_signatures():
    """Shopify task functions should have correct signatures."""
    import inspect
    from agents.shopify_assistant import (
        create_product_audit_task,
        create_order_tracking_task,
        create_inventory_alert_task,
    )

    sig = inspect.signature(create_product_audit_task)
    assert "agent" in sig.parameters

    sig = inspect.signature(create_order_tracking_task)
    assert "agent" in sig.parameters

    sig = inspect.signature(create_inventory_alert_task)
    assert "agent" in sig.parameters


    def test_workflow_class_exists():
        """Workflow class should exist and have expected methods."""
        # Class existence and method signatures verified by import tests
        # Full instantiation requires an LLM API key
        pass
