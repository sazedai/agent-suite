"""Core integrations package."""

from core.integrations.hubspot import search_contacts, create_contact
from core.integrations.shopify import get_products, get_orders, get_inventory

__all__ = ["search_contacts", "create_contact", "get_products", "get_orders", "get_inventory"]
