"""Core integrations package."""

from core.integrations.hubspot import (
    search_contacts,
    create_contact,
    update_contact,
    get_contact,
    create_deal,
    update_deal_stage,
    list_deals,
    get_deal,
    associate_deal_contact,
    associate_deal_company,
    list_pipelines,
)
from core.integrations.shopify import get_products, get_orders, get_inventory

__all__ = [
    # HubSpot
    "search_contacts",
    "create_contact",
    "update_contact",
    "get_contact",
    "create_deal",
    "update_deal_stage",
    "list_deals",
    "get_deal",
    "associate_deal_contact",
    "associate_deal_company",
    "list_pipelines",
    # Shopify
    "get_products",
    "get_orders",
    "get_inventory",
]
