"""Core integrations package."""

from core.integrations.hubspot import search_contacts, create_contact
from core.integrations.shopify import (
    count_orders,
    count_products,
    create_product,
    delete_product,
    fulfill_order,
    get_customers,
    get_inventory,
    get_inventory_item,
    get_order,
    get_orders,
    get_product,
    get_products,
    search_customers,
    set_inventory_level,
    update_product,
    get_customer,
)

__all__ = [
    "search_contacts",
    "create_contact",
    "get_products",
    "get_product",
    "create_product",
    "update_product",
    "delete_product",
    "count_products",
    "get_orders",
    "get_order",
    "fulfill_order",
    "count_orders",
    "get_inventory",
    "get_inventory_item",
    "set_inventory_level",
    "get_customers",
    "get_customer",
    "search_customers",
]
