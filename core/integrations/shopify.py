"""Shopify integration client.

Provides async wrapper functions for the Shopify Admin REST API (2024-01),
covering products, orders, inventory, and customers. Each function gracefully
returns an empty list/dict when SHOPIFY_API_KEY is unset so that the
suite can be imported and tested without real credentials.
"""
from __future__ import annotations

import httpx

from core.config import settings

BASE_URL_TEMPLATE = "https://{store}.myshopify.com/admin/api/2024-01"
API_VERSION = "2024-01"


def _base_url() -> str:
    """Build the base admin API URL from settings."""
    raw = settings.shopify_store_url.rstrip("/")
    # Accept either "store.myshopify.com" or "https://store.myshopify.com"
    if raw.startswith("https://") or raw.startswith("http://"):
        raw = raw.split("://", 1)[1]
    # Strip trailing /admin/... if the full admin URL was supplied
    if "/admin/api/" in raw:
        raw = raw.split("/admin/api/")[0]
    return f"https://{raw}/admin/api/{API_VERSION}"


def _headers():
    return {
        "X-Shopify-Access-Token": settings.shopify_api_key,
        "Content-Type": "application/json",
    }


def _has_credentials():
    return bool(settings.shopify_api_key and settings.shopify_store_url)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

async def get_products(limit=50, collection_id=None):
    """Fetch products from the store.

    Args:
        limit: Maximum number of products to return (max 250).
        collection_id: Optional collection filter.

    Returns:
        List of product dicts, or empty list when credentials are missing.
    """
    if not _has_credentials():
        return []
    params = {"limit": min(limit, 250)}
    if collection_id is not None:
        params["collection_id"] = collection_id
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/products.json",
            headers=_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("products", [])


async def get_product(product_id):
    """Fetch a single product by ID."""
    if not _has_credentials():
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/products/{product_id}.json",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("product", {})


async def create_product(title, **kwargs):
    """Create a new product.

    Args:
        title: Product title.
        **kwargs: Additional product fields (body_html, vendor,
            product_type, variants, images, etc.).

    Returns:
        Created product dict, or empty dict when credentials are missing.
    """
    if not _has_credentials():
        return {}
    payload = {"product": {"title": title, **kwargs}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_base_url()}/products.json",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("product", {})


async def update_product(product_id, **kwargs):
    """Update an existing product.

    Args:
        product_id: Shopify product ID.
        **kwargs: Fields to update on the product.

    Returns:
        Updated product dict, or empty dict when credentials are missing.
    """
    if not _has_credentials():
        return {}
    payload = {"product": kwargs}
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{_base_url()}/products/{product_id}.json",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("product", {})


async def delete_product(product_id):
    """Delete a product. Returns True on success."""
    if not _has_credentials():
        return False
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{_base_url()}/products/{product_id}.json",
            headers=_headers(),
            timeout=15,
        )
        return resp.status_code == 200


async def count_products():
    """Return total product count."""
    if not _has_credentials():
        return 0
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/products/count.json",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("count", 0)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

async def get_orders(
    limit=50,
    status="any",
    financial_status=None,
    fulfillment_status=None,
):
    """Fetch orders with optional filters.

    Args:
        limit: Maximum orders to return.
        status: 'open', 'closed', 'cancelled', or 'any'.
        financial_status: Filter by 'paid', 'pending', 'refunded', etc.
        fulfillment_status: Filter by 'fulfilled', 'partial', 'unfulfilled'.

    Returns:
        List of order dicts, or empty list when credentials are missing.
    """
    if not _has_credentials():
        return []
    params = {"limit": min(limit, 250), "status": status}
    if financial_status:
        params["financial_status"] = financial_status
    if fulfillment_status:
        params["fulfillment_status"] = fulfillment_status
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/orders.json",
            headers=_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("orders", [])


async def get_order(order_id):
    """Fetch a single order by ID."""
    if not _has_credentials():
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/orders/{order_id}.json",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("order", {})


async def fulfill_order(order_id, location_id, tracking_number=""):
    """Create a fulfillment for an order.

    Args:
        order_id: Shopify order ID.
        location_id: Fulfillment location ID.
        tracking_number: Optional tracking number.

    Returns:
        Fulfillment dict, or empty dict when credentials are missing.
    """
    if not _has_credentials():
        return {}
    payload = {
        "fulfillment": {
            "location_id": location_id,
        }
    }
    if tracking_number:
        payload["fulfillment"]["tracking_number"] = tracking_number
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_base_url()}/orders/{order_id}/fulfillments.json",
            headers=_headers(),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("fulfillment", {})


async def count_orders(status="any"):
    """Return total order count, optionally filtered by status."""
    if not _has_credentials():
        return 0
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/orders/count.json",
            headers=_headers(),
            params={"status": status},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("count", 0)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

async def get_inventory(inventory_item_ids=None):
    """Fetch inventory levels.

    Args:
        inventory_item_ids: Optional list of inventory item IDs to filter.

    Returns:
        List of inventory level dicts, or empty list when
        credentials are missing.
    """
    if not _has_credentials():
        return []
    params = {}
    if inventory_item_ids:
        params["inventory_item_ids"] = ",".join(str(i) for i in inventory_item_ids)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/inventory_levels.json",
            headers=_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("inventory_levels", [])


async def get_inventory_item(inventory_item_id):
    """Fetch a single inventory item by ID."""
    if not _has_credentials():
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/inventory_items/{inventory_item_id}.json",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("inventory_item", {})


async def set_inventory_level(inventory_item_id, location_id, available):
    """Set (adjust) inventory level at a specific location.

    Args:
        inventory_item_id: The inventory item ID.
        location_id: The location ID.
        available: The available quantity to set.

    Returns:
        Updated inventory level dict, or empty dict
        when credentials are missing.
    """
    if not _has_credentials():
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_base_url()}/inventory_levels/set.json",
            headers=_headers(),
            json={
                "location_id": location_id,
                "inventory_item_id": inventory_item_id,
                "available": available,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("inventory_level", {})


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

async def get_customers(limit=50):
    """Fetch customers from the store."""
    if not _has_credentials():
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/customers.json",
            headers=_headers(),
            params={"limit": min(limit, 250)},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("customers", [])


async def get_customer(customer_id):
    """Fetch a single customer by ID."""
    if not _has_credentials():
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/customers/{customer_id}.json",
            headers=_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("customer", {})


async def search_customers(query, limit=50):
    """Search customers by name, email, or phone."""
    if not _has_credentials():
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/customers/search.json",
            headers=_headers(),
            params={"query": query, "limit": min(limit, 250)},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("customers", [])
