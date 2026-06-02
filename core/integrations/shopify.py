"""Shopify integration."""

from __future__ import annotations

import httpx
from core.config import settings


def _headers() -> dict:
    return {
        "X-Shopify-Access-Token": settings.shopify_api_key,
        "Content-Type": "application/json",
    }


def _store_url() -> str:
    base = settings.shopify_store_url.rstrip("/")
    if not base.startswith("https://"):
        base = f"https://{base}"
    return f"{base}/admin/api/2024-01"


async def get_products(limit: int = 50) -> list[dict]:
    """Get Shopify products."""
    if not settings.shopify_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_store_url()}/products.json",
            headers=_headers(),
            params={"limit": limit},
            timeout=15,
        )
        return resp.json().get("products", [])


async def get_orders(limit: int = 50) -> list[dict]:
    """Get Shopify orders."""
    if not settings.shopify_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_store_url()}/orders.json",
            headers=_headers(),
            params={"limit": limit, "status": "any"},
            timeout=15,
        )
        return resp.json().get("orders", [])


async def get_inventory(product_id: int) -> list[dict]:
    """Get inventory levels for a product."""
    if not settings.shopify_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_store_url()}/inventory_levels.json",
            headers=_headers(),
            params={"inventory_item_ids": product_id},
            timeout=15,
        )
        return resp.json().get("inventory_levels", [])
