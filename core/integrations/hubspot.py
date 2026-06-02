"""HubSpot CRM integration."""

from __future__ import annotations

import httpx
from core.config import settings

BASE_URL = "https://api.hubapi.com"


async def search_contacts(query: str, limit: int = 10) -> list[dict]:
    """Search HubSpot contacts."""
    if not settings.hubspot_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE_URL}/crm/v3/objects/contacts/search",
            headers={"Authorization": f"Bearer {settings.hubspot_api_key}"},
            json={
                "filterGroups": [{"filters": [{"propertyName": "email", "operator": "CONTAINS_TOKEN", "value": query}]}],
                "limit": limit,
                "properties": ["email", "firstname", "lastname", "company", "phone"],
            },
            timeout=15,
        )
        data = resp.json()
        return [r["properties"] for r in data.get("results", [])]


async def create_contact(email: str, firstname: str = "", lastname: str = "", company: str = "", phone: str = "") -> dict:
    """Create a HubSpot contact."""
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/crm/v3/objects/contacts",
            headers={"Authorization": f"Bearer {settings.hubspot_api_key}"},
            json={"properties": {"email": email, "firstname": firstname, "lastname": lastname, "company": company, "phone": phone}},
            timeout=15,
        )
        return resp.json()
