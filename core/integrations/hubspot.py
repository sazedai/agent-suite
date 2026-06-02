"""HubSpot CRM integration.

Provides contact management, deal/pipeline operations, and
contact enrichment via the HubSpot CRM API v3.
"""

from __future__ import annotations

import httpx
from core.config import settings

BASE_URL = "https://api.hubapi.com"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.hubspot_api_key}"}


async def search_contacts(query: str, limit: int = 10) -> list[dict]:
    """Search HubSpot contacts by email or name token.

    Args:
        query: Search term matched against email / firstname / lastname.
        limit: Maximum results to return.

    Returns:
        List of contact property dicts. Empty list if API key is unset.
    """
    if not settings.hubspot_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/crm/v3/objects/contacts/search",
            headers=_headers(),
            json={
                "filterGroups": [{
                    "filters": [
                        {"propertyName": "email", "operator": "CONTAINS_TOKEN", "value": query},
                    ]
                }],
                "limit": limit,
                "properties": ["email", "firstname", "lastname", "company", "phone", "lifecyclestage"],
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [r["properties"] for r in data.get("results", [])]


async def create_contact(
    email: str,
    firstname: str = "",
    lastname: str = "",
    company: str = "",
    phone: str = "",
    lifecyclestage: str = "",
) -> dict:
    """Create a HubSpot contact.

    Args:
        email: Contact email (required).
        firstname: First name.
        lastname: Last name.
        company: Company name.
        phone: Phone number.
        lifecyclestage: Pipeline stage — e.g. ``lead``, ``qualified``,
            ``opportunity``, ``customer``, ``churned``.

    Returns:
        HubSpot response JSON, or empty dict if API key is unset.
    """
    if not settings.hubspot_api_key:
        return {}
    properties: dict[str, str] = {"email": email}
    if firstname:
        properties["firstname"] = firstname
    if lastname:
        properties["lastname"] = lastname
    if company:
        properties["company"] = company
    if phone:
        properties["phone"] = phone
    if lifecyclestage:
        properties["lifecyclestage"] = lifecyclestage

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/crm/v3/objects/contacts",
            headers=_headers(),
            json={"properties": properties},
            timeout=15,
        )
        return resp.json()


async def update_contact(contact_id: str, properties: dict[str, str]) -> dict:
    """Update an existing HubSpot contact.

    Args:
        contact_id: HubSpot contact ID.
        properties: Key-value pairs to update.

    Returns:
        HubSpot response JSON, or empty dict if API key is unset.
    """
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{BASE_URL}/crm/v3/objects/contacts/{contact_id}",
            headers=_headers(),
            json={"properties": properties},
            timeout=15,
        )
        return resp.json()


async def get_contact(contact_id: str) -> dict:
    """Retrieve a single HubSpot contact by ID.

    Args:
        contact_id: HubSpot contact ID.

    Returns:
        Contact properties dict, or empty dict if not found / no API key.
    """
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE_URL}/crm/v3/objects/contacts/{contact_id}",
            headers=_headers(),
            params={
                "properties": "email,firstname,lastname,company,phone,lifecyclestage,notes_last_updated"
            },
            timeout=15,
        )
        if resp.status_code == 404:
            return {}
        return resp.json().get("properties", {})


# ── Deal / Pipeline operations ──────────────────────────────────────────────

async def create_deal(
    dealname: str,
    pipeline: str = "default",
    dealstage: str = "appointmentscheduled",
    amount: str = "",
    contact_id: str = "",
    company_id: str = "",
) -> dict:
    """Create a HubSpot deal.

    Args:
        dealname: Human-readable deal name.
        pipeline: Pipeline ID (``default`` is the built-in pipeline).
        dealstage: Stage ID within the pipeline.
        amount: Deal value as string.
        contact_id: HubSpot contact ID to associate.
        company_id: HubSpot company ID to associate.

    Returns:
        HubSpot response JSON, or empty dict if API key is unset.
    """
    if not settings.hubspot_api_key:
        return {}
    props: dict[str, str] = {
        "dealname": dealname,
        "pipeline": pipeline,
        "dealstage": dealstage,
    }
    if amount:
        props["amount"] = amount

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/crm/v3/objects/deals",
            headers=_headers(),
            json={"properties": props},
            timeout=15,
        )
        data = resp.json()

    # Associate contact if provided
    if contact_id and data.get("id"):
        await associate_deal_contact(data["id"], contact_id)

    # Associate company if provided
    if company_id and data.get("id"):
        await associate_deal_company(data["id"], company_id)

    return data


async def update_deal_stage(deal_id: str, dealstage: str) -> dict:
    """Move a deal to a different pipeline stage.

    Args:
        deal_id: HubSpot deal ID.
        dealstage: Target stage ID.

    Returns:
        HubSpot response JSON.
    """
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{BASE_URL}/crm/v3/objects/deals/{deal_id}",
            headers=_headers(),
            json={"properties": {"dealstage": dealstage}},
            timeout=15,
        )
        return resp.json()


async def list_deals(limit: int = 20, pipeline: str = "") -> list[dict]:
    """List deals, optionally filtered by pipeline.

    Args:
        limit: Maximum number of deals to return.
        pipeline: Pipeline ID to filter by (empty = all).

    Returns:
        List of deal property dicts.
    """
    if not settings.hubspot_api_key:
        return []
    body: dict = {
        "limit": limit,
        "properties": ["dealname", "pipeline", "dealstage", "amount", "closedate", "createdate"],
    }
    if pipeline:
        body["filterGroups"] = [{
            "filters": [{"propertyName": "pipeline", "operator": "EQ", "value": pipeline}]
        }]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/crm/v3/objects/deals/search",
            headers=_headers(),
            json=body,
            timeout=15,
        )
        data = resp.json()
        return [r["properties"] for r in data.get("results", [])]


async def get_deal(deal_id: str) -> dict:
    """Retrieve a single deal by ID.

    Returns:
        Deal properties dict, or empty dict if not found / no API key.
    """
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE_URL}/crm/v3/objects/deals/{deal_id}",
            headers=_headers(),
            params={
                "properties": "dealname,pipeline,dealstage,amount,closedate,createdate"
            },
            timeout=15,
        )
        if resp.status_code == 404:
            return {}
        return resp.json().get("properties", {})


async def associate_deal_contact(deal_id: str, contact_id: str) -> dict:
    """Associate a contact with a deal."""
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{BASE_URL}/crm/v3/objects/deals/{deal_id}/associations/contacts/{contact_id}/3",
            headers=_headers(),
            timeout=15,
        )
        return resp.json()


async def associate_deal_company(deal_id: str, company_id: str) -> dict:
    """Associate a company with a deal."""
    if not settings.hubspot_api_key:
        return {}
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{BASE_URL}/crm/v3/objects/deals/{deal_id}/associations/companies/{company_id}/5",
            headers=_headers(),
            timeout=15,
        )
        return resp.json()


# ── Pipeline management ─────────────────────────────────────────────────────

async def list_pipelines(object_type: str = "deals") -> list[dict]:
    """List all pipelines for a given object type (default: deals).

    Returns:
        List of pipeline dicts with ``id``, ``label``, ``stages`` keys.
    """
    if not settings.hubspot_api_key:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE_URL}/crm/v3/pipelines/{object_type}",
            headers=_headers(),
            timeout=15,
        )
        data = resp.json()
        return [
            {
                "id": p["id"],
                "label": p["label"],
                "stages": [
                    {"id": s["id"], "label": s["label"]}
                    for s in p.get("stages", [])
                ],
            }
            for p in data.get("results", [])
        ]
