"""Shopify Assistant Agent.

Manages products, tracks orders, monitors inventory,
handles customer service for Shopify stores.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.integrations import get_products, get_orders, get_inventory


def create_shopify_assistant() -> Agent:
    return Agent(
        role="Shopify Store Manager",
        goal="Optimize and manage all aspects of a Shopify store for maximum sales and efficiency",
        backstory=(
            "You are an experienced e-commerce manager who specializes in "
            "Shopify. You handle product management, order fulfillment tracking, "
            "inventory monitoring, customer service, and store optimization. "
            "You know Shopify inside out and can manage the store via API."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_product_audit_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Audit the Shopify store's product catalog. Fetch all products and analyze: "
            "products missing images, products with empty descriptions, "
            "products with zero inventory, duplicate product names, "
            "pricing inconsistencies, and missing SEO titles/descriptions. "
            "Return prioritized fix list."
        ),
        expected_output="Product audit report with issues ranked by revenue impact and fix priority",
        agent=agent,
    )


def create_order_tracking_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Check and report on recent Shopify orders. Fetch all recent orders "
            "and identify: unfulfilled orders older than 3 days, "
            "orders with potential fraud signals, high-value orders requiring "
            "manual review, and orders from repeat customers. "
            "Also flag any refund patterns."
        ),
        expected_output="Order tracking report with attention items, fulfillment status, and fraud flags",
        agent=agent,
    )


def create_inventory_alert_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Monitor Shopify inventory levels. Fetch all products and identify: "
            "out-of-stock items, low-stock items (less than 10 units), "
            "best-sellers running low, seasonal preparation recommendations, "
            "and dead stock (no sales in 90+ days). "
            "Generate a reorder priority list."
        ),
        expected_output="Inventory status report with low-stock alerts, dead stock, and reorder recommendations",
        agent=agent,
    )


class ShopifyAssistantWorkflow:
    """Shopify store management workflow."""

    def __init__(self):
        self.agent = create_shopify_assistant()

    def audit_products(self) -> str:
        task = create_product_audit_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def track_orders(self) -> str:
        task = create_order_tracking_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def check_inventory(self) -> str:
        task = create_inventory_alert_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
