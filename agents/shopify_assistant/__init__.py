"""Shopify Assistant Agent.

Manages products, tracks orders, monitors inventory,
handles customer service for Shopify stores.

Provides product management (create, update, audit, pricing analysis),
order tracking (fulfillment, fraud detection, status monitoring),
and inventory monitoring (low-stock alerts, reorder recommendations,
dead stock identification) via the Shopify Admin REST API.
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm
from core.integrations import (
    get_products,
    get_orders,
    get_inventory,
    get_customers,
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_shopify_assistant() -> Agent:
    """Create the Shopify Store Manager agent."""
    return Agent(
        role="Shopify Store Manager",
        goal=(
            "Optimize and manage all aspects of a Shopify store for maximum "
            "sales, efficiency, and customer satisfaction. Handle product "
            "management, order fulfillment tracking, inventory monitoring, "
            "and customer service operations via the Shopify API."
        ),
        backstory=(
            "You are an experienced e-commerce manager who specializes in "
            "Shopify. You handle product management, order fulfillment tracking, "
            "inventory monitoring, customer service, and store optimization. "
            "You know Shopify inside out and can manage the store via API. "
            "You stay on top of inventory levels, catch fulfillment delays "
            "before customers complain, and always look for opportunities "
            "to increase conversion rates through better product presentation."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Task factories — Product Management
# ---------------------------------------------------------------------------

def create_product_audit_task(agent: Agent) -> Task:
    """Audit the product catalog for quality issues."""
    return Task(
        description=(
            "Audit the Shopify store's product catalog. Fetch all products "
            "and analyze: products missing images, products with empty "
            "descriptions, products with zero inventory, duplicate product "
            "names, pricing inconsistencies, and missing SEO titles/"
            "descriptions. Return prioritized fix list."
        ),
        expected_output=(
            "Product audit report with issues ranked by revenue impact "
            "and fix priority"
        ),
        agent=agent,
    )


def create_product_pricing_analysis_task(agent: Agent) -> Task:
    """Analyze product pricing for optimization opportunities."""
    return Task(
        description=(
            "Analyze the Shopify store's product pricing strategy. "
            "Fetch all products and: identify products priced significantly "
            "below or above market norms, flag products with zero or "
            "negative margins (if cost data available), detect pricing "
            "inconsistencies across variants, suggest bundle pricing "
            "opportunities for complementary products, and recommend "
            "dynamic pricing adjustments based on inventory levels."
        ),
        expected_output=(
            "Pricing analysis report with under/overpriced products, "
            "margin flags, variant inconsistencies, and pricing "
            "recommendations"
        ),
        agent=agent,
    )


def create_product_seo_optimization_task(agent: Agent, focus: str = "") -> Task:
    """Optimize product listings for search engines."""
    context = f" Focus on: {focus}." if focus else ""
    return Task(
        description=(
            f"Review and improve product SEO across the store.{context} "
            "For each product, evaluate: title keyword optimization, "
            "meta description quality, alt text on images, URL handle "
            "structure, and collection organization. Create an action "
            "plan listing each product with its current SEO score and "
            "specific improvements needed."
        ),
        expected_output=(
            "SEO optimization plan with per-product scores, current "
            "issues, and specific improvement recommendations"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Task factories — Order Tracking
# ---------------------------------------------------------------------------

def create_order_tracking_task(agent: Agent) -> Task:
    """Check and report on recent orders."""
    return Task(
        description=(
            "Check and report on recent Shopify orders. Fetch all recent "
            "orders and identify: unfulfilled orders older than 3 days, "
            "orders with potential fraud signals (unusual billing/"
            "shipping mismatches, unusually large orders from new "
            "customers), high-value orders requiring manual review, "
            "orders from repeat customers, and any refund patterns."
        ),
        expected_output=(
            "Order tracking report with attention items, fulfillment "
            "status, and fraud flags"
        ),
        agent=agent,
    )


def create_order_fulfillment_task(agent: Agent, order_id: int = 0) -> Task:
    """Review fulfillment status and identify delays."""
    order_context = f" Pay special attention to order #{order_id}." if order_id else ""
    return Task(
        description=(
            f"Review the fulfillment pipeline for the Shopify store.{order_context} "
            "Fetch all open orders and: identify orders awaiting fulfillment "
            "by age (1 day, 3 days, 7+ days), flag orders with inventory "
            "conflicts (items out of stock), calculate average fulfillment "
            "time, and create an action plan to clear any backlog."
        ),
        expected_output=(
            "Fulfillment status report with aging analysis, inventory "
            "conflicts, and backlog action plan"
        ),
        agent=agent,
    )


def create_refund_analysis_task(agent: Agent) -> Task:
    """Analyze refund patterns to identify systemic issues."""
    return Task(
        description=(
            "Analyze refund and return patterns in the Shopify store. "
            "Fetch recent orders with refund data and: identify products "
            "with high refund rates, common refund reasons, customers "
            "with multiple refunds, time-between-purchase-and-refund "
            "patterns, and whether refunds correlate with specific "
            "product categories or order values. Flag systemic issues "
            "that should be addressed."
        ),
        expected_output=(
            "Refund analysis report with high-risk products, common "
            "causes, customer patterns, and systemic issue flags"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Task factories — Inventory Monitoring
# ---------------------------------------------------------------------------

def create_inventory_alert_task(agent: Agent) -> Task:
    """Monitor inventory levels and generate alerts."""
    return Task(
        description=(
            "Monitor Shopify inventory levels. Fetch all products and "
            "identify: out-of-stock items, low-stock items (less than "
            "10 units), best-sellers running low, seasonal preparation "
            "recommendations, and dead stock (no sales in 90+ days). "
            "Generate a reorder priority list."
        ),
        expected_output=(
            "Inventory status report with low-stock alerts, dead "
            "stock, and reorder recommendations"
        ),
        agent=agent,
    )


def create_inventory_reorder_task(agent: Agent) -> Task:
    """Generate a data-driven reorder plan."""
    return Task(
        description=(
            "Create a data-driven reorder plan for the Shopify store. "
            "Analyze current inventory levels, recent sales velocity, "
            "and seasonality to: calculate economic order quantities, "
            "identify products that need immediate reordering, estimate "
            "reorder costs, suggest optimal reorder timing, and flag "
            "supply chain risks (single-source dependencies, long "
            "lead-time items)."
        ),
        expected_output=(
            "Reorder plan with prioritized product list, estimated "
            "costs, timing recommendations, and supply chain risk flags"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Task factories — Customer Insights
# ---------------------------------------------------------------------------

def create_customer_insights_task(agent: Agent) -> Task:
    """Analyze customer data for retention and growth."""
    return Task(
        description=(
            "Analyze Shopify customer data to drive retention and "
            "growth. Fetch all customers and: segment by purchase "
            "frequency (one-time, repeat, loyal), identify high-value "
            "customers for VIP treatment, flag at-risk customers "
            "(lapsed >6 months), analyze geographic distribution, "
            "and suggest targeted marketing segments."
        ),
        expected_output=(
            "Customer insights report with segmentation, VIP list, "
            "at-risk customers, and marketing segment suggestions"
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Analysis helpers (pure functions for use in tests / workflows)
# ---------------------------------------------------------------------------

def analyze_product_quality(products: list[dict]) -> dict:
    """Analyze a list of product dicts for quality issues.

    Returns a dict with keys:
        missing_images, empty_descriptions, zero_inventory,
        duplicate_names, missing_seo, total_products
    """
    missing_images = []
    empty_descriptions = []
    zero_inventory = []
    seen_names: dict[str, int] = {}
    duplicates = []
    missing_seo = []

    for p in products:
        pid = p.get("id", "?")
        title = p.get("title", "")

        # Check images
        if not p.get("images"):
            missing_images.append(pid)

        # Check description
        body = (p.get("body_html") or "").strip()
        if not body:
            empty_descriptions.append(pid)

        # Check inventory
        total_inventory = sum(
            v.get("inventory_quantity", 0) or 0
            for v in p.get("variants", [])
        )
        if total_inventory == 0:
            zero_inventory.append(pid)

        # Check duplicates
        lower_title = title.lower().strip()
        if lower_title:
            seen_names[lower_title] = seen_names.get(lower_title, 0) + 1

        # Check SEO
        if not p.get("seo") and not (title and body):
            missing_seo.append(pid)

    duplicates = [
        name for name, count in seen_names.items() if count > 1
    ]

    return {
        "missing_images": missing_images,
        "empty_descriptions": empty_descriptions,
        "zero_inventory": zero_inventory,
        "duplicate_names": duplicates,
        "missing_seo": missing_seo,
        "total_products": len(products),
    }


def analyze_orders(orders: list[dict]) -> dict:
    """Analyze a list of order dicts for tracking issues.

    Returns a dict with keys:
        unfulfilled_old, high_value, fraud_flags,
        repeat_customers, refund_patterns, total_orders
    """
    unfulfilled_old = []
    high_value = []
    fraud_flags = []
    customers: dict[str, int] = {}
    refunds = []

    for o in orders:
        oid = o.get("id", "?")
        total = float(o.get("total_price", "0") or "0")
        fulfillment = o.get("fulfillment_status")
        created = o.get("created_at", "")
        email = (o.get("email") or "").lower()
        financial = o.get("financial_status", "")

        # Unfulfilled older than 3 days (simplified: just check unfulfilled)
        if fulfillment != "fulfilled":
            unfulfilled_old.append(oid)

        # High value orders (> $500)
        if total > 500:
            high_value.append({"id": oid, "total": total})

        # Fraud signals: billing/shipping mismatch
        billing = o.get("billing_address", {}) or {}
        shipping = o.get("shipping_address", {}) or {}
        if billing and shipping:
            if (
                (billing.get("city") or "").lower()
                != (shipping.get("city") or "").lower()
                or (billing.get("country") or "").lower()
                != (shipping.get("country") or "").lower()
            ):
                fraud_flags.append({
                    "id": oid,
                    "reason": "billing/shipping address mismatch",
                    "billing_city": billing.get("city"),
                    "shipping_city": shipping.get("city"),
                })

        # Customer tracking
        if email:
            customers[email] = customers.get(email, 0) + 1

        # Refunds
        if financial in ("refunded", "partially_refunded"):
            refunds.append({
                "id": oid,
                "status": financial,
                "total": total,
                "created_at": created,
            })

    repeat_customers = {
        email: count for email, count in customers.items() if count > 1
    }

    return {
        "unfulfilled_old": unfulfilled_old,
        "high_value": high_value,
        "fraud_flags": fraud_flags,
        "repeat_customers": repeat_customers,
        "refund_patterns": refunds,
        "total_orders": len(orders),
    }


def analyze_inventory(products: list[dict]) -> dict:
    """Analyze product inventory levels.

    Returns a dict with keys:
        out_of_stock, low_stock, healthy_stock,
        total_tracked_inventory
    """
    out_of_stock = []
    low_stock = []
    healthy_stock = []
    total_inventory = 0

    for p in products:
        pid = p.get("id", "?")
        title = p.get("title", "")
        variants = p.get("variants", [])
        product_total = sum(
            v.get("inventory_quantity", 0) or 0 for v in variants
        )
        total_inventory += product_total

        if product_total == 0:
            out_of_stock.append({"id": pid, "title": title})
        elif product_total < 10:
            low_stock.append({
                "id": pid,
                "title": title,
                "quantity": product_total,
            })
        else:
            healthy_stock.append({
                "id": pid,
                "title": title,
                "quantity": product_total,
            })

    return {
        "out_of_stock": out_of_stock,
        "low_stock": low_stock,
        "healthy_stock": healthy_stock,
        "total_tracked_inventory": total_inventory,
    }


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class ShopifyAssistantWorkflow:
    """Shopify store management workflow with full product, order,
    inventory, and customer capabilities."""

    def __init__(self):
        self.agent = create_shopify_assistant()

    def audit_products(self) -> str:
        task = create_product_audit_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_pricing(self) -> str:
        task = create_product_pricing_analysis_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def optimize_seo(self, focus: str = "") -> str:
        task = create_product_seo_optimization_task(self.agent, focus=focus)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def track_orders(self) -> str:
        task = create_order_tracking_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def check_fulfillment(self, order_id: int = 0) -> str:
        task = create_order_fulfillment_task(self.agent, order_id=order_id)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_refunds(self) -> str:
        task = create_refund_analysis_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def check_inventory(self) -> str:
        task = create_inventory_alert_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def plan_reorder(self) -> str:
        task = create_inventory_reorder_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def customer_insights(self) -> str:
        task = create_customer_insights_task(self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
