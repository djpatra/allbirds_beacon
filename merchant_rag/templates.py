"""Predefined merchant query templates (N=5)."""

from __future__ import annotations

TOP_N = 5

# Each template: id, display question, keyword triggers, handler name, copy.
TEMPLATES = [
    {
        "id": "restock",
        "question": "Which products should I restock first?",
        "keywords": [
            "restock", "replenish", "out of stock", "oos", "unavailable",
            "back in stock", "notify", "bis", "availability risk",
        ],
        "handler": "restock_first",
        "headline": "Expedite replenishment for {n} styles — ranked by missed-opportunity loss (urgency).",
        "why": (
            "Missed-opportunity loss (lost units × price over the lookback) is the urgency signal — "
            "styles that left demand on the table while unavailable or size-gapped rise first."
        ),
        "action": "Investigate replenishment and expected availability; trigger BIS notifications.",
        "columns": [
            "product_name", "subcategory", "lost_units", "lost_revenue",
            "lost_units_14d", "lost_revenue_14d", "urgency",
            "u_stockout", "catalog_size_gap", "interest_trend", "u_bi", "r_pv", "A_score",
        ],
    },
    {
        "id": "findability",
        "question": "Which products are well stocked, but hard to find / under-ranked?",
        "keywords": [
            "findability", "well stocked", "can't find", "cannot find",
            "search relevance", "ranking", "filters", "buried", "discovery",
        ],
        "handler": "findability_gap",
        "headline": "In-stock styles look under-discovered vs peers — ranking/filters/attributes likely.",
        "why": (
            "Demand shows up as category traffic and attribute gaps, but these in-stock "
            "products get weak on-product actions (proxy for poor findability)."
        ),
        "action": "Investigate search relevance, filters, and catalog attributes used for boosting.",
        "columns": [
            "product_name", "subcategory", "lost_units", "lost_revenue", "urgency",
            "r_pv", "r_actions", "action_per_view", "catalog_size_gap", "r_stockout", "B_score",
        ],
    },
    {
        "id": "price",
        "question": "Which products are people not buying because they are expensive?",
        "keywords": [
            "expensive", "price", "pricing", "price drop", "pricedrop",
            "too costly", "resistance", "markdown",
        ],
        "handler": "price_resistance",
        "headline": "High interest with slow conversion — possible price resistance vs peers.",
        "why": (
            "Shoppers wishlist/cart and subscribe to price-drop alerts, but sales stay "
            "weak while stock is healthy."
        ),
        "action": "Compare pricing to similar product types; test promo or reposition.",
        "columns": [
            "product_name", "subcategory", "price", "lost_units", "lost_revenue", "urgency",
            "r_pdalert", "r_wl", "r_ca", "r_sales", "peer_sales_rate", "price_score",
        ],
    },
    {
        "id": "pdp_weak",
        "question": "Which products have high traffic, but no sales?",
        "keywords": [
            "high traffic", "no sales", "no actions", "weak interest",
            "pdp", "product page", "visits but", "traffic waste",
        ],
        "handler": "pdp_weak",
        "headline": "High visits with almost no cart/wishlist/alerts — PDP or relevance issue.",
        "why": (
            "Traffic arrives, inventory is fine, but shoppers take almost no commercial "
            "actions (distinct from findability: they *did* land here)."
        ),
        "action": "Investigate broken product page, media, sizing info, and product relevance.",
        "columns": [
            "product_name", "subcategory", "lost_units", "lost_revenue", "urgency",
            "r_pv", "r_actions", "action_per_view", "r_sales", "r_stockout", "C_score",
        ],
    },
    {
        "id": "dead_stock",
        "question": "Which products are dead stock?",
        "keywords": [
            "dead stock", "deadstock", "tied up", "capital", "not moving",
            "stale", "no engagement",
        ],
        "handler": "dead_stock",
        "headline": "Capital tied in styles shoppers are not engaging.",
        "why": "Fat inventory with flat/falling traffic and little intent or sales.",
        "action": "Markdown, bundle, or delist after a content check.",
        "columns": [
            "product_name", "subcategory", "mean_stock", "lost_units", "lost_revenue",
            "pv_trend", "r_pv", "r_actions", "r_sales", "dead_score",
        ],
    },
    {
        "id": "rising_star",
        "question": "Which rising products should I protect before they sell out?",
        "keywords": [
            "rising", "rising star", "protect", "trending", "guard inventory",
            "coming hot", "breakout",
        ],
        "handler": "rising_star",
        "headline": "Intent rising while still available — protect inventory before they become restock emergencies.",
        "why": "Steep interest trend, healthy actions, and stock still present.",
        "action": "Raise reorder points / fund ads on winners before stockout.",
        "columns": [
            "product_name", "subcategory", "interest_trend", "lost_units", "lost_revenue",
            "urgency", "r_actions", "r_stockout", "r_sales", "r_pv", "star_score",
        ],
    },
]


def list_questions() -> list[str]:
    return [t["question"] for t in TEMPLATES]
