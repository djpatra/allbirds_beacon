#!/usr/bin/env python3
"""
shopper_journey_generator.py
============================

Shopper-centric synthetic journey dataset (contrast with allbirds_pv_generator.py).

Old generator (demand-centric)
------------------------------
  Per SKU-day: draw latent Poisson-ish demand from Zipf base * seasonality * trend
  * noise, censor by stock, then fabricate page_views / cart / wishlist / alerts
  as noisy functions of that demand. Shoppers are not first-class entities.

This generator (shopper-centric)
--------------------------------
  N shoppers with opaque UUID ids and behavioral archetypes. Each shopper
  generates a variable number of sessions over the simulation window. Sessions
  emit an event chain (search → product_view → sku_view → cart / wishlist /
  BIS / price_drop → purchase or bounce). Cart / wishlist / BIS / price_drop
  can schedule a conversion attempt after a lag; at conversion time stock is
  checked and either a purchase decodes stock or a lost opportunity is recorded.

Catalog attraction weights, (s,S) restock with lead time, and sibling-size
substitution helpers are copied from allbirds_pv_generator.py so the two
datasets share the same SKU universe and inventory physics — but daily_metrics
here are ROLLED UP from journey events, and ground_truth fields are
journey-derived (not the old latent Poisson formulas).

Daily sales floor (baseline volume)
----------------------------------
  After organic journeys each calendar day, if successful purchases that day
  are below --daily-sales (default 2000), inject short filler sessions
  (product_view → sku_view → purchase) to fill the shortfall. Sampling uses
  attraction weights over in-stock SKUs so the SKU mix varies day to day.
  This is a FLOOR, not an exact quota: organic journeys may exceed the target;
  injection only fills shortfalls. If not enough in-stock units exist to hit
  the floor, sell as many as possible and log a WARN.

Usage:
  python shopper_journey_generator.py allbirds_products.json \\
      --days 90 --n-shoppers 50000 --daily-sales 2000 --seed 42 \\
      --out-dir ./demand_shopper

Outputs under --out-dir:
  shoppers.jsonl / shoppers.json
  sessions.jsonl
  events.jsonl
  catalog.json          (SKU list without hidden priors)
  daily_metrics.jsonl   (+ .parquet if pandas/pyarrow available)
  ground_truth.jsonl    (+ .parquet if available)

Ground-truth field definitions (journey-derived; NOT identical to old GT):
  purchase_attempts   — times a shopper tried to buy this SKU that day
                        (same-session purchase or delayed conversion attempt)
  sales               — successful purchases that decremented stock
  lost_units          — purchase attempts blocked by OOS (stock == 0)
  strong_oos_intents  — cart / wishlist / sku_view while focused SKU was OOS
                        (size disappointment / lost opportunity signals)
  journey_demand      — purchase_attempts + strong_oos_intents
                        (rough latent demand proxy from journeys)
  availability        — sales / max(purchase_attempts, 1) when attempts > 0,
                        else 1.0 if stock_start > 0 else 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator


# =============================================================================
# CONFIGURATION (shared physics with allbirds_pv_generator.py where noted)
# =============================================================================

TOP_SKU_DAILY_DEMAND = 20.0
ZIPF_ALPHA = 1.05
MIN_BASE_DEMAND = 0.05

WEEKLY_FACTOR = [0.88, 0.92, 1.00, 1.04, 1.10, 1.22, 1.05]  # Mon..Sun
TREND_SLOPE_STD = 0.002

SUBCAT_SEASONAL = {
    "running":  {3: 1.12, 4: 1.18, 5: 1.15, 9: 1.10, 10: 1.08},
    "sandals":  {5: 1.25, 6: 1.35, 7: 1.30, 8: 1.20},
    "slip_on":  {4: 1.10, 5: 1.12, 6: 1.10},
    "apparel":  {9: 1.15, 10: 1.20, 11: 1.25, 12: 1.15, 1: 1.08},
    "socks":    {11: 1.20, 12: 1.30, 1: 1.15, 2: 1.10},
}

SIZE_DEMAND_WEIGHT = {
    "UK 3": 0.55, "UK 3.5": 0.60, "UK 4": 0.75, "UK 4.5": 0.85,
    "UK 5": 0.95, "UK 5.5": 1.05, "UK 6": 1.15, "UK 6.5": 1.20,
    "UK 7": 1.25, "UK 7.5": 1.22, "UK 8": 1.28, "UK 8.5": 1.18,
    "UK 9": 1.15, "UK 9.5": 1.05, "UK 10": 1.00, "UK 10.5": 0.90,
    "UK 11": 0.80, "UK 11.5": 0.70, "UK 12": 0.65, "UK 13": 0.55,
    "UK 14": 0.45,
}

# Inventory physics (copied)
STOCK_MIN_INITIAL = 10
STOCK_MAX_INITIAL = 400
UNAVAILABLE_TARGET_RANGE = (40, 180)
RESTOCK_THRESHOLD_RATIO = 0.20
RESTOCK_LEAD_TIME_DAYS = 7
SUBSTITUTION_RATE = 0.45

# Archetype mix: decisive ~5%; five others at 19% each (sum = 100%)
ARCHETYPE_MIX: dict[str, float] = {
    "decisive": 0.05,
    "casual": 0.19,
    "comparer": 0.19,
    "deal_hunter": 0.19,
    "size_constrained": 0.19,
    "wishlist_filler": 0.19,
}

# Per-archetype behavioral knobs (sessions over the full window; clarity > tuning)
# sessions_mean: expected sessions across the whole simulation window
# session_events_mean: expected events per session (before early bounce)
ARCHETYPE_BEHAVIOR: dict[str, dict[str, float]] = {
    "decisive": {
        "sessions_mean": 3.0,
        "session_events_mean": 4.0,
        "p_start_search": 0.35,
        "p_multi_product": 0.12,
        "p_wishlist": 0.04,
        "p_price_drop": 0.02,
        "p_cart": 0.55,
        "p_same_session_purchase": 0.70,
        "p_bis_when_oos": 0.25,
        "delay_days_mean": 0.4,
        "size_edge_boost": 1.0,
        "return_boost": 1.0,
    },
    "casual": {
        "sessions_mean": 7.0,
        "session_events_mean": 5.0,
        "p_start_search": 0.55,
        "p_multi_product": 0.30,
        "p_wishlist": 0.12,
        "p_price_drop": 0.05,
        "p_cart": 0.22,
        "p_same_session_purchase": 0.25,
        "p_bis_when_oos": 0.30,
        "delay_days_mean": 2.0,
        "size_edge_boost": 1.0,
        "return_boost": 1.1,
    },
    "comparer": {
        "sessions_mean": 11.0,
        "session_events_mean": 10.0,
        "p_start_search": 0.70,
        "p_multi_product": 0.75,
        "p_wishlist": 0.18,
        "p_price_drop": 0.08,
        "p_cart": 0.28,
        "p_same_session_purchase": 0.18,
        "p_bis_when_oos": 0.35,
        "delay_days_mean": 3.0,
        "size_edge_boost": 1.0,
        "return_boost": 1.2,
    },
    "deal_hunter": {
        "sessions_mean": 9.0,
        "session_events_mean": 7.0,
        "p_start_search": 0.60,
        "p_multi_product": 0.40,
        "p_wishlist": 0.28,
        "p_price_drop": 0.40,
        "p_cart": 0.30,
        "p_same_session_purchase": 0.15,
        "p_bis_when_oos": 0.25,
        "delay_days_mean": 4.0,
        "size_edge_boost": 1.0,
        "return_boost": 1.3,
    },
    "size_constrained": {
        "sessions_mean": 8.0,
        "session_events_mean": 6.0,
        "p_start_search": 0.50,
        "p_multi_product": 0.35,  # try sibling sizes / colours
        "p_wishlist": 0.15,
        "p_price_drop": 0.06,
        "p_cart": 0.32,
        "p_same_session_purchase": 0.30,
        "p_bis_when_oos": 0.65,
        "delay_days_mean": 2.5,
        "size_edge_boost": 2.4,  # bias mid + edge sizes
        "return_boost": 1.15,
    },
    "wishlist_filler": {
        "sessions_mean": 14.0,
        "session_events_mean": 6.0,
        "p_start_search": 0.45,
        "p_multi_product": 0.50,
        "p_wishlist": 0.48,
        "p_price_drop": 0.12,
        "p_cart": 0.15,
        "p_same_session_purchase": 0.08,
        "p_bis_when_oos": 0.40,
        "delay_days_mean": 5.0,
        "size_edge_boost": 1.0,
        "return_boost": 1.6,
    },
}

EVENT_TYPES = [
    "search",
    "product_view",
    "sku_view",
    "add_to_cart",
    "add_to_wishlist",
    "subscribe_back_in_stock",
    "subscribe_price_drop",
    "purchase",
    "bounce",
]


# =============================================================================
# UTILITIES (copied / adapted from allbirds_pv_generator.py)
# =============================================================================

def slugify(s: str, max_len: int = 60) -> str:
    out = "".join(c if c.isalnum() else "_" for c in (s or "").lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:max_len]


def parse_price(p: Any) -> float:
    if isinstance(p, (int, float)):
        return float(p)
    s = str(p).replace("£", "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def infer_subcategory(product_name: str, product_type: str) -> str:
    name = (product_name or "").lower()
    ptype = (product_type or "").lower()
    for kws, sc in [
        (("runner", "run "), "running"),
        (("cruiser", "slip on", "slip-on"), "slip_on"),
        (("sandal", "flip flop", "flip-flop"), "sandals"),
        (("sweater", "hoodie", "tee", "shirt", "jacket", "pant", "short"), "apparel"),
        (("sock",), "socks"),
    ]:
        if any(k in name for k in kws):
            return sc
    return ptype or "other"


def sample_count(mean: float, rng: random.Random) -> int:
    """Non-negative integer count ~ Poisson (or normal approx for large mean)."""
    if mean <= 0:
        return 0
    if mean < 30:
        L = math.exp(-mean)
        k, p = 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    return max(0, int(round(rng.gauss(mean, math.sqrt(mean)))))


def size_weight(size: str) -> float:
    if size in SIZE_DEMAND_WEIGHT:
        return SIZE_DEMAND_WEIGHT[size]
    return 0.85


def is_edge_size(size: str) -> bool:
    return size in {
        "UK 3", "UK 3.5", "UK 4", "UK 4.5",
        "UK 11", "UK 11.5", "UK 12", "UK 13", "UK 14",
    }


def is_mid_size(size: str) -> bool:
    return size in {
        "UK 6", "UK 6.5", "UK 7", "UK 7.5", "UK 8", "UK 8.5", "UK 9",
    }


# =============================================================================
# CATALOG (copied from allbirds_pv_generator.py)
# =============================================================================

def load_skus(catalog_json_path: str) -> list[dict]:
    with open(catalog_json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    if not isinstance(products, list):
        raise ValueError("Catalog JSON must be a list of product records.")

    skus: list[dict] = []
    for p in products:
        if "error" in p:
            continue
        name = p.get("product_name", "").strip()
        if not name:
            continue
        pid = "P_" + slugify(name)
        subcat = infer_subcategory(name, p.get("product_type", ""))
        price = parse_price(p.get("price", "0"))

        for colour, info in (p.get("colour_variants") or {}).items():
            if not isinstance(info, dict) or "error" in info:
                continue
            vid = f"{pid}__{slugify(colour)}"
            for size in info.get("available_sizes", []):
                skus.append(_mk_sku(vid, pid, name, colour, size, subcat, price, True))
            for size in info.get("unavailable_sizes", []):
                skus.append(_mk_sku(vid, pid, name, colour, size, subcat, price, False))
    return skus


def _mk_sku(vid, pid, name, colour, size, subcat, price, available) -> dict:
    return {
        "sku_id": f"{vid}__{slugify(size)}",
        "variant_id": vid,
        "product_id": pid,
        "product_name": name,
        "colour": colour,
        "size": size,
        "subcategory": subcat,
        "price": price,
        "available_in_scrape": available,
    }


def assign_priors(skus: list[dict], rng: random.Random) -> None:
    """Assign per-SKU constant base_demand, trend_slope, and stock params."""
    n = len(skus)
    if n == 0:
        return

    ranks = list(range(1, n + 1))
    rng.shuffle(ranks)
    weights = [1.0 / (r ** ZIPF_ALPHA) for r in ranks]
    scale = TOP_SKU_DAILY_DEMAND / max(weights)

    for sku, w in zip(skus, weights):
        raw = w * scale * size_weight(sku["size"])
        sku["base_demand"] = max(MIN_BASE_DEMAND, raw)
        sku["trend_slope"] = rng.gauss(0, TREND_SLOPE_STD)

        if sku["available_in_scrape"]:
            sku["initial_stock"] = rng.randint(STOCK_MIN_INITIAL, STOCK_MAX_INITIAL)
            sku["target_stock"] = sku["initial_stock"]
        else:
            sku["initial_stock"] = 0
            sku["target_stock"] = rng.randint(*UNAVAILABLE_TARGET_RANGE)

        sku["reorder_point"] = max(5, int(sku["target_stock"] * RESTOCK_THRESHOLD_RATIO))


def seasonal_factor(day_idx: int, date: datetime, subcategory: str) -> float:
    weekly = WEEKLY_FACTOR[date.weekday()]
    monthly = 1.0 + 0.08 * math.sin(2 * math.pi * day_idx / 30.0)
    sub = SUBCAT_SEASONAL.get(subcategory, {}).get(date.month, 1.0)
    return weekly * monthly * sub


def attraction_weight(sku: dict, day_idx: int, date: datetime) -> float:
    """base_demand * seasonality * mild trend — used to sample product focus."""
    trend = math.exp(sku["trend_slope"] * day_idx)
    return max(1e-9, sku["base_demand"] * seasonal_factor(day_idx, date, sku["subcategory"]) * trend)


def apply_substitution_hint(
    sku_id: str,
    stocks: dict[str, int],
    skus_by_variant: dict[str, list[str]],
    sku_by_id: dict[str, dict],
    rng: random.Random,
    substitution_rate: float = SUBSTITUTION_RATE,
) -> str:
    """If focused SKU is OOS, maybe redirect to an in-stock sibling size."""
    if stocks.get(sku_id, 0) > 0:
        return sku_id
    if rng.random() > substitution_rate:
        return sku_id
    variant = sku_by_id[sku_id]["variant_id"]
    siblings = [
        s for s in skus_by_variant.get(variant, [])
        if s != sku_id and stocks.get(s, 0) > 0
    ]
    if not siblings:
        return sku_id
    return rng.choice(siblings)


# =============================================================================
# SHOPPERS
# =============================================================================

@dataclass
class Shopper:
    shopper_id: str
    archetype: str
    # Soft prefs: preferred size label (may be None), preferred subcategory
    preferred_size: str | None = None
    preferred_subcat: str | None = None
    wishlist: set[str] = field(default_factory=set)
    bis_subs: set[str] = field(default_factory=set)
    price_drop_subs: set[str] = field(default_factory=set)
    purchased: set[str] = field(default_factory=set)


def assign_archetypes(n: int, rng: random.Random) -> list[str]:
    names = list(ARCHETYPE_MIX.keys())
    weights = [ARCHETYPE_MIX[a] for a in names]
    # Exact-ish multinomial via sequential rounding for reproducibility clarity
    counts = [int(round(n * w)) for w in weights]
    # Fix rounding so sum == n
    while sum(counts) > n:
        i = max(range(len(counts)), key=lambda j: counts[j] - n * weights[j])
        counts[i] -= 1
    while sum(counts) < n:
        i = min(range(len(counts)), key=lambda j: counts[j] - n * weights[j])
        counts[i] += 1
    archetypes: list[str] = []
    for name, c in zip(names, counts):
        archetypes.extend([name] * c)
    rng.shuffle(archetypes)
    return archetypes


def create_shoppers(n: int, skus: list[dict], rng: random.Random) -> list[Shopper]:
    archetypes = assign_archetypes(n, rng)
    sizes = sorted({s["size"] for s in skus})
    subcats = sorted({s["subcategory"] for s in skus})
    shoppers: list[Shopper] = []
    for arch in archetypes:
        pref_size = None
        if arch == "size_constrained" and sizes:
            # Prefer edge or mid sizes that often disappoint
            edge_mid = [s for s in sizes if is_edge_size(s) or is_mid_size(s)]
            pool = edge_mid or sizes
            pref_size = rng.choice(pool)
        elif rng.random() < 0.6 and sizes:
            # Soft size preference for others (weighted by size curve)
            sw = [size_weight(s) for s in sizes]
            pref_size = rng.choices(sizes, weights=sw, k=1)[0]
        pref_sub = rng.choice(subcats) if subcats and rng.random() < 0.4 else None
        shoppers.append(Shopper(
            shopper_id=str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            archetype=arch,
            preferred_size=pref_size,
            preferred_subcat=pref_sub,
        ))
    return shoppers


# =============================================================================
# SCHEDULED CONVERSION ATTEMPTS
# =============================================================================

@dataclass(order=True)
class ScheduledConversion:
    day_idx: int
    minute: int  # minute-of-day 0..1439
    shopper_id: str = field(compare=False)
    sku_id: str = field(compare=False)
    source: str = field(compare=False)  # cart | wishlist | bis | price_drop
    origin_session_id: str = field(compare=False)
    delayed: bool = field(compare=False, default=True)


# =============================================================================
# SESSION / EVENT SIMULATION
# =============================================================================

def _weighted_sku_sample(
    skus: list[dict],
    weights: list[float],
    shopper: Shopper,
    rng: random.Random,
) -> dict:
    """Sample a SKU with archetype / preference biases applied to weights."""
    beh = ARCHETYPE_BEHAVIOR[shopper.archetype]
    adj = list(weights)
    edge_boost = beh["size_edge_boost"]
    for i, sku in enumerate(skus):
        w = adj[i]
        if shopper.preferred_size and sku["size"] == shopper.preferred_size:
            w *= 3.0
        if shopper.preferred_subcat and sku["subcategory"] == shopper.preferred_subcat:
            w *= 1.8
        if edge_boost > 1.0 and (is_edge_size(sku["size"]) or is_mid_size(sku["size"])):
            w *= edge_boost
        # Slightly de-emphasize already purchased SKUs
        if sku["sku_id"] in shopper.purchased:
            w *= 0.15
        adj[i] = w
    return rng.choices(skus, weights=adj, k=1)[0]


def _schedule_conversion(
    pending: list[ScheduledConversion],
    shopper: Shopper,
    sku_id: str,
    source: str,
    day_idx: int,
    session_minute: int,
    origin_session_id: str,
    days: int,
    rng: random.Random,
    force_same_session: bool = False,
) -> tuple[bool, ScheduledConversion | None]:
    """
    Maybe schedule a purchase attempt. Returns (same_session_now, scheduled_or_none).
    If same-session purchase should fire immediately, returns (True, None) and caller
    should attempt purchase in-session.
    """
    beh = ARCHETYPE_BEHAVIOR[shopper.archetype]
    if force_same_session or rng.random() < beh["p_same_session_purchase"]:
        # Decisive / lucky path: attempt in this session
        return True, None

    # Delayed attempt
    mean_delay = beh["delay_days_mean"]
    # Source modulates delay: cart sooner, wishlist/price_drop later
    if source == "cart":
        mean_delay *= 0.6
    elif source == "wishlist":
        mean_delay *= 1.3
    elif source == "price_drop":
        mean_delay *= 1.5
    elif source == "bis":
        mean_delay *= 1.1

    delay_days = sample_count(mean_delay, rng)
    if delay_days == 0 and rng.random() < 0.5:
        # Later same day
        minute = min(1439, session_minute + rng.randint(30, 300))
        target_day = day_idx
    else:
        target_day = day_idx + max(1, delay_days)
        minute = rng.randint(60, 1200)
    if target_day >= days:
        return False, None
    sc = ScheduledConversion(
        day_idx=target_day,
        minute=minute,
        shopper_id=shopper.shopper_id,
        sku_id=sku_id,
        source=source,
        origin_session_id=origin_session_id,
        delayed=True,
    )
    pending.append(sc)
    return False, sc


def simulate_session(
    shopper: Shopper,
    day_idx: int,
    date: datetime,
    start_minute: int,
    skus: list[dict],
    day_weights: list[float],
    stocks: dict[str, int],
    sku_by_id: dict[str, dict],
    skus_by_variant: dict[str, list[str]],
    skus_by_product: dict[str, list[str]],
    pending_conversions: list[ScheduledConversion],
    days: int,
    rng: random.Random,
    event_writer,
    metrics_day: dict[str, dict],
    gt_day: dict[str, dict],
) -> dict:
    """
    Simulate one browsing session. Writes events via event_writer(callback).
    Returns session summary dict.
    """
    beh = ARCHETYPE_BEHAVIOR[shopper.archetype]
    session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    n_target = max(2, sample_count(beh["session_events_mean"], rng))
    minute = start_minute
    date_str = date.date().isoformat()

    events_emitted = 0
    purchased_same = 0
    focused_sku: dict | None = None
    focused_product_id: str | None = None
    viewed_products: set[str] = set()

    def emit(etype: str, sku: dict | None = None, extra: dict | None = None) -> dict:
        nonlocal events_emitted, minute
        events_emitted += 1
        minute = min(1439, minute + rng.randint(1, 4))
        ts = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
        row = {
            "event_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "shopper_id": shopper.shopper_id,
            "session_id": session_id,
            "date": date_str,
            "ts": ts.isoformat(timespec="seconds"),
            "minute_of_day": minute,
            "event_type": etype,
            "archetype": shopper.archetype,
            "product_id": sku["product_id"] if sku else None,
            "variant_id": sku["variant_id"] if sku else None,
            "sku_id": sku["sku_id"] if sku else None,
            "product_name": sku["product_name"] if sku else None,
            "colour": sku["colour"] if sku else None,
            "size": sku["size"] if sku else None,
            "price": sku["price"] if sku else None,
            "stock_at_event": stocks.get(sku["sku_id"], 0) if sku else None,
        }
        if extra:
            row.update(extra)
        event_writer(row)
        return row

    def bump_metric(sku_id: str, field: str, n: int = 1) -> None:
        m = metrics_day[sku_id]
        m[field] = m.get(field, 0) + n

    def bump_gt(sku_id: str, field: str, n: int = 1) -> None:
        g = gt_day[sku_id]
        g[field] = g.get(field, 0) + n

    def attempt_purchase(sku: dict, source: str, delayed: bool) -> bool:
        nonlocal purchased_same
        sid = sku["sku_id"]
        bump_gt(sid, "purchase_attempts", 1)
        if stocks.get(sid, 0) <= 0:
            bump_gt(sid, "lost_units", 1)
            emit(
                "purchase",
                sku,
                extra={
                    "purchase_status": "oos_blocked",
                    "purchase_source": source,
                    "delayed_purchase": delayed,
                },
            )
            # Strong OOS reaction → maybe BIS
            if sid not in shopper.bis_subs and rng.random() < beh["p_bis_when_oos"]:
                shopper.bis_subs.add(sid)
                emit("subscribe_back_in_stock", sku)
                bump_metric(sid, "add_back_in_stock_alert", 1)
                _schedule_conversion(
                    pending_conversions, shopper, sid, "bis",
                    day_idx, minute, session_id, days, rng,
                )
            return False

        stocks[sid] -= 1
        shopper.purchased.add(sid)
        purchased_same += 1 if not delayed else 0
        bump_metric(sid, "sales", 1)
        emit(
            "purchase",
            sku,
            extra={
                "purchase_status": "success",
                "purchase_source": source,
                "delayed_purchase": delayed,
            },
        )
        return True

    # --- opening event ---
    if rng.random() < beh["p_start_search"]:
        emit("search", extra={"query_subcat": shopper.preferred_subcat})
    else:
        # Land directly on a product
        focused_sku = _weighted_sku_sample(skus, day_weights, shopper, rng)
        focused_product_id = focused_sku["product_id"]
        viewed_products.add(focused_product_id)
        emit("product_view", focused_sku)
        bump_metric(focused_sku["sku_id"], "page_views", 1)

    # --- event loop ---
    while events_emitted < n_target:
        # Occasionally bounce early
        if events_emitted >= 2 and rng.random() < 0.08:
            break

        # Need a focused product?
        if focused_sku is None or (
            beh["p_multi_product"] > 0
            and rng.random() < beh["p_multi_product"] * 0.25
            and len(viewed_products) < 8
        ):
            focused_sku = _weighted_sku_sample(skus, day_weights, shopper, rng)
            # Sibling substitution if OOS
            alt_id = apply_substitution_hint(
                focused_sku["sku_id"], stocks, skus_by_variant, sku_by_id, rng,
            )
            focused_sku = sku_by_id[alt_id]
            focused_product_id = focused_sku["product_id"]
            if focused_product_id not in viewed_products:
                viewed_products.add(focused_product_id)
                emit("product_view", focused_sku)
                bump_metric(focused_sku["sku_id"], "page_views", 1)
                continue

        assert focused_sku is not None
        sid = focused_sku["sku_id"]
        in_stock = stocks.get(sid, 0) > 0

        # Choose next action from sku-focused state
        r = rng.random()
        if r < 0.35:
            # sku_view
            emit("sku_view", focused_sku)
            bump_metric(sid, "page_views", 1)
            if not in_stock:
                bump_gt(sid, "strong_oos_intents", 1)
                if sid not in shopper.bis_subs and rng.random() < beh["p_bis_when_oos"]:
                    shopper.bis_subs.add(sid)
                    emit("subscribe_back_in_stock", focused_sku)
                    bump_metric(sid, "add_back_in_stock_alert", 1)
                    _schedule_conversion(
                        pending_conversions, shopper, sid, "bis",
                        day_idx, minute, session_id, days, rng,
                    )
        elif r < 0.35 + beh["p_cart"] * 0.5:
            # add_to_cart path (scaled so probs stay sane across archetypes)
            emit("add_to_cart", focused_sku)
            bump_metric(sid, "add_to_cart", 1)
            if not in_stock:
                bump_gt(sid, "strong_oos_intents", 1)
            same_now, _ = _schedule_conversion(
                pending_conversions, shopper, sid, "cart",
                day_idx, minute, session_id, days, rng,
            )
            if same_now:
                attempt_purchase(focused_sku, source="cart", delayed=False)
                if shopper.archetype == "decisive" and rng.random() < 0.6:
                    break  # decisive often leaves after buying
        elif r < 0.35 + beh["p_cart"] * 0.5 + beh["p_wishlist"]:
            emit("add_to_wishlist", focused_sku)
            bump_metric(sid, "add_to_wishlist", 1)
            shopper.wishlist.add(sid)
            if not in_stock:
                bump_gt(sid, "strong_oos_intents", 1)
            _schedule_conversion(
                pending_conversions, shopper, sid, "wishlist",
                day_idx, minute, session_id, days, rng,
            )
        elif r < 0.35 + beh["p_cart"] * 0.5 + beh["p_wishlist"] + beh["p_price_drop"]:
            if sid not in shopper.price_drop_subs:
                shopper.price_drop_subs.add(sid)
                emit("subscribe_price_drop", focused_sku)
                bump_metric(sid, "add_price_drop_alert", 1)
                _schedule_conversion(
                    pending_conversions, shopper, sid, "price_drop",
                    day_idx, minute, session_id, days, rng,
                )
            else:
                emit("sku_view", focused_sku)
                bump_metric(sid, "page_views", 1)
        else:
            # Switch product (comparer / browser) or bounce-ish deepen
            if rng.random() < beh["p_multi_product"]:
                focused_sku = None  # force resample next loop
            else:
                emit("sku_view", focused_sku)
                bump_metric(sid, "page_views", 1)

    emit("bounce", focused_sku)

    return {
        "session_id": session_id,
        "shopper_id": shopper.shopper_id,
        "archetype": shopper.archetype,
        "date": date_str,
        "start_ts": (
            date.replace(hour=0, minute=0, second=0, microsecond=0)
            + timedelta(minutes=start_minute)
        ).isoformat(timespec="seconds"),
        "n_events": events_emitted,
        "n_products_viewed": len(viewed_products),
        "same_session_purchases": purchased_same,
    }


def process_delayed_conversion(
    sc: ScheduledConversion,
    shopper: Shopper,
    day_idx: int,
    date: datetime,
    stocks: dict[str, int],
    sku_by_id: dict[str, dict],
    pending_conversions: list[ScheduledConversion],
    days: int,
    rng: random.Random,
    event_writer,
    session_writer,
    metrics_day: dict[str, dict],
    gt_day: dict[str, dict],
) -> None:
    """Fire a delayed conversion attempt as its own tiny session."""
    if sc.sku_id not in sku_by_id:
        return
    sku = sku_by_id[sc.sku_id]
    beh = ARCHETYPE_BEHAVIOR[shopper.archetype]
    date_str = date.date().isoformat()
    session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    minute = sc.minute
    ts = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
    n_events = 0

    def emit(etype: str, extra: dict | None = None) -> None:
        nonlocal n_events, minute
        n_events += 1
        row = {
            "event_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "shopper_id": shopper.shopper_id,
            "session_id": session_id,
            "date": date_str,
            "ts": ts.isoformat(timespec="seconds"),
            "minute_of_day": minute,
            "event_type": etype,
            "archetype": shopper.archetype,
            "product_id": sku["product_id"],
            "variant_id": sku["variant_id"],
            "sku_id": sku["sku_id"],
            "product_name": sku["product_name"],
            "colour": sku["colour"],
            "size": sku["size"],
            "price": sku["price"],
            "stock_at_event": stocks.get(sku["sku_id"], 0),
        }
        if extra:
            row.update(extra)
        event_writer(row)

    # Light revisit: sku_view then purchase attempt
    emit("sku_view", extra={"return_visit": True, "return_source": sc.source})
    metrics_day[sc.sku_id]["page_views"] = metrics_day[sc.sku_id].get("page_views", 0) + 1

    gt_day[sc.sku_id]["purchase_attempts"] = gt_day[sc.sku_id].get("purchase_attempts", 0) + 1

    if stocks.get(sc.sku_id, 0) <= 0:
        gt_day[sc.sku_id]["lost_units"] = gt_day[sc.sku_id].get("lost_units", 0) + 1
        emit(
            "purchase",
            extra={
                "purchase_status": "oos_blocked",
                "purchase_source": sc.source,
                "delayed_purchase": True,
            },
        )
        if sc.sku_id not in shopper.bis_subs and rng.random() < beh["p_bis_when_oos"]:
            shopper.bis_subs.add(sc.sku_id)
            emit("subscribe_back_in_stock")
            metrics_day[sc.sku_id]["add_back_in_stock_alert"] = (
                metrics_day[sc.sku_id].get("add_back_in_stock_alert", 0) + 1
            )
            # Reschedule BIS follow-up
            _schedule_conversion(
                pending_conversions, shopper, sc.sku_id, "bis",
                day_idx, minute, session_id, days, rng,
            )
    else:
        stocks[sc.sku_id] -= 1
        shopper.purchased.add(sc.sku_id)
        metrics_day[sc.sku_id]["sales"] = metrics_day[sc.sku_id].get("sales", 0) + 1
        emit(
            "purchase",
            extra={
                "purchase_status": "success",
                "purchase_source": sc.source,
                "delayed_purchase": True,
            },
        )

    emit("bounce")
    session_writer({
        "session_id": session_id,
        "shopper_id": shopper.shopper_id,
        "archetype": shopper.archetype,
        "date": date_str,
        "start_ts": ts.isoformat(timespec="seconds"),
        "n_events": n_events,
        "n_products_viewed": 1,
        "same_session_purchases": 0,
        "is_return_conversion": True,
        "return_source": sc.source,
        "origin_session_id": sc.origin_session_id,
    })


# =============================================================================
# DAILY SALES FLOOR (fill shortfall to at least daily_sales successful purchases)
# =============================================================================

def inject_daily_sales_floor(
    day_idx: int,
    date: datetime,
    target: int,
    organic_sales: int,
    skus: list[dict],
    day_weights: list[float],
    stocks: dict[str, int],
    sku_by_id: dict[str, dict],
    shoppers: list[Shopper],
    rng: random.Random,
    event_writer,
    session_writer,
    metrics_day: dict[str, dict],
    gt_day: dict[str, dict],
) -> tuple[int, int]:
    """
    Ensure at least `target` successful purchases today (FLOOR semantics).

    If organic_sales >= target, do nothing. Otherwise inject short journey-shaped
    sessions (product_view → sku_view → purchase) attached to random shoppers,
    sampling in-stock SKUs by attraction weights so SKU mix varies by day.

    Returns (injected_success_count, remaining_shortfall_after_stock_limit).
    """
    shortfall = max(0, target - organic_sales)
    if shortfall <= 0:
        return 0, 0

    date_str = date.date().isoformat()
    injected = 0

    # In-stock pool rebuilt only when depleted entries accumulate (avoid O(n) each sale)
    in_stock_idxs = [i for i, s in enumerate(skus) if stocks.get(s["sku_id"], 0) > 0]
    if not in_stock_idxs:
        return 0, shortfall

    for _ in range(shortfall):
        # Drop any indices that went OOS since last rebuild
        live = [i for i in in_stock_idxs if stocks.get(skus[i]["sku_id"], 0) > 0]
        if len(live) < len(in_stock_idxs) * 0.5 or not live:
            in_stock_idxs = [i for i, s in enumerate(skus) if stocks.get(s["sku_id"], 0) > 0]
            live = in_stock_idxs
        else:
            in_stock_idxs = live
        if not in_stock_idxs:
            break
        weights = [day_weights[i] for i in in_stock_idxs]
        chosen_i = rng.choices(in_stock_idxs, weights=weights, k=1)[0]
        sku = skus[chosen_i]
        sid = sku["sku_id"]
        if stocks.get(sid, 0) <= 0:
            continue

        shopper = rng.choice(shoppers)
        session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        # Spread filler sales across the afternoon/evening so they sit after
        # typical organic browse minutes and stay chronologically plausible.
        minute = rng.randint(12 * 60, 22 * 60)
        n_events = 0
        start_minute = minute

        def emit(etype: str, extra: dict | None = None) -> None:
            nonlocal n_events, minute
            n_events += 1
            minute = min(1439, minute + rng.randint(1, 3))
            ts = (
                date.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(minutes=minute)
            )
            row = {
                "event_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "shopper_id": shopper.shopper_id,
                "session_id": session_id,
                "date": date_str,
                "ts": ts.isoformat(timespec="seconds"),
                "minute_of_day": minute,
                "event_type": etype,
                "archetype": shopper.archetype,
                "product_id": sku["product_id"],
                "variant_id": sku["variant_id"],
                "sku_id": sku["sku_id"],
                "product_name": sku["product_name"],
                "colour": sku["colour"],
                "size": sku["size"],
                "price": sku["price"],
                "stock_at_event": stocks.get(sku["sku_id"], 0),
                "injected_daily_sales": True,
            }
            if extra:
                row.update(extra)
            event_writer(row)

        # Short journey-shaped chain
        emit("product_view")
        metrics_day[sid]["page_views"] = metrics_day[sid].get("page_views", 0) + 1
        emit("sku_view")
        metrics_day[sid]["page_views"] = metrics_day[sid].get("page_views", 0) + 1

        # Successful purchase (stock already verified)
        gt_day[sid]["purchase_attempts"] = gt_day[sid].get("purchase_attempts", 0) + 1
        stocks[sid] -= 1
        shopper.purchased.add(sid)
        metrics_day[sid]["sales"] = metrics_day[sid].get("sales", 0) + 1
        emit(
            "purchase",
            extra={
                "purchase_status": "success",
                "purchase_source": "daily_sales_floor",
                "delayed_purchase": False,
            },
        )
        emit("bounce")

        session_writer({
            "session_id": session_id,
            "shopper_id": shopper.shopper_id,
            "archetype": shopper.archetype,
            "date": date_str,
            "start_ts": (
                date.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(minutes=start_minute)
            ).isoformat(timespec="seconds"),
            "n_events": n_events,
            "n_products_viewed": 1,
            "same_session_purchases": 1,
            "is_daily_sales_floor": True,
        })
        injected += 1

    remaining = shortfall - injected
    return injected, remaining


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def plan_session_days(
    shoppers: list[Shopper],
    days: int,
    rng: random.Random,
) -> dict[int, list[tuple[Shopper, int]]]:
    """
    For each day_idx, list of (shopper, start_minute) who browse that day.
    Session counts are archetype-dependent Poisson over the window, then
    days are sampled (with replacement allowed for multi-session days rarely).
    """
    by_day: dict[int, list[tuple[Shopper, int]]] = defaultdict(list)
    for shopper in shoppers:
        beh = ARCHETYPE_BEHAVIOR[shopper.archetype]
        n_sess = max(1, sample_count(beh["sessions_mean"] * (days / 90.0), rng))
        # Weight later/earlier days mildly by weekly traffic via day-of-week
        # (actual date applied later; here uniform-ish with slight clustering)
        for _ in range(n_sess):
            d = rng.randrange(days)
            # Wishlist fillers / deal hunters return more — already in sessions_mean
            start_minute = rng.randint(8 * 60, 22 * 60)
            by_day[d].append((shopper, start_minute))
    return by_day


def simulate_journeys(
    skus: list[dict],
    shoppers: list[Shopper],
    days: int,
    seed: int,
    out_dir: str,
    restock_lead_time_days: int = RESTOCK_LEAD_TIME_DAYS,
    start_date: datetime | None = None,
    daily_sales: int = 2000,
) -> dict[str, Any]:
    rng = random.Random(seed)
    if start_date is None:
        start_date = (
            datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=days)
        )

    sku_by_id = {s["sku_id"]: s for s in skus}
    skus_by_variant: dict[str, list[str]] = defaultdict(list)
    skus_by_product: dict[str, list[str]] = defaultdict(list)
    for s in skus:
        skus_by_variant[s["variant_id"]].append(s["sku_id"])
        skus_by_product[s["product_id"]].append(s["sku_id"])

    stocks = {s["sku_id"]: int(s["initial_stock"]) for s in skus}
    pending_restock: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pending_conversions: list[ScheduledConversion] = []

    shopper_by_id = {s.shopper_id: s for s in shoppers}
    sessions_by_day = plan_session_days(shoppers, days, rng)

    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, "events.jsonl")
    sessions_path = os.path.join(out_dir, "sessions.jsonl")
    metrics_path = os.path.join(out_dir, "daily_metrics.jsonl")
    gt_path = os.path.join(out_dir, "ground_truth.jsonl")

    # Streaming writers
    events_f = open(events_path, "w", encoding="utf-8")
    sessions_f = open(sessions_path, "w", encoding="utf-8")
    metrics_f = open(metrics_path, "w", encoding="utf-8")
    gt_f = open(gt_path, "w", encoding="utf-8")

    def write_jsonl(fh, row: dict) -> None:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    total_events = 0
    total_sessions = 0
    total_purchases_success = 0
    total_purchases_oos = 0
    same_session_purchases = 0
    delayed_purchases_success = 0
    total_injected_sales = 0
    funnel = defaultdict(int)
    daily_purchase_counts: list[int] = []
    daily_injected_counts: list[int] = []
    days_below_floor: list[tuple[str, int, int]] = []  # (date, achieved, target)

    def event_writer(row: dict) -> None:
        nonlocal total_events, total_purchases_success, total_purchases_oos
        nonlocal same_session_purchases, delayed_purchases_success
        total_events += 1
        et = row["event_type"]
        funnel[et] += 1
        if et == "purchase":
            if row.get("purchase_status") == "success":
                total_purchases_success += 1
                if row.get("delayed_purchase"):
                    delayed_purchases_success += 1
                else:
                    same_session_purchases += 1
            elif row.get("purchase_status") == "oos_blocked":
                total_purchases_oos += 1
        write_jsonl(events_f, row)

    def session_writer(row: dict) -> None:
        nonlocal total_sessions
        total_sessions += 1
        write_jsonl(sessions_f, row)

    try:
        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)
            date_str = date.date().isoformat()

            # Restock arrivals
            for sku_id, arrivals in list(pending_restock.items()):
                if day_idx in arrivals:
                    stocks[sku_id] += arrivals.pop(day_idx)

            # Snapshot stock_start
            stock_start = {sid: int(stocks[sid]) for sid in stocks}

            # Per-day metric / GT accumulators
            metrics_day: dict[str, dict] = {
                sid: {
                    "page_views": 0,
                    "add_to_cart": 0,
                    "add_to_wishlist": 0,
                    "add_back_in_stock_alert": 0,
                    "add_price_drop_alert": 0,
                    "sales": 0,
                }
                for sid in stocks
            }
            gt_day: dict[str, dict] = {
                sid: {
                    "purchase_attempts": 0,
                    "lost_units": 0,
                    "strong_oos_intents": 0,
                }
                for sid in stocks
            }

            # Attraction weights for today
            day_weights = [attraction_weight(s, day_idx, date) for s in skus]

            # Merge today's delayed conversions + browse sessions, process in time order
            todays_conv = [sc for sc in pending_conversions if sc.day_idx == day_idx]
            pending_conversions[:] = [sc for sc in pending_conversions if sc.day_idx != day_idx]
            todays_conv.sort()

            browse = list(sessions_by_day.get(day_idx, []))
            # Represent browse as (minute, kind, payload)
            timeline: list[tuple[int, str, Any]] = []
            for sc in todays_conv:
                timeline.append((sc.minute, "conv", sc))
            for shopper, start_minute in browse:
                timeline.append((start_minute, "browse", (shopper, start_minute)))
            timeline.sort(key=lambda x: x[0])

            for _minute, kind, payload in timeline:
                if kind == "conv":
                    sc: ScheduledConversion = payload
                    shopper = shopper_by_id[sc.shopper_id]
                    process_delayed_conversion(
                        sc, shopper, day_idx, date, stocks, sku_by_id,
                        pending_conversions, days, rng,
                        event_writer, session_writer, metrics_day, gt_day,
                    )
                else:
                    shopper, start_minute = payload
                    summary = simulate_session(
                        shopper, day_idx, date, start_minute,
                        skus, day_weights, stocks, sku_by_id,
                        skus_by_variant, skus_by_product,
                        pending_conversions, days, rng,
                        event_writer, metrics_day, gt_day,
                    )
                    session_writer(summary)

            # --- Daily sales FLOOR: fill shortfall to at least daily_sales ---
            # Organic journeys may already exceed the floor; we only inject when below.
            organic_sales = sum(int(m.get("sales", 0)) for m in metrics_day.values())
            injected, remaining = inject_daily_sales_floor(
                day_idx=day_idx,
                date=date,
                target=daily_sales,
                organic_sales=organic_sales,
                skus=skus,
                day_weights=day_weights,
                stocks=stocks,
                sku_by_id=sku_by_id,
                shoppers=shoppers,
                rng=rng,
                event_writer=event_writer,
                session_writer=session_writer,
                metrics_day=metrics_day,
                gt_day=gt_day,
            )
            day_sales = organic_sales + injected
            daily_purchase_counts.append(day_sales)
            daily_injected_counts.append(injected)
            total_injected_sales += injected
            if remaining > 0:
                days_below_floor.append((date_str, day_sales, daily_sales))
                print(
                    f"  [WARN] {date_str}: only {day_sales} successful purchases "
                    f"(target floor {daily_sales}; stock limited, shortfall {remaining})",
                    flush=True,
                )

            # End-of-day restock triggers + flush metrics / GT
            for sku in skus:
                sid = sku["sku_id"]
                end_stock = int(stocks[sid])
                if end_stock <= sku["reorder_point"]:
                    qty = sku["target_stock"] - end_stock
                    if qty > 0:
                        pending_restock[sid][day_idx + restock_lead_time_days] += qty

                m = metrics_day[sid]
                common = {
                    "date": date_str,
                    "sku_id": sid,
                    "variant_id": sku["variant_id"],
                    "product_id": sku["product_id"],
                    "product_name": sku["product_name"],
                    "colour": sku["colour"],
                    "size": sku["size"],
                    "subcategory": sku["subcategory"],
                }
                metrics_row = {
                    **common,
                    "stock_start": stock_start[sid],
                    "stock_end": end_stock,
                    "sales": int(m["sales"]),
                    "page_views": int(m["page_views"]),
                    "add_to_cart": int(m["add_to_cart"]),
                    "add_to_wishlist": int(m["add_to_wishlist"]),
                    "add_back_in_stock_alert": int(m["add_back_in_stock_alert"]),
                    "add_price_drop_alert": int(m["add_price_drop_alert"]),
                    "price": sku["price"],
                }
                write_jsonl(metrics_f, metrics_row)

                g = gt_day[sid]
                attempts = int(g["purchase_attempts"])
                sales = int(m["sales"])
                lost = int(g["lost_units"])
                strong = int(g["strong_oos_intents"])
                journey_demand = attempts + strong
                if attempts > 0:
                    avail = sales / attempts
                else:
                    avail = 1.0 if stock_start[sid] > 0 else 0.0
                gt_row = {
                    **common,
                    "purchase_attempts": attempts,
                    "sales": sales,
                    "lost_units": lost,
                    "strong_oos_intents": strong,
                    "journey_demand": journey_demand,
                    "availability": round(avail, 4),
                    "stock_start": stock_start[sid],
                    "stock_end": end_stock,
                }
                write_jsonl(gt_f, gt_row)

            if (day_idx + 1) % max(1, days // 10) == 0 or day_idx == days - 1:
                print(
                    f"  [day {day_idx + 1}/{days}] events={total_events:,} "
                    f"sessions={total_sessions:,} purchases_ok={total_purchases_success:,} "
                    f"day_sales={day_sales:,} (injected={injected:,})",
                    flush=True,
                )
    finally:
        events_f.close()
        sessions_f.close()
        metrics_f.close()
        gt_f.close()

    return {
        "total_events": total_events,
        "total_sessions": total_sessions,
        "total_purchases_success": total_purchases_success,
        "total_purchases_oos": total_purchases_oos,
        "same_session_purchases": same_session_purchases,
        "delayed_purchases_success": delayed_purchases_success,
        "total_injected_sales": total_injected_sales,
        "daily_purchase_counts": daily_purchase_counts,
        "daily_injected_counts": daily_injected_counts,
        "days_below_floor": days_below_floor,
        "daily_sales_target": daily_sales,
        "funnel": dict(funnel),
        "events_path": events_path,
        "sessions_path": sessions_path,
        "metrics_path": metrics_path,
        "gt_path": gt_path,
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def write_shoppers(path_jsonl: str, path_json: str, shoppers: list[Shopper]) -> None:
    rows = []
    with open(path_jsonl, "w", encoding="utf-8") as f:
        for s in shoppers:
            row = {
                "shopper_id": s.shopper_id,
                "archetype": s.archetype,
                "preferred_size": s.preferred_size,
                "preferred_subcat": s.preferred_subcat,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))


def write_catalog(path: str, skus: list[dict]) -> None:
    catalog_out = [
        {k: s[k] for k in (
            "sku_id", "variant_id", "product_id", "product_name",
            "colour", "size", "subcategory", "price", "available_in_scrape",
        )}
        for s in skus
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog_out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[+] Wrote {len(catalog_out):,} SKUs to {path}")


def maybe_write_parquet(jsonl_path: str) -> str | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_json(jsonl_path, lines=True)
        pq = jsonl_path.replace(".jsonl", ".parquet")
        df.to_parquet(pq, index=False)
        print(f"[+] Wrote parquet {pq} ({os.path.getsize(pq):,} bytes)")
        return pq
    except Exception as e:  # noqa: BLE001 — optional path
        print(f"[!] parquet skip for {jsonl_path}: {e}")
        return None


def print_summary(
    shoppers: list[Shopper],
    stats: dict[str, Any],
    days: int,
    out_dir: str,
) -> None:
    print("\n--- Summary (shopper journeys) ---")
    print(f"Days:                 {days}")
    print(f"Shoppers:             {len(shoppers):,}")
    print(f"Sessions:             {stats['total_sessions']:,}")
    print(f"Events:               {stats['total_events']:,}")
    print(f"Purchases (success):  {stats['total_purchases_success']:,}")
    print(f"Purchases (OOS block):{stats['total_purchases_oos']:,}")
    print(f"  same-session ok:    {stats['same_session_purchases']:,}")
    print(f"  delayed ok:         {stats['delayed_purchases_success']:,}")
    print(f"  injected (floor):   {stats.get('total_injected_sales', 0):,}")
    target = stats.get("daily_sales_target", 0)
    dpc = stats.get("daily_purchase_counts") or []
    if dpc:
        mean_p = sum(dpc) / len(dpc)
        print(f"Daily sales floor:    {target:,} (at-least / fill shortfall)")
        print(
            f"Purchases/day:        min={min(dpc):,}  mean={mean_p:,.1f}  "
            f"max={max(dpc):,}"
        )
        below = stats.get("days_below_floor") or []
        if below:
            print(f"Days below floor:     {len(below)} (stock-limited)")
            for d, ach, tgt in below[:10]:
                print(f"  {d}: {ach} < {tgt}")
            if len(below) > 10:
                print(f"  ... and {len(below) - 10} more")
        else:
            print("Days below floor:     0")

    print("\nArchetype counts:")
    counts: dict[str, int] = defaultdict(int)
    for s in shoppers:
        counts[s.archetype] += 1
    for a in ARCHETYPE_MIX:
        print(f"  {a:20s} {counts[a]:6d}  ({counts[a]/len(shoppers):.1%})")

    print("\nEvent-type funnel totals:")
    for et in EVENT_TYPES:
        print(f"  {et:30s} {stats['funnel'].get(et, 0):10,d}")

    print("\nSanity checks:")
    uniq = len({s.shopper_id for s in shoppers})
    print(f"  {'PASS' if uniq == len(shoppers) else 'FAIL'} unique shoppers ({uniq})")
    mix_ok = all(
        (a == "decisive" and abs(counts[a] / len(shoppers) - 0.05) < 0.02)
        or (a != "decisive" and 0.06 <= counts[a] / len(shoppers) <= 0.22)
        for a in ARCHETYPE_MIX
    )
    print(f"  {'PASS' if mix_ok else 'WARN'} archetype mix within band")
    pv = stats["funnel"].get("product_view", 0) + stats["funnel"].get("sku_view", 0)
    cart = stats["funnel"].get("add_to_cart", 0)
    wish = stats["funnel"].get("add_to_wishlist", 0)
    pdrop = stats["funnel"].get("subscribe_price_drop", 0)
    funnel_ok = pv >= cart and cart >= 0 and wish >= 0
    print(
        f"  {'PASS' if funnel_ok else 'WARN'} views({pv:,}) >= cart({cart:,}); "
        f"wishlist={wish:,} pricedrop={pdrop:,}"
    )
    print(
        f"  {'PASS' if stats['total_purchases_success'] > 0 else 'WARN'} "
        f"at least one successful purchase"
    )
    dpc = stats.get("daily_purchase_counts") or []
    target = stats.get("daily_sales_target", 0)
    if dpc and target > 0:
        below = stats.get("days_below_floor") or []
        floor_ok = len(below) == 0 and min(dpc) >= target
        label = "PASS" if floor_ok else ("WARN" if below else "PASS")
        # PASS if every day >= target OR only WARN when stock-limited shortfalls
        if below:
            label = "WARN"
        elif min(dpc) >= target:
            label = "PASS"
        else:
            label = "WARN"
        print(
            f"  {label} daily sales floor: min/day={min(dpc):,} "
            f"vs target>={target:,} (injected={stats.get('total_injected_sales', 0):,})"
        )
    print(f"\nOutputs under: {out_dir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate shopper-centric journey dataset (vs demand-centric PV generator)."
    )
    parser.add_argument("catalog_json", help="Path to scraped products JSON.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--n-shoppers", type=int, default=50000)
    parser.add_argument(
        "--daily-sales", type=int, default=2000,
        help=(
            "Minimum successful purchase events per calendar day (FLOOR). "
            "Organic journeys may exceed this; shortfalls are filled with "
            "injected journey-shaped sales. Default: %(default)s"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./demand_shopper")
    parser.add_argument(
        "--restock-lead-days", type=int, default=RESTOCK_LEAD_TIME_DAYS,
        help="Days between restock trigger and arrival. Default: %(default)s",
    )
    parser.add_argument(
        "--no-parquet", action="store_true",
        help="Skip parquet sidecars even if pandas/pyarrow are available.",
    )
    args = parser.parse_args()

    # Validate mix
    mix_sum = sum(ARCHETYPE_MIX.values())
    if abs(mix_sum - 1.0) > 1e-6:
        print(f"[!] ARCHETYPE_MIX sums to {mix_sum}, expected 1.0", file=sys.stderr)
        return 1

    skus = load_skus(args.catalog_json)
    print(f"[+] Loaded {len(skus)} SKU(s) from {args.catalog_json}")
    if not skus:
        print("[!] No SKUs found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    assign_priors(skus, rng)
    # Shopper UUIDs use a derived RNG so catalog priors stay aligned with seed
    shopper_rng = random.Random(args.seed + 17)
    shoppers = create_shoppers(args.n_shoppers, skus, shopper_rng)
    print(
        f"[+] Created {len(shoppers)} shoppers; "
        f"simulating {args.days} day(s), seed={args.seed}"
    )
    print(
        f"[+] Daily sales floor: {args.daily_sales} successful purchases/day "
        f"(fill shortfall; organic may exceed)"
    )
    print(f"[+] Archetype mix: {ARCHETYPE_MIX}")
    print(f"[+] Event types: {EVENT_TYPES}")

    os.makedirs(args.out_dir, exist_ok=True)
    write_catalog(os.path.join(args.out_dir, "catalog.json"), skus)
    write_shoppers(
        os.path.join(args.out_dir, "shoppers.jsonl"),
        os.path.join(args.out_dir, "shoppers.json"),
        shoppers,
    )
    print(f"[+] Wrote shoppers → {args.out_dir}/shoppers.jsonl")

    stats = simulate_journeys(
        skus=skus,
        shoppers=shoppers,
        days=args.days,
        seed=args.seed + 99,
        out_dir=args.out_dir,
        restock_lead_time_days=args.restock_lead_days,
        daily_sales=args.daily_sales,
    )

    if not args.no_parquet:
        maybe_write_parquet(stats["metrics_path"])
        maybe_write_parquet(stats["gt_path"])

    print_summary(shoppers, stats, args.days, args.out_dir)
    print("\n[+] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
