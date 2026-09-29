"""Row retrievers for each merchant intent (observables only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .templates import TOP_N


def _top(df: pd.DataFrame, score_col: str, n: int = TOP_N) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(score_col, ascending=False).head(n).reset_index(drop=True)


def restock_first(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Restock urgency = missed-opportunity loss (units × price) over lookback.

    Loss is the primary sort key — it naturally encodes urgency. Rising interest
    is a boost, not a hard filter (a SKU can have large historical lost sales
    even if the latest 14d trend cooled after prolonged OOS).
    """
    df = pc.copy()
    lost_rev = df["lost_revenue"].fillna(0) if "lost_revenue" in df.columns else 0
    lost_u = df["lost_units"].fillna(0) if "lost_units" in df.columns else 0

    # Candidates: any realized loss, or unavailable / A-tagged styles
    mask = (
        (lost_u > 0)
        | (lost_rev > 0)
        | df.get("A_availability_risk", False)
        | (df.get("u_stockout", 0) >= 0.05)
        | (df.get("catalog_size_gap", 0) >= 0.30)
    )
    out = df.loc[mask].copy()
    if out.empty:
        out = df.copy()

    interest = out["interest_trend"].clip(lower=0) if "interest_trend" in out.columns else 0
    out["_rank"] = (
        out["lost_revenue"].fillna(0)
        * (1.0 + interest)
        + out["lost_units"].fillna(0)  # tie-break in units
        + 0.001 * out.get("A_score", pd.Series(0, index=out.index)).fillna(0)
    )
    return _top(out, "_rank", n)


def findability_gap(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Well stocked but weakly engaged / attribute gaps (discovery)."""
    mask = (
        pc["B_discovery_opportunity"]
        | (
            (pc["r_stockout"] <= 0.10)
            & (pc["catalog_size_gap"] >= 0.30)
            & (pc["r_pv"] >= pc["r_pv"].median())
        )
    )
    out = pc.loc[mask].copy()
    return _top(out, "B_score", n)


def price_resistance(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """High wl/cart/pdalert, slow sales, in stock — vs subcategory peers."""
    df = pc.copy()
    if "price" not in df.columns:
        df["price"] = np.nan

    intent = df["r_wl"] + df["r_ca"] + df["r_pdalert"]
    df["price_intent"] = intent
    # peer sales rate in subcategory
    peer = (
        df.groupby("subcategory")["r_sales"]
        .transform(lambda s: s.mean())
        .replace(0, np.nan)
    )
    df["peer_sales_rate"] = peer
    df["price_score"] = (
        np.log1p(intent)
        * (1.0 + np.log1p(df["r_pdalert"]))
        / np.log1p(df["r_sales"] + 1)
        * (df["r_stockout"] <= 0.10).astype(float).clip(lower=0.5)
    )
    mask = (
        (intent >= intent.quantile(0.60))
        & (df["r_sales"] <= df["r_sales"].median())
        & (df["r_stockout"] <= 0.15)
        & (df["r_pdalert"] + df["r_wl"] > 0)
    )
    out = df.loc[mask].copy()
    if out.empty:
        out = df.loc[intent > 0].copy()
    return _top(out, "price_score", n)


def pdp_weak(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """High traffic + weak actions (landed but did nothing)."""
    mask = pc["C_needs_investigation"] | (
        (pc["r_pv"] >= pc["r_pv"].quantile(0.75))
        & (pc["r_stockout"] <= 0.10)
        & (pc["r_actions"] <= pc["r_actions"].quantile(0.50))
    )
    out = pc.loc[mask].copy()
    return _top(out, "C_score", n)


def dead_stock(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Fat inventory + flat/falling PV + low actions."""
    df = pc.copy()
    stock = df["mean_stock"].fillna(df.get("r_mean_stock", 0))
    df["dead_score"] = (
        np.log1p(stock)
        * (1.0 + (-df["pv_trend"]).clip(lower=0))
        / np.log1p(df["r_actions"] + df["r_sales"] + 1)
    )
    mask = (
        (stock >= stock.quantile(0.60))
        & (df["pv_trend"] <= 0.05)
        & (df["r_actions"] <= df["r_actions"].median())
    )
    out = df.loc[mask].copy()
    return _top(out, "dead_score", n)


def rising_star(pc: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Rising interest, still available, some actions — protect inventory."""
    df = pc.copy()
    df["star_score"] = (
        (1.0 + df["interest_trend"].clip(lower=0))
        * np.log1p(df["r_actions"])
        * np.log1p(df["r_pv"])
        * (1.0 - df["r_stockout"].clip(0, 1))
    )
    mask = (
        (df["interest_trend"] >= 0.10)
        & (df["r_stockout"] <= 0.10)
        & (df["r_actions"] >= df["r_actions"].median())
        & (~df["A_availability_risk"])
    )
    out = df.loc[mask].copy()
    if out.empty:
        out = df.loc[df["interest_trend"] > 0].copy()
    return _top(out, "star_score", n)


HANDLERS = {
    "restock_first": restock_first,
    "findability_gap": findability_gap,
    "price_resistance": price_resistance,
    "pdp_weak": pdp_weak,
    "dead_stock": dead_stock,
    "rising_star": rising_star,
}
