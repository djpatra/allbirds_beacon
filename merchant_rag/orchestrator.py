"""
Fake-LLM orchestrator (LangChain-shaped PoC, no tokenizer / no LLM).

Flow:
  merchant query text
    → keyword router (mimics intent classification)
    → retriever over product_categories
    → predefined answer template
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import retrievers
from .templates import TEMPLATES, TOP_N, list_questions


class MerchantRAG:
    """Minimal orchestration framework around curated product category data."""

    def __init__(self, categories_path: str | Path, catalog_path: str | Path | None = None):
        self.categories_path = Path(categories_path)
        self.pc = self._load_categories(self.categories_path)
        if catalog_path and Path(catalog_path).exists():
            self.pc = self._attach_price(self.pc, Path(catalog_path))
        elif "price" not in self.pc.columns:
            self.pc["price"] = pd.NA

    @staticmethod
    def _load_categories(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported categories file: {path}")

    @staticmethod
    def _attach_price(pc: pd.DataFrame, catalog_path: Path) -> pd.DataFrame:
        if catalog_path.suffix == ".json":
            cat = pd.DataFrame(json.loads(catalog_path.read_text(encoding="utf-8")))
        elif catalog_path.suffix == ".parquet":
            cat = pd.read_parquet(catalog_path)
        else:
            cat = pd.read_csv(catalog_path)
        if "price" not in cat.columns or "product_id" not in cat.columns:
            return pc
        price = cat.groupby("product_id")["price"].median().rename("price")
        out = pc.drop(columns=["price"], errors="ignore").merge(
            price, on="product_id", how="left"
        )
        return out

    def available_queries(self) -> list[dict[str, str]]:
        return [{"id": t["id"], "question": t["question"]} for t in TEMPLATES]

    def route(self, query: str) -> dict[str, Any]:
        """Mimic intent classification: keyword score → best template."""
        q = (query or "").lower()
        best, best_score = None, -1
        for t in TEMPLATES:
            score = sum(1 for kw in t["keywords"] if kw in q)
            # light boost if full question paraphrased
            if t["id"].replace("_", " ") in q:
                score += 2
            if score > best_score:
                best, best_score = t, score
        if best is None or best_score <= 0:
            return {
                "matched": False,
                "message": "I only handle predefined merchant questions.",
                "try": list_questions(),
            }
        return {"matched": True, "template": best, "route_score": best_score}

    def retrieve(self, template: dict[str, Any], n: int = TOP_N) -> pd.DataFrame:
        fn = retrievers.HANDLERS[template["handler"]]
        rows = fn(self.pc, n=n)
        cols = [c for c in template["columns"] if c in rows.columns]
        # always keep ids for traceability
        for extra in ("product_id", "product_name"):
            if extra in rows.columns and extra not in cols:
                cols.insert(0 if extra == "product_id" else 1, extra)
        # dedupe columns
        seen = set()
        cols = [c for c in cols if not (c in seen or seen.add(c))]
        return rows[cols] if cols else rows

    def format_answer(
        self,
        template: dict[str, Any],
        rows: pd.DataFrame,
    ) -> dict[str, Any]:
        n = len(rows)
        table = rows.copy()
        # round floats for readability
        for c in table.select_dtypes(include="number").columns:
            table[c] = table[c].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
        return {
            "query_id": template["id"],
            "question": template["question"],
            "headline": template["headline"].format(n=n),
            "why_it_matters": template["why"],
            "recommended_action": template["action"],
            "top_n": n,
            "products": table.to_dict(orient="records"),
            "source": str(self.categories_path.name),
        }

    def ask(self, query: str, n: int = TOP_N) -> dict[str, Any]:
        """End-to-end: route → retrieve → format (the 'chain')."""
        routed = self.route(query)
        if not routed.get("matched"):
            return routed
        template = routed["template"]
        rows = self.retrieve(template, n=n)
        answer = self.format_answer(template, rows)
        answer["route_score"] = routed["route_score"]
        return answer

    def ask_all(self, n: int = TOP_N) -> list[dict[str, Any]]:
        return [self.ask(t["question"], n=n) for t in TEMPLATES]

    def render_text(self, answer: dict[str, Any]) -> str:
        if not answer.get("query_id"):
            lines = [answer.get("message", "No match."), "", "Try:"]
            for q in answer.get("try", []):
                lines.append(f"  • {q}")
            return "\n".join(lines)
        lines = [
            f"Q. {answer['question']}",
            f"",
            f"{answer['headline']}",
            f"Why: {answer['why_it_matters']}",
            f"Action: {answer['recommended_action']}",
            f"",
            f"Top {answer['top_n']} products:",
        ]
        for i, row in enumerate(answer["products"], 1):
            name = row.get("product_name", row.get("product_id", "?"))
            # pick 3 informative metrics if present
            bits = []
            for k in (
                "lost_revenue", "lost_units", "urgency",
                "A_score", "B_score", "C_score", "price_score", "dead_score",
                "star_score", "interest_trend", "r_pv", "r_sales", "price",
                "catalog_size_gap", "u_stockout",
            ):
                if k in row and row[k] is not None and k != "product_name":
                    bits.append(f"{k}={row[k]}")
                if len(bits) >= 3:
                    break
            extra = f" ({', '.join(bits)})" if bits else ""
            lines.append(f"  {i}. {name}{extra}")
        return "\n".join(lines)
