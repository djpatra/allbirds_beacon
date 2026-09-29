#!/usr/bin/env python3
"""CLI for the merchant RAG PoC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merchant_rag.orchestrator import MerchantRAG
from merchant_rag.templates import TOP_N, list_questions


def default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "demand" / "product_categories.parquet",
        root / "demand" / "catalog.json",
    )


def main() -> int:
    cat_default, catalog_default = default_paths()
    parser = argparse.ArgumentParser(description="Allbirds merchant intelligence PoC")
    parser.add_argument("--categories", default=str(cat_default))
    parser.add_argument("--catalog", default=str(catalog_default))
    parser.add_argument("--query", "-q", default=None, help="Merchant question text")
    parser.add_argument("--all", action="store_true", help="Run all 6 predefined queries")
    parser.add_argument("--list", action="store_true", help="List predefined questions")
    parser.add_argument("--n", type=int, default=TOP_N)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to persist answers",
    )
    args = parser.parse_args()

    if args.list:
        for i, q in enumerate(list_questions(), 1):
            print(f"{i}. {q}")
        return 0

    rag = MerchantRAG(args.categories, catalog_path=args.catalog)

    if args.all:
        answers = rag.ask_all(n=args.n)
    elif args.query:
        answers = [rag.ask(args.query, n=args.n)]
    else:
        parser.print_help()
        print("\nPredefined questions:")
        for q in list_questions():
            print(f"  • {q}")
        return 1

    for ans in answers:
        print("=" * 72)
        print(rag.render_text(ans))
        print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # products already serializable
        out.write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
        print(f"[+] Wrote {len(answers)} answer(s) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
