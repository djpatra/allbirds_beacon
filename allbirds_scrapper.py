#!/usr/bin/env python3
"""
Allbirds Shopify Product Scraper
================================

Given one or more Allbirds product page URLs, this script:

  1. Discovers every sibling colour variant of the same product
     (Allbirds models each colour as its own Shopify product page).
  2. Scrapes each colour page via the public Shopify `.js` endpoint.
  3. Extracts product name, type, price, description, variants,
     and available/unavailable sizes for every colour.
  4. Writes results to JSON incrementally, with atomic writes so a
     crash mid-run never corrupts the output file.
  5. Optionally resumes from a previous run, skipping URLs already
     scraped successfully.

Usage:
    python allbirds_scraper.py urls.txt
    python allbirds_scraper.py urls.txt -o results.json
    python allbirds_scraper.py urls.txt --sample
    python allbirds_scraper.py urls.txt --save-every 5 --no-resume

Input file format:
    One product URL per line. Blank lines and lines starting with `#`
    are ignored. Any colour of a product is sufficient — the script
    fans out to all sibling colours automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from urllib.parse import urlparse, urljoin

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

REQUEST_TIMEOUT = 30          # seconds per HTTP request
RETRY_ATTEMPTS = 3            # attempts per URL before giving up
RETRY_DELAY = 2               # base seconds for exponential backoff
POLITE_DELAY = 0.5            # seconds between successive colour scrapes

# Allbirds uses a consistent handle prefix for each model. If you target a
# different model (e.g. "mens-wool-runner"), update this prefix accordingly.
COLOUR_HANDLE_PREFIX = "womens-cruiser-slip-on-"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, as_json: bool = False):
    """GET a URL with retries. Returns parsed JSON or raw text."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json() if as_json else resp.text
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _build_js_url(product_url: str) -> str:
    """Convert a product page URL into its Shopify `.js` JSON endpoint."""
    parsed = urlparse(product_url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}.js"


# ---------------------------------------------------------------------------
# Text / formatting helpers
# ---------------------------------------------------------------------------

def _clean_description(raw_html: str) -> str:
    """Strip HTML tags and collapse whitespace in a product description."""
    if not raw_html:
        return ""
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_price(cents) -> str:
    """Convert Shopify's cent-based price into a display string."""
    if cents is None:
        return ""
    return f"£{cents / 100:.2f}"


def _colour_label_from_url(url: str, prefix: str = COLOUR_HANDLE_PREFIX) -> str:
    """
    Derive a human-friendly colour label from a product handle.
    e.g. '.../womens-cruiser-slip-on-deep-navy' -> 'Deep Navy'
    """
    handle = urlparse(url).path.rstrip("/").split("/")[-1]
    slug = handle[len(prefix):] if handle.startswith(prefix) else handle
    return slug.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Colour-variant discovery
# ---------------------------------------------------------------------------

def discover_colour_urls(
    product_url: str,
    prefix: str = COLOUR_HANDLE_PREFIX,
) -> list[str]:
    """
    Fetch a product page and extract sibling colour-variant URLs.

    Allbirds renders a colour selector whose links point at sibling Shopify
    products sharing the same handle prefix but with different colour
    suffixes. We harvest every distinct same-host URL matching that pattern.

    The input URL is always included as a fallback, so the function never
    returns an empty list.
    """
    html = _fetch(product_url)
    parsed = urlparse(product_url)

    pattern = re.compile(
        r'href=["\']((?:https?://[^"\']+)?/products/'
        + re.escape(prefix)
        + r'[a-z0-9\-]+)["\']',
        re.IGNORECASE,
    )

    found: set[str] = set()
    for match in pattern.finditer(html):
        raw = match.group(1)
        absolute = urljoin(f"{parsed.scheme}://{parsed.netloc}", raw)
        if urlparse(absolute).netloc == parsed.netloc:
            found.add(absolute.rstrip("/"))

    found.add(product_url.rstrip("/"))
    return sorted(found)


# ---------------------------------------------------------------------------
# Per-colour scraping
# ---------------------------------------------------------------------------

def _split_sizes(variants: list[dict]) -> dict[str, list[str]]:
    """Split a flat Shopify variant list into available / unavailable sizes."""
    available, unavailable = [], []
    for v in variants:
        size = v.get("option1") or v.get("title") or "Unknown"
        (available if v.get("available") else unavailable).append(size)
    return {"available": available, "unavailable": unavailable}


def scrape_colour(product_url: str) -> dict:
    """
    Scrape a single colour page and return its size availability.

    Returns:
        {
            "colour": "Deep Navy",
            "url": "...",
            "available_sizes": [...],
            "unavailable_sizes": [...]
        }
    """
    data = _fetch(_build_js_url(product_url), as_json=True)

    if not isinstance(data, dict) or "variants" not in data:
        raise ValueError(f"Unexpected JSON shape from {product_url}")

    sizes = _split_sizes(data.get("variants", []))
    return {
        "colour": _colour_label_from_url(product_url),
        "url": product_url,
        "available_sizes": sizes["available"],
        "unavailable_sizes": sizes["unavailable"],
    }


# ---------------------------------------------------------------------------
# Main product scraper
# ---------------------------------------------------------------------------

def scrape_product(product_url: str) -> dict:
    """
    Scrape a product and all its sibling colour variants.

    Returns a single record with product-level metadata taken from the
    starting URL, plus a `colour_variants` mapping keyed by colour name.
    """
    print(f"[+] Discovering colour variants from {product_url}")
    colour_urls = discover_colour_urls(product_url)
    print(f"    -> found {len(colour_urls)} colour page(s)")

    # Product-level metadata from the starting page
    main_data = _fetch(_build_js_url(product_url), as_json=True)

    colour_variants: dict[str, dict] = {}
    for curl in colour_urls:
        try:
            print(f"    -> scraping {curl}")
            record = scrape_colour(curl)
            colour_variants[record["colour"]] = {
                "url": record["url"],
                "available_sizes": record["available_sizes"],
                "unavailable_sizes": record["unavailable_sizes"],
            }
        except Exception as exc:
            print(f"    [!] Failed {curl}: {exc}", file=sys.stderr)
            colour_variants[_colour_label_from_url(curl)] = {
                "url": curl,
                "error": str(exc),
            }
        time.sleep(POLITE_DELAY)

    # Flat variant summary from the starting page
    variant_summary = [
        {
            "id": v.get("id"),
            "title": v.get("title"),
            "sku": v.get("sku"),
            "available": bool(v.get("available")),
            "price": _format_price(v.get("price")),
            "option1": v.get("option1"),
            "option2": v.get("option2"),
            "option3": v.get("option3"),
        }
        for v in main_data.get("variants", [])
    ]

    return {
        "source_url": product_url,
        "product_name": main_data.get("title", ""),
        "product_type": main_data.get("type", ""),
        "price": _format_price(main_data.get("price")),
        "description": _clean_description(main_data.get("description", "")),
        "variants": variant_summary,
        "colour_variants": colour_variants,
    }


# ---------------------------------------------------------------------------
# Incremental, atomic JSON persistence
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data) -> None:
    """
    Write JSON atomically: temp file in the same directory, fsync, then
    os.replace() into place. A crash mid-write leaves the previous file
    intact rather than a truncated/corrupt one.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _load_existing_results(path: str) -> dict[str, dict]:
    """
    Load previously scraped results for resume. Returns a dict keyed by
    source_url -> record. Missing or malformed files yield {}.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if not isinstance(existing, list):
            return {}
        return {
            item["source_url"]: item
            for item in existing
            if isinstance(item, dict) and "source_url" in item
        }
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def scrape_multiple(
    urls: list[str],
    output_path: str | None = None,
    resume: bool = True,
    save_every: int = 1,
) -> list[dict]:
    """
    Scrape a list of product URLs, saving to `output_path` incrementally.

    Args:
        urls:        List of product page URLs.
        output_path: If provided, JSON is rewritten here after every
                     `save_every` products (and immediately on any error).
        resume:      If True and output_path exists, skip URLs already
                     present in the file with a successful record.
        save_every:  Flush to disk every N products. Default 1.

    Returns:
        A list of product dicts (successful) and error dicts (failed),
        ordered to match the input URLs.
    """
    existing = _load_existing_results(output_path) if (resume and output_path) else {}
    results_by_url: dict[str, dict] = dict(existing)

    # Preserve input order, appending any URLs only present in the old file
    ordered_urls = list(urls)
    for url in existing:
        if url not in ordered_urls:
            ordered_urls.append(url)

    completed_since_save = 0

    for i, url in enumerate(urls, start=1):
        if resume and url in existing and "error" not in existing[url]:
            print(f"[=] Skipping (already scraped): {url}")
            continue

        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            product = scrape_product(url)
            results_by_url[url] = product
            print(f"    -> {product['product_name']}")
        except Exception as exc:
            print(f"    [!] Failed: {exc}", file=sys.stderr)
            results_by_url[url] = {"source_url": url, "error": str(exc)}

        completed_since_save += 1

        if output_path and (
            completed_since_save >= save_every
            or "error" in results_by_url[url]
        ):
            snapshot = [results_by_url[u] for u in ordered_urls if u in results_by_url]
            _atomic_write_json(output_path, snapshot)
            completed_since_save = 0

    if output_path:
        snapshot = [results_by_url[u] for u in ordered_urls if u in results_by_url]
        _atomic_write_json(output_path, snapshot)
        print(f"\n[+] Wrote {len(snapshot)} record(s) to {output_path}")

    return [results_by_url[u] for u in ordered_urls if u in results_by_url]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_urls_from_file(path: str) -> list[str]:
    """Read newline-separated URLs, skipping blanks and `#` comments."""
    urls: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Allbirds (Shopify) product pages into JSON, "
                    "including per-colour size availability.",
    )
    parser.add_argument(
        "urls_file",
        help="Path to a text file containing one product URL per line.",
    )
    parser.add_argument(
        "-o", "--output",
        default="allbirds_products.json",
        help="Output JSON file path (default: allbirds_products.json).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print a truncated sample of the first result to stdout.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Flush the output file every N products (default: 1).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing output file and re-scrape everything.",
    )
    args = parser.parse_args()

    try:
        urls = _load_urls_from_file(args.urls_file)
    except OSError as exc:
        print(f"[!] Could not read {args.urls_file}: {exc}", file=sys.stderr)
        return 1

    if not urls:
        print(f"[!] No URLs found in {args.urls_file}", file=sys.stderr)
        return 1

    print(f"[+] Loaded {len(urls)} URL(s) from {args.urls_file}")

    results = scrape_multiple(
        urls,
        output_path=args.output,
        resume=not args.no_resume,
        save_every=args.save_every,
    )

    if args.sample and results and "error" not in results[0]:
        print("\n--- Sample output (first product, truncated) ---")
        print(json.dumps(results[0], indent=2, ensure_ascii=False)[:4000])

    return 0


if __name__ == "__main__":
    sys.exit(main())
