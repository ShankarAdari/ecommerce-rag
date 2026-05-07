"""
backend/rag/ingestion.py
========================
Data Ingestion Pipeline
=======================
Responsibilities:
  1. Load products from JSON / CSV
  2. Build the semantic fusion text (specs → natural language)
  3. Generate AI captions for products (lightweight fallback when BLIP-2 unavailable)
  4. Prepare records for embedding + indexing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("rag.ingestion")


# ── Spec-to-text fusion ───────────────────────────────────────────────────────

def specs_to_text(specs: Dict[str, Any]) -> str:
    """Convert a structured specs dict into a human-readable sentence."""
    parts = []
    for k, v in specs.items():
        key_label = k.replace("_", " ").title()
        if isinstance(v, bool):
            if v:
                parts.append(key_label)
        elif isinstance(v, (int, float)):
            parts.append(f"{key_label}: {v}")
        elif isinstance(v, str):
            parts.append(f"{key_label}: {v}")
        elif isinstance(v, list):
            parts.append(f"{key_label}: {', '.join(str(x) for x in v)}")
    return ". ".join(parts)


def build_embedding_text(product: Dict[str, Any]) -> str:
    """
    Fuse all product fields into a single rich text for embedding.

    Strategy: name → brand → category → description → specs → tags → colors
    This gives the embedding model maximum semantic signal.
    """
    parts = [
        f"Product: {product['name']}",
        f"Brand: {product['brand']}",
        f"Category: {product['category']} > {product['subcategory']}",
        f"Price: {product['currency']} {product['price']:.2f}",
        f"Color: {product['color']}",
        f"Rating: {product['rating']}/5 from {product['reviews_count']} reviews",
        product["description"],
        f"Specifications: {specs_to_text(product.get('specs', {}))}",
        f"Tags: {', '.join(product.get('tags', []))}",
        f"Available colors: {', '.join(product.get('colors_available', []))}",
    ]
    return " | ".join(parts)


def generate_caption(product: Dict[str, Any]) -> str:
    """
    Lightweight caption generator.

    In production: replace with BLIP-2 inference for actual image → caption.
    Here we build a structured caption from product metadata that approximates
    what BLIP-2 would output given the product image.
    """
    specs = product.get("specs", {})
    key_specs = []
    priority_keys = ["battery_life", "screen_size", "resolution", "capacity",
                     "material", "weight", "waterproof", "connectivity"]
    for k in priority_keys:
        if k in specs:
            key_specs.append(f"{k.replace('_', ' ')}: {specs[k]}")

    caption = (
        f"A {product['color'].lower()} {product['brand']} {product['name']} — "
        f"{product['subcategory']} priced at {product['currency']} {product['price']:.2f}."
    )
    if key_specs:
        caption += f" Key features: {', '.join(key_specs[:3])}."
    return caption


# ── Main ingestion function ───────────────────────────────────────────────────

def load_products(data_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load and enrich products from a JSON file.

    Steps:
    1. Parse raw JSON
    2. Build embedding text (structured → semantic fusion)
    3. Generate caption (simulated BLIP-2 output)
    4. Return enriched product list
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Product data not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    enriched = []
    for p in raw_products:
        p["embedding_text"] = build_embedding_text(p)
        p["caption"] = generate_caption(p)
        enriched.append(p)

    logger.info("Loaded and enriched %d products from %s", len(enriched), data_path)
    return enriched
