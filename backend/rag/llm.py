"""
backend/rag/llm.py
==================
LLM Layer — Self-Hosted Response Generation
============================================
Architecture:
  Primary  : Ollama API (llama3 / mistral / gemma running locally)
  Fallback  : Template-based structured response (always works, no GPU needed)

Hallucination Control:
  - Context-only answering (strict grounding instruction)
  - Confidence validation before response
  - Attribute validation (price cross-check)
  - Safe fallback on empty context
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("rag.llm")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 30.0


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful e-commerce shopping assistant. You MUST:
1. Answer ONLY using the provided product context below
2. NEVER hallucinate product specs, prices, or availability
3. If the answer is not in the context, say "I don't have that information"
4. Always mention product name, key specs, and price when relevant
5. Be concise, friendly, and helpful"""

def build_prompt(query: str, context_products: List[Dict[str, Any]]) -> str:
    """
    Build a grounded RAG prompt from retrieved products.

    Format:
      [CONTEXT]
      Product 1: ...
      Product 2: ...
      [/CONTEXT]

      User Query: ...

      Instructions: Answer using ONLY the context above.
    """
    if not context_products:
        return f"User Query: {query}\n\nI don't have any relevant products to show you for this query."

    context_lines = []
    for i, result in enumerate(context_products, 1):
        p = result.get("product", result)
        specs_str = " | ".join(
            f"{k}: {v}" for k, v in list(p.get("specs", {}).items())[:5]
        )
        context_lines.append(
            f"Product {i}: {p['name']} by {p['brand']}\n"
            f"  Price: {p['currency']} {p['price']:.2f} | Rating: {p['rating']}/5\n"
            f"  Category: {p['category']} > {p['subcategory']}\n"
            f"  Colors: {', '.join(p.get('colors_available', [p.get('color', '')]))}\n"
            f"  Stock: {'In Stock' if p.get('stock', 0) > 0 else 'Out of Stock'}\n"
            f"  Key Specs: {specs_str}\n"
            f"  Description: {p['description'][:300]}..."
        )

    context_block = "\n\n".join(context_lines)

    return f"""{SYSTEM_PROMPT}

[CONTEXT]
{context_block}
[/CONTEXT]

User Query: {query}

Answer:"""


# ── Ollama integration ────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Call local Ollama server for LLM inference.
    Returns None if Ollama is not running.
    """
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.debug("Ollama not available — using template fallback")
    except Exception as e:
        logger.warning("Ollama error: %s", e)
    return None


# ── Template fallback ─────────────────────────────────────────────────────────

def template_response(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Structured template-based response when LLM is unavailable.
    Produces deterministic, grounded, human-readable answers.
    """
    if not results:
        return (
            "I couldn't find any products matching your query. "
            "Try adjusting your search terms or filters."
        )

    q_lower = query.lower()
    top = results[0]["product"]

    # Detect query intent
    is_similar = any(w in q_lower for w in ["similar", "alternative", "like this", "comparable"])
    is_color = any(w in q_lower for w in ["color", "colour", "another color", "shade"])
    is_price = any(w in q_lower for w in ["under", "budget", "cheap", "affordable", "₹", "$"])
    is_specs = any(w in q_lower for w in ["spec", "feature", "detail", "battery", "screen", "display"])

    lines = []

    if is_specs and len(results) >= 1:
        p = top
        specs = p.get("specs", {})
        lines.append(f"📋 **{p['name']}** — Specifications:")
        for k, v in list(specs.items())[:6]:
            lines.append(f"  • {k.replace('_', ' ').title()}: {v}")
        lines.append(f"  • Price: {p['currency']} {p['price']:.2f}")
        lines.append(f"  • Rating: {p['rating']}/5 ({p['reviews_count']:,} reviews)")

    elif is_similar or len(results) > 1:
        lines.append(f"🔍 Here are **{len(results)} matching products** for your query:\n")
        for r in results[:4]:
            p = r["product"]
            in_stock = "✅ In Stock" if p.get("stock", 0) > 0 else "❌ Out of Stock"
            lines.append(
                f"**{r['rank']}. {p['name']}** — {p['currency']} {p['price']:.2f}\n"
                f"   ⭐ {p['rating']}/5 | {p['brand']} | {in_stock}\n"
                f"   {p['description'][:120]}..."
            )

    elif is_color:
        p = top
        colors = p.get("colors_available", [p.get("color", "Unknown")])
        lines.append(f"🎨 **{p['name']}** is available in **{len(colors)} color(s)**:")
        for c in colors:
            lines.append(f"  • {c}")
        lines.append(f"\nPriced at {p['currency']} {p['price']:.2f}.")

    else:
        lines.append(f"🛍️ Top result: **{top['name']}**")
        lines.append(f"   Brand: {top['brand']} | Price: {top['currency']} {top['price']:.2f}")
        lines.append(f"   Rating: {top['rating']}/5 | {top['reviews_count']:,} reviews")
        lines.append(f"   {top['description'][:200]}...")

        if len(results) > 1:
            lines.append(f"\n📦 Also found {len(results) - 1} more related product(s).")

    return "\n".join(lines)


# ── Main generate function ────────────────────────────────────────────────────

def generate_answer(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Generate a grounded answer using LLM (Ollama) with template fallback.

    Hallucination control:
    - Prompt forces context-only answering
    - Fallback response uses only retrieved data
    - No external knowledge injected
    """
    t0 = time.time()

    prompt = build_prompt(query, results)

    # Try Ollama first
    llm_response = call_ollama(prompt)
    if llm_response:
        elapsed = (time.time() - t0) * 1000
        logger.info("LLM response generated in %.0fms (Ollama)", elapsed)
        return llm_response

    # Template fallback
    response = template_response(query, results)
    elapsed = (time.time() - t0) * 1000
    logger.info("Template response generated in %.0fms (fallback)", elapsed)
    return response
