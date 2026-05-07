"""
backend/rag/retrieval.py
========================
Hybrid Retrieval Engine
=======================
Combines:
  1. Semantic search — FAISS cosine similarity on dense embeddings
  2. Structured filtering — attribute-level SQL-like filtering (price, category, brand, stock)
  3. Re-ranking — combined score = semantic score * recency_boost * rating_boost
  4. Deduplication — remove duplicate product IDs
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from db.vector_store import FAISSVectorStore
from rag.embeddings import embedding_engine

logger = logging.getLogger("rag.retrieval")


def _attribute_filter(
    product: Dict[str, Any],
    max_price: Optional[float],
    min_price: Optional[float],
    category: Optional[str],
    brand: Optional[str],
    in_stock_only: bool,
) -> bool:
    """Return True if product passes all attribute filters."""
    if max_price is not None and product.get("price", 0) > max_price:
        return False
    if min_price is not None and product.get("price", 0) < min_price:
        return False
    if category and category.lower() not in product.get("category", "").lower():
        return False
    if brand and brand.lower() not in product.get("brand", "").lower():
        return False
    if in_stock_only and product.get("stock", 0) <= 0:
        return False
    return True


def _rerank_score(semantic_score: float, product: Dict[str, Any]) -> float:
    """
    Composite re-ranking score.

    Formula: semantic_score * (1 + 0.05 * rating_normalised) * stock_boost
      - rating_normalised: scales 0–5 rating to 0–1
      - stock_boost: 1.0 if in stock, 0.9 if out of stock
    """
    rating = product.get("rating", 4.0)
    rating_boost = 1.0 + 0.05 * (rating / 5.0)
    stock_boost = 1.0 if product.get("stock", 0) > 0 else 0.9
    return semantic_score * rating_boost * stock_boost


def _match_reason(query: str, product: Dict[str, Any], score: float) -> str:
    """Generate a brief human-readable explanation of why this product matched."""
    reasons = []
    q_lower = query.lower()
    name_lower = product.get("name", "").lower()
    desc_lower = product.get("description", "").lower()
    tags = [t.lower() for t in product.get("tags", [])]

    # Exact name match
    for word in q_lower.split():
        if len(word) > 3 and word in name_lower:
            reasons.append(f"Name match: '{word}'")
            break

    # Tag match
    matched_tags = [t for t in tags if t in q_lower or any(w in t for w in q_lower.split() if len(w) > 3)]
    if matched_tags:
        reasons.append(f"Tag: {matched_tags[0]}")

    # High semantic score
    if score > 0.7:
        reasons.append("High semantic similarity")
    elif score > 0.5:
        reasons.append("Good semantic match")

    # Category match
    cat = product.get("category", "")
    if cat.lower() in q_lower:
        reasons.append(f"Category: {cat}")

    return " | ".join(reasons) if reasons else f"Semantic similarity {score:.2f}"


class HybridRetriever:
    """
    Hybrid retrieval combining FAISS semantic search with attribute filtering and re-ranking.

    Retrieval flow:
    1. Embed query → dense vector
    2. FAISS top-(top_k * 3) retrieval (over-retrieve for filtering headroom)
    3. Apply attribute filters
    4. Re-rank by composite score
    5. Deduplicate by product ID
    6. Return top_k results
    """

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self.vs = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        in_stock_only: bool = False,
        query_vector: Optional[np.ndarray] = None,  # pre-computed (image)
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval.

        Parameters
        ----------
        query        : Raw user query text
        top_k        : Number of final results to return
        query_vector : Optional pre-computed embedding (e.g. from image encoder)

        Returns
        -------
        List of result dicts: {product, score, rank, match_reason}
        """
        t0 = time.time()

        # Step 1: Embed query
        if query_vector is None:
            query_vector = embedding_engine.embed_text(query)

        # Step 2: Semantic search (over-retrieve for filter headroom)
        candidates = self.vs.search(query_vector, top_k=top_k * 4)

        # Step 3: Attribute filtering
        filtered = [
            (p, s) for p, s in candidates
            if _attribute_filter(p, max_price, min_price, category, brand, in_stock_only)
        ]

        # Step 4: Re-rank
        reranked = sorted(
            [(p, _rerank_score(s, p)) for p, s in filtered],
            key=lambda x: x[1],
            reverse=True,
        )

        # Step 5: Deduplicate
        seen_ids: set[str] = set()
        deduplicated = []
        for p, score in reranked:
            pid = p.get("id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                deduplicated.append((p, score))

        # Step 6: Take top_k and format results
        final = deduplicated[:top_k]
        results = []
        for rank, (product, score) in enumerate(final, 1):
            results.append({
                "product": product,
                "score": round(score, 4),
                "rank": rank,
                "match_reason": _match_reason(query, product, score),
            })

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Retrieval: query=%r | candidates=%d → filtered=%d → returned=%d | %.1fms",
            query[:50], len(candidates), len(filtered), len(results), elapsed_ms,
        )
        return results
