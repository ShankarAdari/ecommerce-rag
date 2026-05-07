"""
backend/db/vector_store.py
==========================
FAISS Vector Store
==================
Manages the in-memory / on-disk FAISS index for semantic search.

Index type: IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)
Dimensionality: 384 (Sentence Transformers text) or 512 (CLIP)

Design decisions:
  - IndexFlatIP: exact search, optimal for catalogs < 1M products
  - For > 1M products, upgrade to IndexIVFFlat (approximate, needs training)
  - All vectors normalised before insertion → dot product == cosine similarity
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger("db.vector_store")

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


class FAISSVectorStore:
    """
    Wrapper around a FAISS IndexFlatIP with metadata sidecar storage.

    Attributes
    ----------
    dim       : Embedding dimensionality (384 for text, 512 for CLIP)
    index     : FAISS index object
    metadata  : List of product dicts aligned to index row positions
    """

    def __init__(self, dim: int = 384, index_name: str = "products") -> None:
        self.dim = dim
        self.index_name = index_name
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("FAISSVectorStore ready | dim=%d | index=%s", dim, index_name)

    # ── Write operations ──────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors and their metadata to the index.

        Parameters
        ----------
        vectors  : float32 array of shape (N, dim), L2-normalised
        metadata : list of N product dicts
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        assert vectors.shape[1] == self.dim, (
            f"Vector dim mismatch: got {vectors.shape[1]}, expected {self.dim}"
        )
        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.info(
            "Added %d vectors | total indexed: %d",
            len(metadata), self.index.ntotal,
        )

    # ── Search operations ─────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-K nearest neighbors.

        Returns list of (metadata_dict, cosine_score) tuples, sorted descending.
        """
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        q = query_vector.reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS uses -1 for empty slots
                continue
            if score < score_threshold:
                continue
            results.append((self.metadata[idx], float(score)))

        logger.debug(
            "Search returned %d results (top score: %.4f)",
            len(results), results[0][1] if results else 0,
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        index_path = INDEX_DIR / f"{self.index_name}.faiss"
        meta_path = INDEX_DIR / f"{self.index_name}_meta.pkl"
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Index saved to %s (%d vectors)", index_path, self.index.ntotal)

    def load(self) -> bool:
        """Load persisted index from disk. Returns True if found."""
        index_path = INDEX_DIR / f"{self.index_name}.faiss"
        meta_path = INDEX_DIR / f"{self.index_name}_meta.pkl"
        if not index_path.exists() or not meta_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Loaded index from disk: %d vectors", self.index.ntotal)
        return True

    @property
    def total(self) -> int:
        return self.index.ntotal
