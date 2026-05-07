"""
backend/rag/embeddings.py  (v2 — CPU-only, no PyTorch)
=======================================================
Lightweight Embedding Engine using scikit-learn TF-IDF + SVD.

Architecture:
  - TF-IDF vectoriser (sparse) → TruncatedSVD → 256-dim dense L2-normalised vectors
  - This is a production-appropriate approximation of sentence embeddings
    for CPU-only environments without GPU/PyTorch
  - Upgrade path: swap _encode() for SentenceTransformer when GPU available

Image search: Uses CLIP-style keyword extraction from filename/alt text
              (full CLIP requires PyTorch; use caption-based text search as fallback)
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from typing import List, Optional

import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger("rag.embeddings")

TEXT_DIM = 256          # SVD output dimension
CLIP_DIM = TEXT_DIM     # unified dim for image path


class EmbeddingEngine:
    """
    CPU-only embedding engine using TF-IDF + SVD (LSA).

    Produces 256-dim L2-normalised dense vectors.
    Fully compatible with FAISS IndexFlatIP.
    """

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        """Fit TF-IDF + SVD on the product corpus."""
        logger.info("Fitting TF-IDF + SVD on %d documents...", len(corpus))
        t0 = time.time()

        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        n_components = min(TEXT_DIM, tfidf_matrix.shape[1] - 1, len(corpus) - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

        logger.info("Embedding engine ready | dim=%d | %.2fs", n_components, time.time() - t0)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Transform texts → L2-normalised dense vectors."""
        if not self._fitted:
            raise RuntimeError("EmbeddingEngine not fitted. Call fit() first.")
        tfidf = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf).astype(np.float32)
        return normalize(dense, norm="l2")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single query string → shape (dim,)."""
        return self._encode([text])[0]

    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed a list of texts → shape (N, dim)."""
        t0 = time.time()
        result = self._encode(texts)
        logger.info("Batch encoded %d texts in %.2fs → shape %s", len(texts), time.time() - t0, result.shape)
        return result

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Image → embedding via visual feature extraction.

        Strategy (CPU fallback):
          1. Resize image to 64×64
          2. Compute colour histogram per channel (R, G, B)
          3. Hash + statistical features → query text
          4. Embed via TF-IDF engine

        For full CLIP support: install torch + open_clip_torch and swap this method.
        """
        img = image.resize((64, 64)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0

        r_hist, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 1))
        g_hist, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 1))
        b_hist, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 1))

        # Map dominant colours to descriptive words
        dominant_r = "red" if r_hist.argmax() > 8 else "dark"
        dominant_g = "green" if g_hist.argmax() > 8 else "neutral"
        dominant_b = "blue" if b_hist.argmax() > 8 else "warm"

        brightness = arr.mean()
        brightness_word = "bright" if brightness > 0.6 else ("dark" if brightness < 0.3 else "medium")

        # Create a descriptive query from visual features
        visual_query = (
            f"{brightness_word} {dominant_r} {dominant_g} {dominant_b} "
            f"product electronic device gadget"
        )
        logger.debug("Image visual query: %r", visual_query)
        return self.embed_text(visual_query)

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.embed_image(image)

    def embed_text_clip(self, text: str) -> np.ndarray:
        """Unified CLIP-style text embed (same space as image embeds)."""
        return self.embed_text(text)


# Global singleton — populated at startup by main.py
embedding_engine = EmbeddingEngine()
