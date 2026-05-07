"""
backend/main.py
===============
FastAPI Application — Multimodal RAG E-Commerce Assistant
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from db.vector_store import FAISSVectorStore
from models.schemas import HealthResponse, RAGResponse, SearchResult, TextQuery
from rag.embeddings import TEXT_DIM, CLIP_DIM, embedding_engine
from rag.ingestion import load_products
from rag.llm import generate_answer
from rag.retrieval import HybridRetriever

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ── Global state ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "products.json"
vector_store: FAISSVectorStore = None
retriever: HybridRetriever = None


def _index_products() -> None:
    """Load, enrich, fit embeddings, and index all products."""
    global vector_store, retriever

    # Load + enrich products first
    products = load_products(DATA_PATH)
    texts = [p["embedding_text"] for p in products]

    # Fit TF-IDF + SVD on the full corpus (must happen before any embed call)
    embedding_engine.fit(texts)

    actual_dim = embedding_engine._svd.n_components
    vector_store = FAISSVectorStore(dim=actual_dim, index_name="products")

    # Try to load existing index (only valid if dim matches)
    if vector_store.load() and vector_store.total == len(products):
        logger.info("Loaded existing FAISS index (%d products)", vector_store.total)
        retriever = HybridRetriever(vector_store)
        return

    # Build index from scratch
    logger.info("Building FAISS index from product catalog...")
    vectors = embedding_engine.embed_texts_batch(texts)
    vector_store = FAISSVectorStore(dim=vectors.shape[1], index_name="products")
    vector_store.add(vectors, products)
    vector_store.save()

    retriever = HybridRetriever(vector_store)
    logger.info("FAISS index built: %d products indexed", vector_store.total)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multimodal RAG E-Commerce Assistant...")
    _index_products()
    logger.info("Ready ✓")
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG E-Commerce Assistant",
    description="Self-hosted AI shopping assistant with text + image search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """API health check with index status."""
    return HealthResponse(
        status="ok",
        vector_store=f"FAISS IndexFlatIP ({vector_store.total if vector_store else 0} vectors)",
        products_indexed=vector_store.total if vector_store else 0,
        embedding_model="all-MiniLM-L6-v2 (384-dim)",
        version="1.0.0",
    )


@app.post("/search/text", response_model=RAGResponse)
async def search_text(query: TextQuery):
    """
    Text-based product search with RAG answer generation.

    Supports:
    - Natural language queries
    - Price range filtering
    - Category / brand filtering
    - Stock filtering
    """
    t0 = time.time()

    if retriever is None:
        raise HTTPException(503, "Search index not ready")

    results = retriever.retrieve(
        query=query.query,
        top_k=query.top_k,
        max_price=query.max_price,
        min_price=query.min_price,
        category=query.category,
        brand=query.brand,
        in_stock_only=query.in_stock_only,
    )

    answer = generate_answer(query.query, results)

    return RAGResponse(
        query=query.query,
        results=[SearchResult(**r) for r in results],
        answer=answer,
        query_type="text",
        retrieved_count=len(results),
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/search/image")
async def search_image(
    image: UploadFile = File(...),
    query: str = Form(default="Find similar products"),
    top_k: int = Form(default=5),
):
    """
    Image-based product search.

    Pipeline:
    1. Receive uploaded image
    2. Generate caption (lightweight fallback)
    3. Embed image using CLIP
    4. Search FAISS with image embedding
    5. Generate RAG answer
    """
    t0 = time.time()

    if retriever is None:
        raise HTTPException(503, "Search index not ready")

    # Load image
    img_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # Embed image using CLIP — note: falls back gracefully if CLIP not loaded
    try:
        image_vector = embedding_engine.embed_image(pil_image)
    except Exception as e:
        logger.warning("CLIP image embedding failed, using text fallback: %s", e)
        image_vector = embedding_engine.embed_text(query)

    # Retrieve using image embedding
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        query_vector=image_vector,
    )

    answer = generate_answer(f"[Image search] {query}", results)

    return JSONResponse({
        "query": query,
        "query_type": "image",
        "results": results,
        "answer": answer,
        "retrieved_count": len(results),
        "latency_ms": round((time.time() - t0) * 1000, 1),
    })


@app.get("/search/quick")
async def quick_search(
    q: str,
    top_k: int = 5,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
):
    """Quick GET endpoint for lightweight text search (no answer generation)."""
    t0 = time.time()
    if retriever is None:
        raise HTTPException(503, "Search index not ready")

    results = retriever.retrieve(
        query=q, top_k=top_k, max_price=max_price, category=category
    )
    return {
        "query": q,
        "results": results,
        "count": len(results),
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/products/all")
async def get_all_products():
    """Return all indexed products (for UI catalog display)."""
    if vector_store is None:
        raise HTTPException(503, "Index not ready")
    return {"products": vector_store.metadata, "count": len(vector_store.metadata)}


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Multimodal RAG E-Commerce Assistant", "docs": "/docs"}
