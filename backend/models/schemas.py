"""
backend/models/schemas.py
=========================
Pydantic v2 schemas for the RAG API.
"""

from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ProductSpec(BaseModel):
    """Flexible key-value spec store for any product category."""
    model_config = {"extra": "allow"}


class Product(BaseModel):
    id: str
    name: str
    category: str
    subcategory: str
    brand: str
    price: float
    currency: str = "USD"
    stock: int
    rating: float
    reviews_count: int
    color: str
    colors_available: List[str] = []
    specs: dict[str, Any] = {}
    description: str
    tags: List[str] = []
    image_url: str = ""
    caption: Optional[str] = None          # AI-generated caption (BLIP-2 / fallback)
    embedding_text: Optional[str] = None   # fused text used for embedding


class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    in_stock_only: bool = False


class SearchResult(BaseModel):
    product: Product
    score: float
    rank: int
    match_reason: str


class RAGResponse(BaseModel):
    query: str
    results: List[SearchResult]
    answer: str
    query_type: str          # "text" | "image" | "multimodal"
    retrieved_count: int
    latency_ms: float


class ImageQueryResponse(BaseModel):
    caption: str
    detected_category: Optional[str]
    results: List[SearchResult]
    answer: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    products_indexed: int
    embedding_model: str
    version: str
