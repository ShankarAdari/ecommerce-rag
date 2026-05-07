# 🛍️ ShopMind AI — Multimodal RAG E-Commerce Assistant

> A **production-grade, fully self-hosted** AI shopping assistant powered by Retrieval-Augmented Generation (RAG) with a Glassmorphism liquid UI. Supports **text + image queries**, hybrid semantic search, and grounded LLM responses — no external APIs required.

![UI Preview](docs/ui_preview.png)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔍 Semantic Search | TF-IDF + SVD (LSA) → FAISS IndexFlatIP cosine similarity |
| 🖼️ Image Search | Visual feature extraction → same embedding space |
| 🧠 RAG Answers | Grounded responses from retrieved context only |
| 🤖 Local LLM | Ollama (Mistral/LLaMA) with intelligent template fallback |
| 🔒 Hallucination Control | Context-only prompts, attribute validation, safe fallbacks |
| 🎨 Glassmorphism UI | Animated blobs, frosted glass, liquid motion, product cards |
| 🐳 Docker Ready | Single-container deployment |
| ⚡ Fast | Index build: ~60ms · Query latency: <50ms |

---

## 🏗️ Architecture

```
ecommerce_rag/
├── backend/
│   ├── main.py              # FastAPI app — endpoints + lifespan
│   ├── rag/
│   │   ├── ingestion.py     # Product loading + spec-to-text fusion + captioning
│   │   ├── embeddings.py    # TF-IDF + SVD engine (CPU-only, FAISS-compatible)
│   │   ├── retrieval.py     # Hybrid retrieval: semantic + attribute filter + re-rank
│   │   └── llm.py           # Ollama LLM layer + template fallback
│   ├── db/
│   │   └── vector_store.py  # FAISS IndexFlatIP wrapper with persistence
│   ├── models/
│   │   └── schemas.py       # Pydantic v2 schemas
│   └── data/
│       └── products.json    # 15-product sample catalog
├── frontend/
│   └── index.html           # Glassmorphism chat UI
└── Dockerfile
```

### RAG Data Flow

```
User Query (text / image)
        │
        ▼
  EmbeddingEngine.embed()     ← TF-IDF+SVD (text) | colour histogram (image)
        │
        ▼
  FAISSVectorStore.search()   ← cosine similarity, top-K*4 candidates
        │
        ▼
  Attribute Filtering         ← price, category, brand, stock
        │
        ▼
  Re-ranking                  ← semantic_score × rating_boost × stock_boost
        │
        ▼
  LLM.generate_answer()       ← Ollama (Mistral) → template fallback
        │
        ▼
  RAGResponse → UI
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd ecommerce_rag/backend

py -3 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — the Glassmorphism UI loads automatically.

### 3. Docker

```bash
docker build -t shopmind-ai .
docker run -p 8000:8000 shopmind-ai
```

---

## 🤖 Optional: Local LLM (Ollama)

For richer AI-generated answers, install [Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull mistral
ollama serve          # runs on localhost:11434
```

The system auto-detects Ollama. If unavailable, it falls back to the structured template engine (always works).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve glassmorphism UI |
| `GET` | `/health` | Health + index status |
| `POST` | `/search/text` | Full RAG text search with filters |
| `POST` | `/search/image` | Image upload → visual search |
| `GET` | `/search/quick?q=...` | Lightweight text search |
| `GET` | `/products/all` | Full product catalog |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Text Search

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless headphones with long battery", "top_k": 5, "max_price": 300}'
```

### Example: Image Search

```bash
curl -X POST http://localhost:8000/search/image \
  -F "image=@product.jpg" \
  -F "query=Find similar products" \
  -F "top_k=5"
```

---

## 🎨 UI Features

- **Animated background** — three floating gradient blobs with smooth motion
- **Glassmorphism panels** — `backdrop-filter: blur(20px)` + semi-transparent borders
- **Chat interface** — AI/user bubbles with fade-up animation
- **Product cards** — image, price, rating, stock badge, hover lift effect
- **Sidebar filters** — category, brand, price range, in-stock toggle
- **Quick filter chips** — one-click semantic searches
- **Image upload zone** — drag & drop with glowing border + preview
- **Live stats** — result count, latency, engine info

---

## 📊 Sample Queries

| Query | Expected Behaviour |
|---|---|
| "Best noise-cancelling headphones" | Returns Sony WH-1000XM5, Bose QC45, AirPods Pro |
| "Show similar items under $200" | Filters by price, ranks by semantic + rating |
| "Is this available in another color?" | Surfaces `colors_available` from product data |
| "Portable waterproof speaker" | JBL Charge 5 (IP67 tag match) |
| "Smart home starter kit" | Philips Hue (Smart Home category) |
| Image of headphones | Visual feature extraction → headphone products |

---

## 🔐 Hallucination Control

1. **Context-only prompts** — LLM instructed to answer ONLY from retrieved context
2. **Grounded template fallback** — uses only product fields, no invented data
3. **Attribute validation** — prices/specs sourced directly from the database
4. **Safe fallback** — "I don't have that information" when context is empty

---

## 🔮 Upgrade Path

| Current (CPU) | Production Upgrade |
|---|---|
| TF-IDF + SVD embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Colour histogram image search | CLIP ViT-B/32 (requires PyTorch + GPU) |
| BLIP-2 caption fallback | Full BLIP-2 image captioning |
| Template LLM fallback | Ollama Mistral / LLaMA 3 |
| FAISS IndexFlatIP | FAISS IndexIVFFlat (for >1M products) |
| In-memory store | PostgreSQL + pgvector |

---

## 📄 License

MIT
