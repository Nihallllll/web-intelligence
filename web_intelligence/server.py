"""
REST API server for Web Intelligence.

Exposes the pipeline over HTTP so any LLM, app, or script can:
  - Index URLs (POST /index)
  - Search content (POST /search)
  - Retrieve LLM-ready context (POST /retrieve)
  - Manage documents (GET/DELETE /documents)

Start with:
    python -m web_intelligence.server
    # or
    web-intelligence serve
"""

from typing import List, Optional, Dict, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .optimized_pipeline import FastPipeline
from .config import Config, default_config


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IndexURLRequest(BaseModel):
    url: str = Field(..., description="URL to crawl and index")
    skip_cache: bool = Field(False, description="Force re-crawl even if cached")

class IndexBatchRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to index")
    skip_cached: bool = Field(True, description="Skip already-cached URLs")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(5, ge=1, le=50, description="Max results")
    filter: Optional[Dict] = Field(None, description="Metadata filter (e.g. {\"url\": \"...\"})")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    limit: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    format: Literal["plain", "numbered", "structured"] = Field(
        "numbered", description="Context format: plain, numbered, or structured"
    )
    max_context_words: int = Field(3000, ge=100, description="Max words in context")
    filter: Optional[Dict] = Field(None, description="Metadata filter")
    min_score: float = Field(0.0, ge=0.0, le=1.0)

class DeleteURLRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_pipeline: Optional[FastPipeline] = None


def get_pipeline() -> FastPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FastPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    get_pipeline()
    yield


app = FastAPI(
    title="Web Intelligence API",
    description=(
        "Crawl the web, index content, and retrieve LLM-ready context. "
        "No API keys, no cloud — runs 100% locally. "
        "Plug the /retrieve endpoint into any LLM (Ollama, OpenAI, Anthropic, etc.)."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/index")
def index_url(req: IndexURLRequest):
    """Crawl and index a single URL."""
    pipeline = get_pipeline()
    result = pipeline.index_url(req.url, skip_cache=req.skip_cache)
    return result


@app.post("/index/batch")
def index_batch(req: IndexBatchRequest):
    """Crawl and index multiple URLs concurrently."""
    pipeline = get_pipeline()
    results = pipeline.index_batch(req.urls, skip_cached=req.skip_cached)
    return {
        "total": len(results),
        "success": sum(1 for r in results if r.get("success")),
        "results": results,
    }


@app.post("/search")
def search(req: SearchRequest):
    """Raw semantic search. Returns ranked chunks with scores."""
    pipeline = get_pipeline()
    results = pipeline.search(
        req.query,
        limit=req.limit,
        filter=req.filter,
        min_score=req.min_score,
    )
    return {"query": req.query, "results": results}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """
    Retrieve LLM-ready context for a question.

    Returns formatted context text that you can inject directly into
    any LLM prompt (Ollama, OpenAI, Anthropic, local Llama, etc.).

    Also returns OpenAI-compatible messages via the `messages` field.
    """
    pipeline = get_pipeline()
    ctx = pipeline.retrieve(
        req.query,
        limit=req.limit,
        format=req.format,
        max_context_words=req.max_context_words,
        filter=req.filter,
        min_score=req.min_score,
    )
    result = ctx.to_dict()
    result["messages"] = ctx.as_messages()
    return result


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    return {"total": len(docs), "documents": docs}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """Get a specific document with all its chunks."""
    pipeline = get_pipeline()
    doc = pipeline.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "doc_id": doc_id}


@app.delete("/documents/by-url")
def delete_by_url(req: DeleteURLRequest):
    """Delete all indexed content from a URL."""
    pipeline = get_pipeline()
    count = pipeline.delete_url(req.url)
    return {"deleted_chunks": count, "url": req.url}


@app.get("/stats")
def get_stats():
    """Pipeline statistics."""
    pipeline = get_pipeline()
    return pipeline.stats()


@app.post("/clear")
def clear_all():
    """Clear ALL indexed data and caches. Use with caution."""
    pipeline = get_pipeline()
    pipeline.clear_all()
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_server(host: str = None, port: int = None, reload: bool = None):
    """Start the API server."""
    import uvicorn
    config = default_config.server
    uvicorn.run(
        "web_intelligence.server:app",
        host=host or config.host,
        port=port or config.port,
        reload=reload if reload is not None else config.reload,
    )


if __name__ == "__main__":
    start_server()
