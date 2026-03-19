from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Config, default_config
from .optimized_pipeline import FastPipeline

logger = logging.getLogger("web_intelligence.server")


class IndexURLRequest(BaseModel):
    url: str = Field(..., description="URL to crawl and index")
    skip_cache: bool = Field(False, description="Force re-crawl even if cached")


class IndexBatchRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to index")
    skip_cached: bool = Field(True, description="Skip already-cached URLs")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(5, ge=1, le=50, description="Max results")
    where_filter: Optional[Dict] = Field(None, description="Metadata filter")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    limit: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    output_format: Literal["plain", "numbered", "structured"] = Field(
        "numbered", description="Context format"
    )
    max_context_words: int = Field(3000, ge=100, description="Max words in context")
    where_filter: Optional[Dict] = Field(None, description="Metadata filter")
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class SearchWebRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    max_results: int = Field(5, ge=1, le=20, description="Number of web results to crawl")
    limit: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    output_format: Literal["plain", "numbered", "structured"] = Field("numbered")
    max_context_words: int = Field(3000, ge=100, description="Max words in context")


class DeleteURLRequest(BaseModel):
    url: str


_pipeline: Optional[FastPipeline] = None


def get_pipeline() -> FastPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FastPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield


app = FastAPI(
    title="Web Intelligence API",
    description=(
        "Crawl the web, index content, and retrieve LLM-ready context. "
        "No API keys, no cloud — runs 100% locally. "
        "Plug the /retrieve endpoint into any LLM (Ollama, OpenAI, Anthropic, etc.)."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
async def index_url(req: IndexURLRequest):
    pipeline = get_pipeline()
    result = await pipeline.index_url_async(req.url, skip_cache=req.skip_cache)
    return result


@app.post("/index/batch")
async def index_batch(req: IndexBatchRequest):
    pipeline = get_pipeline()
    results = await pipeline.index_batch_async(req.urls, skip_cached=req.skip_cached)
    return {
        "total": len(results),
        "success": sum(1 for r in results if r.get("success")),
        "results": results,
    }


@app.post("/search")
def search(req: SearchRequest):
    pipeline = get_pipeline()
    results = pipeline.search(
        req.query,
        limit=req.limit,
        where_filter=req.where_filter,
        min_score=req.min_score,
    )
    return {"query": req.query, "results": results}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    pipeline = get_pipeline()
    ctx = pipeline.retrieve(
        req.query,
        limit=req.limit,
        output_format=req.output_format,
        max_context_words=req.max_context_words,
        where_filter=req.where_filter,
        min_score=req.min_score,
    )
    result = ctx.to_dict()
    result["messages"] = ctx.as_messages()
    return result


@app.post("/search-web")
def search_web(req: SearchWebRequest):
    pipeline = get_pipeline()
    try:
        ctx = pipeline.search_web(
            req.query,
            max_results=req.max_results,
            limit=req.limit,
            output_format=req.output_format,
            max_context_words=req.max_context_words,
        )
        result = ctx.to_dict()
        result["messages"] = ctx.as_messages()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    return {"total": len(docs), "documents": docs}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    pipeline = get_pipeline()
    doc = pipeline.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "doc_id": doc_id}


@app.delete("/documents/by-url")
def delete_by_url(req: DeleteURLRequest):
    pipeline = get_pipeline()
    count = pipeline.delete_url(req.url)
    return {"deleted_chunks": count, "url": req.url}


@app.get("/stats")
def get_stats():
    pipeline = get_pipeline()
    return pipeline.stats()


@app.post("/clear")
def clear_all():
    pipeline = get_pipeline()
    pipeline.clear_all()
    return {"cleared": True}


def start_server(host: str = None, port: int = None, reload: bool = None):
    import uvicorn
    config = default_config().server
    uvicorn.run(
        "web_intelligence.server:app",
        host=host or config.host,
        port=port or config.port,
        reload=reload if reload is not None else config.reload,
    )


if __name__ == "__main__":
    start_server()
