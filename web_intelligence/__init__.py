"""
Web Intelligence — free, local web crawling + semantic search.

Fetch web content and serve it to any LLM. No API keys, no cloud.

Quick start:
    >>> from web_intelligence import FastPipeline
    >>> pipeline = FastPipeline()
    >>> pipeline.index_url("https://en.wikipedia.org/wiki/Python")
    >>> ctx = pipeline.retrieve("what is python used for")
    >>> print(ctx.context_text)   # feed this to any LLM
"""

__version__ = "0.3.0"

from .optimized_pipeline import FastPipeline
from .fast_embedder import FastEmbedder
from .async_crawler import crawl_urls_batch, crawl_url_async
from .cache import URLCache, ContentCache, EmbeddingCache
from .vector_store import VectorStore
from .config import Config, default_config
from .context_formatter import (
    RetrievedContext,
    format_context_plain,
    format_context_numbered,
    format_context_structured,
)

__all__ = [
    # Core
    "FastPipeline",
    "Config",
    "RetrievedContext",
    # Components
    "FastEmbedder",
    "VectorStore",
    "URLCache",
    "ContentCache",
    "EmbeddingCache",
    # Crawling
    "crawl_urls_batch",
    "crawl_url_async",
    # Formatters
    "format_context_plain",
    "format_context_numbered",
    "format_context_structured",
    # Config
    "default_config",
]
