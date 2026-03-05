"""
Web Intelligence — free, local web crawling + semantic search.

Fetch web content and serve it to any LLM. No API keys required, no cloud.

Quick start:
    >>> from web_intelligence import FastPipeline
    >>> pipeline = FastPipeline()
    >>> pipeline.index_url("https://en.wikipedia.org/wiki/Python")
    >>> ctx = pipeline.retrieve("what is python used for")
    >>> print(ctx.context_text)   # feed this to any LLM

Web search (searches the internet for you):
    >>> ctx = pipeline.search_web("latest python features")
    >>> print(ctx.context_text)

Pluggable components:
    >>> from web_intelligence.embedders import OpenAIEmbedder
    >>> from web_intelligence.vector_stores import NumpyVectorStore
    >>> pipeline = FastPipeline(
    ...     embedder=OpenAIEmbedder(api_key="sk-..."),
    ...     vector_store=NumpyVectorStore(),
    ... )
"""

__version__ = "0.4.0"

from .optimized_pipeline import FastPipeline
from .config import Config, default_config
from .context_formatter import (
    RetrievedContext,
    format_context_plain,
    format_context_numbered,
    format_context_structured,
)
from .exceptions import (
    WebIntelligenceError,
    CrawlError,
    ExtractionError,
    EmbeddingError,
    NoEmbedderError,
    VectorStoreError,
    SearchProviderError,
    ConfigError,
)
from ._logging import setup_logging

# Lazy-imported subpackages — always available via web_intelligence.embedders etc.
# but actual backends are only loaded on first use to avoid heavy imports.

__all__ = [
    # Core
    "FastPipeline",
    "Config",
    "default_config",
    "RetrievedContext",
    # Formatters
    "format_context_plain",
    "format_context_numbered",
    "format_context_structured",
    # Exceptions
    "WebIntelligenceError",
    "CrawlError",
    "ExtractionError",
    "EmbeddingError",
    "NoEmbedderError",
    "VectorStoreError",
    "SearchProviderError",
    "ConfigError",
    # Utility
    "setup_logging",
]
