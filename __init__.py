"""
Web Intelligence - Fast web crawling, extraction, and semantic search library

A production-ready library for crawling, extracting, and semantically searching web content
with GPU acceleration, async crawling, and caching support.

Example:
    >>> from web_intelligence import FastPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = FastPipeline(use_gpu=True, cache_enabled=True)
    >>> 
    >>> # Index URLs
    >>> pipeline.index_url("https://example.com")
    >>> 
    >>> # Batch indexing
    >>> urls = ["https://site1.com", "https://site2.com"]
    >>> results = pipeline.index_batch(urls)
    >>> 
    >>> # Search
    >>> results = pipeline.search("your query", limit=5)
"""

__version__ = "0.2.0"
__author__ = "Your Name"

# Import main classes for easy access
try:
    from web_intelligence.optimized_pipeline import FastPipeline
    from web_intelligence.fast_embedder import FastEmbedder
    from web_intelligence.async_crawler import crawl_urls_batch, crawl_url_async
    from web_intelligence.cache import URLCache, EmbeddingCache
    from web_intelligence.vector_store import VectorStore
    
    __all__ = [
        'FastPipeline',
        'FastEmbedder',
        'crawl_urls_batch',
        'crawl_url_async',
        'URLCache',
        'EmbeddingCache',
        'VectorStore'
    ]
except ImportError:
    # Fallback for direct script execution
    pass
