"""
Web Intelligence Pipeline — crawl, extract, chunk, embed, store, and retrieve.

This is the core library entry point. It fetches web content and makes it
searchable via semantic search. The retrieved context is formatted so users
can feed it directly to any LLM (Ollama, OpenAI, Anthropic, etc.).

Now with:
  - Pluggable embedders (sentence-transformers, fastembed, OpenAI, Ollama)
  - Pluggable vector stores (ChromaDB, NumPy)
  - Web search integration (DuckDuckGo)
  - Async-safe (works in Jupyter, FastAPI, etc.)
  - Proper logging (no print spam)
  - Retry with backoff, rate limiting, robots.txt
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional

from .async_crawler import crawl_urls_batch
from .cache import ContentCache, URLCache
from .chunker import chunk_text
from .config import Config, default_config
from .context_formatter import (
    RetrievedContext,
    format_context_numbered,
    format_context_plain,
    format_context_structured,
)
from .extractor import extract_content

if TYPE_CHECKING:
    from .embedders.base import BaseEmbedder
    from .search_providers.base import BaseSearchProvider
    from .vector_stores.base import BaseVectorStore

logger = logging.getLogger("web_intelligence.pipeline")


# ---------------------------------------------------------------------- #
# Async-loop helpers: safe in Jupyter, FastAPI, threads, and vanilla scripts
# ---------------------------------------------------------------------- #


def _run_async(coro):
    """
    Run a coroutine from sync code, regardless of whether a loop is already running.

    * No running loop → ``asyncio.run()``.
    * Loop already running (Jupyter/FastAPI) → use ``nest_asyncio`` if available,
      otherwise fall back to a background thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # A loop is already running — try nest_asyncio first
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        pass

    # Final fallback: run the coroutine in a new thread with its own loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class FastPipeline:
    """
    End-to-end pipeline: crawl URLs → extract text → embed → store → retrieve.

    This library does NOT include an LLM. It gives you perfectly formatted
    context that you feed to whatever LLM you want.

    Quick start:
        >>> pipeline = FastPipeline()
        >>> pipeline.index_url("https://en.wikipedia.org/wiki/Python")
        >>> ctx = pipeline.retrieve("what is python used for")
        >>> print(ctx.context_text)        # feed to any LLM
        >>> messages = ctx.as_messages()   # OpenAI-compatible format

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

    def __init__(
        self,
        config: Optional[Config] = None,
        # Pluggable components — pass your own or let the pipeline auto-detect
        embedder: Optional["BaseEmbedder"] = None,
        vector_store: Optional["BaseVectorStore"] = None,
        search_provider: Optional["BaseSearchProvider"] = None,
        # Quick overrides (ignored if you pass explicit components)
        storage_path: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        embedding_model: Optional[str] = None,
        # Progress callback: fn(step: str, current: int, total: int)
        on_progress: Optional[Callable] = None,
    ):
        # Deep copy config to avoid mutating the global/shared instance
        self.config = config.copy() if config else default_config()

        # Apply quick overrides
        if storage_path is not None:
            self.config.vector_store.persist_directory = storage_path
        if cache_enabled is not None:
            self.config.cache_enabled = cache_enabled
        if embedding_model is not None:
            self.config.embedding.model_name = embedding_model

        self.on_progress = on_progress

        # ---- Embedder (pluggable) ----
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = self._auto_detect_embedder(use_gpu)

        # ---- Vector store (pluggable) ----
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = self._auto_detect_vector_store()

        # ---- Search provider (pluggable, optional) ----
        self.search_provider = search_provider  # can be None

        # ---- Caches ----
        self.url_cache = URLCache() if self.config.cache_enabled else None
        self.content_cache = ContentCache() if self.config.cache_enabled else None

        self.stats_data = {
            "urls_processed": 0,
            "urls_cached": 0,
            "duplicates_skipped": 0,
            "chunks_created": 0,
        }

        logger.info(
            "Pipeline ready (embedder=%s, dim=%d, store=%s)",
            self.embedder.model_name,
            self.embedder.dimension,
            type(self.vector_store).__name__,
        )

    # ------------------------------------------------------------------ #
    # Auto-detection helpers
    # ------------------------------------------------------------------ #

    def _auto_detect_embedder(self, use_gpu: Optional[bool] = None):
        """Pick the best available embedding backend."""
        from .embedders import auto_detect_embedder

        kwargs: dict = {
            "model_name": self.config.embedding.model_name,
            "use_cache": self.config.embedding.use_cache,
        }

        device = self.config.embedding.device
        if use_gpu is True:
            device = "cuda"
        elif use_gpu is False:
            device = "cpu"

        if device:
            kwargs["device"] = device

        return auto_detect_embedder(**kwargs)

    def _auto_detect_vector_store(self):
        """Pick the best available vector store backend."""
        from .vector_stores import auto_detect_vector_store

        return auto_detect_vector_store(
            persist_directory=self.config.vector_store.persist_directory,
            collection_name=self.config.vector_store.collection_name,
        )

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index_url(self, url: str, skip_cache: bool = False) -> Dict:
        """Index a single URL. Returns result dict."""
        if self.url_cache and not skip_cache:
            if self.url_cache.is_cached(url):
                cached_data = self.url_cache.get(url)
                logger.debug("Cache hit for %s", url)
                self.stats_data["urls_cached"] += 1
                return {
                    "success": True,
                    "url": url,
                    "cached": True,
                    "title": cached_data.get("data", {}).get("title", "Cached"),
                    "chunks_count": cached_data.get("data", {}).get("chunks", 0),
                    "doc_id": cached_data.get("data", {}).get("doc_id", ""),
                    "indexed_at": cached_data["cached_at"],
                }

        results = _run_async(self._index_urls_async([url]))
        result = results[0]

        if self.url_cache and result["success"] and not result.get("duplicate"):
            self.url_cache.set(url, {
                "doc_id": result["doc_id"],
                "chunks": result["chunks_count"],
                "title": result["title"],
            })

        self.stats_data["urls_processed"] += 1
        return result

    async def index_url_async(self, url: str, skip_cache: bool = False) -> Dict:
        """Async version of ``index_url`` — safe to call from FastAPI / Jupyter."""
        if self.url_cache and not skip_cache:
            if self.url_cache.is_cached(url):
                cached_data = self.url_cache.get(url)
                self.stats_data["urls_cached"] += 1
                return {
                    "success": True,
                    "url": url,
                    "cached": True,
                    "title": cached_data.get("data", {}).get("title", "Cached"),
                    "chunks_count": cached_data.get("data", {}).get("chunks", 0),
                    "doc_id": cached_data.get("data", {}).get("doc_id", ""),
                    "indexed_at": cached_data["cached_at"],
                }

        results = await self._index_urls_async([url])
        result = results[0]

        if self.url_cache and result["success"] and not result.get("duplicate"):
            self.url_cache.set(url, {
                "doc_id": result["doc_id"],
                "chunks": result["chunks_count"],
                "title": result["title"],
            })

        self.stats_data["urls_processed"] += 1
        return result

    async def _index_urls_async(self, urls: List[str]) -> List[Dict]:
        """Crawl → extract → chunk → embed → store for a list of URLs."""
        results: List[Dict] = []

        cfg = self.config.crawler
        crawl_results = await crawl_urls_batch(
            urls,
            max_concurrent=cfg.max_concurrent,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            requests_per_second=cfg.requests_per_second,
            respect_robots=cfg.respect_robots,
        )

        total = len(crawl_results)
        for idx, crawl_result in enumerate(crawl_results, 1):
            if self.on_progress:
                self.on_progress("indexing", idx, total)

            if not crawl_result.success:
                logger.warning("Crawl failed for %s: %s", crawl_result.url, crawl_result.error)
                results.append({
                    "success": False,
                    "url": crawl_result.url,
                    "error": crawl_result.error or "Failed to crawl",
                })
                continue

            extracted = extract_content(crawl_result.html, crawl_result.url)
            if not extracted.text:
                logger.warning("No content extracted from %s", crawl_result.url)
                results.append({
                    "success": False,
                    "url": crawl_result.url,
                    "error": "No content extracted",
                })
                continue

            # Content deduplication
            if self.content_cache:
                if self.content_cache.is_duplicate(extracted.text):
                    existing = self.content_cache.get_existing(extracted.text)
                    self.stats_data["duplicates_skipped"] += 1
                    logger.info("Duplicate content for %s (same as %s)", crawl_result.url, existing["url"])
                    results.append({
                        "success": True,
                        "url": crawl_result.url,
                        "duplicate": True,
                        "original_url": existing["url"],
                        "doc_id": existing.get("doc_id", ""),
                        "title": extracted.title,
                        "chunks_count": 0,
                    })
                    continue

            doc_id = str(uuid.uuid4())
            chunks = chunk_text(
                extracted.text,
                doc_id,
                crawl_result.url,
                chunk_size=self.config.chunker.chunk_size,
                overlap=self.config.chunker.chunk_overlap,
            )

            if not chunks:
                results.append({
                    "success": False,
                    "url": crawl_result.url,
                    "error": "No chunks created",
                })
                continue

            chunk_texts = [c.text for c in chunks]
            vectors = self.embedder.embed_batch(
                chunk_texts, batch_size=self.config.embedding.batch_size
            )

            metadatas = [
                {
                    "chunk_index": c.chunk_index,
                    "title": extracted.title,
                    "url": c.url,
                    "doc_id": doc_id,
                    "word_count": c.word_count,
                    "indexed_at": datetime.now().isoformat(),
                }
                for c in chunks
            ]

            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

            # Store text in documents field (not crammed into metadata)
            self.vector_store.add(
                vectors=vectors,
                metadatas=metadatas,
                ids=ids,
                documents=chunk_texts,
            )

            if self.content_cache:
                self.content_cache.mark_as_indexed(extracted.text, doc_id, crawl_result.url)

            self.stats_data["chunks_created"] += len(chunks)

            logger.info(
                "Indexed %s — %d chunks (%d words)",
                crawl_result.url, len(chunks), extracted.word_count,
            )
            results.append({
                "success": True,
                "url": crawl_result.url,
                "doc_id": doc_id,
                "title": extracted.title,
                "chunks_count": len(chunks),
                "word_count": extracted.word_count,
                "cached": False,
                "duplicate": False,
            })

        return results

    def index_batch(
        self,
        urls: List[str],
        skip_cached: bool = True,
        on_progress: Optional[Callable] = None,
    ) -> List[Dict]:
        """Index multiple URLs concurrently. Skips already-cached URLs by default."""
        import time

        urls_to_process: List[str] = []
        cached_results: List[Dict] = []

        if self.url_cache and skip_cached:
            for url in urls:
                if self.url_cache.is_cached(url):
                    cached_data = self.url_cache.get(url)
                    cached_results.append({
                        "success": True,
                        "url": url,
                        "cached": True,
                        "title": cached_data.get("data", {}).get("title", "Cached"),
                        "chunks_count": cached_data.get("data", {}).get("chunks", 0),
                        "doc_id": cached_data.get("data", {}).get("doc_id", ""),
                        "indexed_at": cached_data["cached_at"],
                    })
                    self.stats_data["urls_cached"] += 1
                else:
                    urls_to_process.append(url)
        else:
            urls_to_process = list(urls)

        if not urls_to_process:
            return cached_results

        logger.info(
            "Processing %d URLs (%d cached, skipping)",
            len(urls_to_process), len(cached_results),
        )

        progress = on_progress or self.on_progress

        # Temporarily set progress callback
        old_progress = self.on_progress
        if progress:
            self.on_progress = progress

        start = time.time()
        new_results = _run_async(self._index_urls_async(urls_to_process))
        elapsed = time.time() - start

        self.on_progress = old_progress

        success_count = sum(1 for r in new_results if r.get("success"))
        duplicate_count = sum(1 for r in new_results if r.get("duplicate"))

        logger.info(
            "Processed %d URLs in %.2fs — success: %d, duplicates: %d, cached: %d",
            len(urls_to_process), elapsed, success_count, duplicate_count, len(cached_results),
        )

        for result in new_results:
            if result.get("success") and not result.get("duplicate") and self.url_cache:
                self.url_cache.set(result["url"], {
                    "doc_id": result.get("doc_id", ""),
                    "chunks": result.get("chunks_count", 0),
                    "title": result.get("title", ""),
                })

        self.stats_data["urls_processed"] += len(urls_to_process)
        return cached_results + new_results

    async def index_batch_async(self, urls: List[str], skip_cached: bool = True) -> List[Dict]:
        """Async version of ``index_batch``."""
        urls_to_process: List[str] = []
        cached_results: List[Dict] = []

        if self.url_cache and skip_cached:
            for url in urls:
                if self.url_cache.is_cached(url):
                    cached_data = self.url_cache.get(url)
                    cached_results.append({
                        "success": True,
                        "url": url,
                        "cached": True,
                        "title": cached_data.get("data", {}).get("title", "Cached"),
                        "chunks_count": cached_data.get("data", {}).get("chunks", 0),
                        "doc_id": cached_data.get("data", {}).get("doc_id", ""),
                        "indexed_at": cached_data["cached_at"],
                    })
                    self.stats_data["urls_cached"] += 1
                else:
                    urls_to_process.append(url)
        else:
            urls_to_process = list(urls)

        if not urls_to_process:
            return cached_results

        new_results = await self._index_urls_async(urls_to_process)

        for result in new_results:
            if result.get("success") and not result.get("duplicate") and self.url_cache:
                self.url_cache.set(result["url"], {
                    "doc_id": result.get("doc_id", ""),
                    "chunks": result.get("chunks_count", 0),
                    "title": result.get("title", ""),
                })

        self.stats_data["urls_processed"] += len(urls_to_process)
        return cached_results + new_results

    # ------------------------------------------------------------------ #
    # Web search (Phase 3)
    # ------------------------------------------------------------------ #

    def search_web(
        self,
        query: str,
        max_results: int = 5,
        limit: int = 5,
        output_format: Literal["plain", "numbered", "structured"] = "numbered",
        max_context_words: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Search the web → crawl top results → index → retrieve context.

        One-liner to go from a question straight to LLM-ready context.

        Args:
            query: Natural language question.
            max_results: Number of web search results to crawl.
            limit: Number of chunks to retrieve.
            output_format: Context format.
            max_context_words: Truncate context to N words.

        Returns:
            RetrievedContext with .context_text, .as_messages(), etc.

        Example:
            >>> ctx = pipeline.search_web("latest python features")
            >>> print(ctx.context_text)
        """
        provider = self._get_search_provider()
        search_results = provider.search(query, max_results=max_results)

        urls = [r.url for r in search_results if r.url]
        if not urls:
            logger.warning("Web search returned no URLs for query: '%s'", query)
            return RetrievedContext(
                query=query, chunks=[], context_text="No results found.",
                sources=[], total_words=0, total_chunks=0,
            )

        logger.info("Web search found %d URLs for '%s'", len(urls), query)
        self.index_batch(urls, skip_cached=True)

        return self.retrieve(query, limit=limit, output_format=output_format,
                             max_context_words=max_context_words)

    async def search_web_async(
        self,
        query: str,
        max_results: int = 5,
        limit: int = 5,
        output_format: Literal["plain", "numbered", "structured"] = "numbered",
        max_context_words: Optional[int] = None,
    ) -> RetrievedContext:
        """Async version of ``search_web``."""
        provider = self._get_search_provider()
        search_results = provider.search(query, max_results=max_results)

        urls = [r.url for r in search_results if r.url]
        if not urls:
            return RetrievedContext(
                query=query, chunks=[], context_text="No results found.",
                sources=[], total_words=0, total_chunks=0,
            )

        await self.index_batch_async(urls, skip_cached=True)

        return self.retrieve(query, limit=limit, output_format=output_format,
                             max_context_words=max_context_words)

    def _get_search_provider(self):
        """Get or auto-detect the web search provider."""
        if self.search_provider is not None:
            return self.search_provider

        # Try to auto-detect
        try:
            from .search_providers import DuckDuckGoSearchProvider
            self.search_provider = DuckDuckGoSearchProvider()
            return self.search_provider
        except ImportError:
            from .exceptions import SearchProviderError
            raise SearchProviderError(
                "none",
                "No search provider available. Install one:\n"
                "  pip install duckduckgo-search\n"
                "  pip install web-intelligence[search]"
            )

    # ------------------------------------------------------------------ #
    # Search & Retrieval
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        limit: int = 5,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Raw semantic search. Returns ranked chunk dicts.

        Args:
            query: Natural language query.
            limit: Max results.
            where_filter: Metadata filter (e.g. ``{"url": "https://..."}``).
            min_score: Minimum similarity (0–1).

        Returns:
            List of dicts with text, score, metadata.
        """
        query_vector = self.embedder.embed(query)
        return self.vector_store.search(
            query_vector, limit=limit, where_filter=where_filter, min_score=min_score
        )

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        output_format: Literal["plain", "numbered", "structured"] = "numbered",
        max_context_words: Optional[int] = None,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> RetrievedContext:
        """
        Search and return LLM-ready context.

        This is the main retrieval method. Returns a ``RetrievedContext`` you
        can feed directly to any LLM.

        Args:
            query: Natural language question.
            limit: Number of chunks to retrieve.
            output_format: ``"plain"``, ``"numbered"``, or ``"structured"``.
            max_context_words: Truncate context to N words.
            where_filter: Metadata filter dict.
            min_score: Minimum relevance score (0–1).

        Returns:
            RetrievedContext with ``.context_text``, ``.sources``,
            ``.as_messages()``, ``.to_dict()``

        Example:
            >>> ctx = pipeline.retrieve("what is python")
            >>> print(ctx.context_text)
            >>> messages = ctx.as_messages()  # OpenAI-compatible
        """
        limit = limit or self.config.search.default_limit
        max_words = max_context_words or self.config.search.max_context_words

        chunks = self.search(query, limit=limit, where_filter=where_filter, min_score=min_score)

        formatters = {
            "plain": format_context_plain,
            "numbered": format_context_numbered,
            "structured": format_context_structured,
        }
        formatter = formatters.get(output_format, format_context_numbered)
        return formatter(chunks, query, max_words=max_words)

    async def retrieve_async(
        self,
        query: str,
        limit: Optional[int] = None,
        output_format: Literal["plain", "numbered", "structured"] = "numbered",
        max_context_words: Optional[int] = None,
        where_filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> RetrievedContext:
        """Async version of ``retrieve`` (useful in FastAPI handlers)."""
        return self.retrieve(
            query, limit=limit, output_format=output_format,
            max_context_words=max_context_words,
            where_filter=where_filter, min_score=min_score,
        )

    def get_context_for_llm(
        self,
        query: str,
        limit: int = 5,
        output_format: Literal["plain", "numbered", "structured"] = "plain",
    ) -> str:
        """
        Shorthand: returns just the context string for direct prompt injection.

        Example:
            >>> context = pipeline.get_context_for_llm("how does auth work")
            >>> answer = my_llm(f"Context: {context}\\nQuestion: how does auth work")
        """
        ctx = self.retrieve(query, limit=limit, output_format=output_format)
        return ctx.context_text

    # ------------------------------------------------------------------ #
    # Document management
    # ------------------------------------------------------------------ #

    def list_documents(self) -> List[Dict]:
        """List all indexed documents with metadata."""
        return self.vector_store.list_documents()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get full text and chunks for a document."""
        return self.vector_store.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from the store."""
        return self.vector_store.delete_document(doc_id)

    def delete_url(self, url: str) -> int:
        """Delete all indexed content from a URL. Returns chunks deleted."""
        count = self.vector_store.delete_by_url(url)
        if self.url_cache:
            self.url_cache.delete(url)
        return count

    # ------------------------------------------------------------------ #
    # Stats & maintenance
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        """Return pipeline statistics."""
        stats = {
            "total_chunks_in_database": self.vector_store.count(),
            "total_documents": len(self.vector_store.list_documents()),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "device": getattr(self.embedder, "device", "n/a"),
            "vector_store": type(self.vector_store).__name__,
            "chunk_size": self.config.chunker.chunk_size,
            "chunk_overlap": self.config.chunker.chunk_overlap,
            "session_stats": self.stats_data,
        }

        if hasattr(self.embedder, "get_cache_stats"):
            stats["embedding_cache"] = self.embedder.get_cache_stats()
        if self.url_cache:
            stats["url_cache"] = self.url_cache.stats()

        return stats

    def clear_all(self):
        """Clear all indexed data, caches, and embeddings."""
        self.vector_store.clear()
        if self.url_cache:
            self.url_cache.clear()
        if self.content_cache:
            self.content_cache.clear()
        if hasattr(self.embedder, "clear_cache"):
            self.embedder.clear_cache()
        self.stats_data = {
            "urls_processed": 0,
            "urls_cached": 0,
            "duplicates_skipped": 0,
            "chunks_created": 0,
        }
        logger.info("All data cleared")

    def clear_caches(self):
        """Clear only caches (URL, content, embedding) — keeps indexed data."""
        if self.url_cache:
            self.url_cache.clear()
        if self.content_cache:
            self.content_cache.clear()
        if hasattr(self.embedder, "clear_cache"):
            self.embedder.clear_cache()
