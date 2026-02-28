"""
Web Intelligence Pipeline — crawl, extract, chunk, embed, store, and retrieve.

This is the core library entry point. It fetches web content and makes it
searchable via semantic search. The retrieved context is formatted so users
can feed it directly to any LLM of their choice (Ollama, OpenAI, Anthropic, etc.).
"""

from typing import List, Dict, Optional, Literal
import asyncio
from .async_crawler import crawl_urls_batch
from .fast_embedder import FastEmbedder
from .cache import URLCache, ContentCache
from .extractor import extract_content
from .chunker import chunk_text
from .vector_store import VectorStore
from .config import Config, default_config
from .context_formatter import (
    RetrievedContext,
    format_context_plain,
    format_context_numbered,
    format_context_structured,
)
import uuid
from datetime import datetime


class FastPipeline:
    """
    End-to-end pipeline: crawl URLs → extract text → embed → store → retrieve.

    This library does NOT include an LLM. It gives you perfectly formatted
    context that you feed to whatever LLM you want.

    Quick start:
        >>> pipeline = FastPipeline()
        >>> pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
        >>> context = pipeline.retrieve("what is python used for")
        >>> # feed context.context_text to your LLM
        >>> # or use context.as_messages() for OpenAI-compatible chat format
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        storage_path: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Full Config object. If None, uses defaults + env vars.
            storage_path: Override vector store path.
            cache_enabled: Override cache setting.
            use_gpu: Force GPU (True), CPU (False), or auto-detect (None).
            embedding_model: Override embedding model name.
        """
        self.config = config or default_config

        # Allow individual overrides
        if storage_path is not None:
            self.config.vector_store.persist_directory = storage_path
        if cache_enabled is not None:
            self.config.cache_enabled = cache_enabled
        if embedding_model is not None:
            self.config.embedding.model_name = embedding_model

        # Resolve device
        device = self.config.embedding.device
        if use_gpu is True:
            device = "cuda"
        elif use_gpu is False:
            device = "cpu"

        print("Initializing pipeline...")

        self.embedder = FastEmbedder(
            model_name=self.config.embedding.model_name,
            device=device,
            use_cache=self.config.embedding.use_cache,
        )

        self.vector_store = VectorStore(
            persist_directory=self.config.vector_store.persist_directory,
            collection_name=self.config.vector_store.collection_name,
        )

        self.url_cache = URLCache() if self.config.cache_enabled else None
        self.content_cache = ContentCache() if self.config.cache_enabled else None

        self.stats_data = {
            "urls_processed": 0,
            "urls_cached": 0,
            "duplicates_skipped": 0,
            "chunks_created": 0,
        }

        print("Pipeline ready")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_url(self, url: str, skip_cache: bool = False) -> Dict:
        """Index a single URL. Returns result dict with success, title, chunks_count."""
        if self.url_cache and not skip_cache:
            if self.url_cache.is_cached(url):
                cached_data = self.url_cache.get(url)
                print(f"  Using cached data for {url}")
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

        results = asyncio.run(self._index_urls_async([url]))
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
        """Crawl, extract, chunk, embed, and store content for a list of URLs."""
        results = []
        timeout = self.config.crawler.timeout
        max_concurrent = self.config.crawler.max_concurrent

        crawl_results = await crawl_urls_batch(urls, max_concurrent=max_concurrent, timeout=timeout)

        for crawl_result in crawl_results:
            if not crawl_result.success:
                results.append({
                    "success": False,
                    "url": crawl_result.url,
                    "error": crawl_result.error or "Failed to crawl",
                })
                continue

            extracted = extract_content(crawl_result.html, crawl_result.url)
            if not extracted.text:
                results.append({
                    "success": False,
                    "url": crawl_result.url,
                    "error": "No content extracted",
                })
                continue

            if self.content_cache:
                if self.content_cache.is_duplicate(extracted.text):
                    existing = self.content_cache.get_existing(extracted.text)
                    self.stats_data["duplicates_skipped"] += 1
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

            if len(chunks) == 0:
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
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "title": extracted.title,
                    "url": c.url,
                    "doc_id": doc_id,
                    "word_count": c.token_count,
                    "indexed_at": datetime.now().isoformat(),
                }
                for c in chunks
            ]

            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            self.vector_store.add(vectors, metadatas, ids)

            if self.content_cache:
                self.content_cache.mark_as_indexed(extracted.text, doc_id, crawl_result.url)

            self.stats_data["chunks_created"] += len(chunks)

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

    def index_batch(self, urls: List[str], skip_cached: bool = True) -> List[Dict]:
        """Index multiple URLs concurrently. Skips already-cached URLs by default."""
        import time

        urls_to_process = []
        cached_results = []

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

        if len(urls_to_process) == 0:
            return cached_results

        print(f"Processing {len(urls_to_process)} URLs...")
        if cached_results:
            print(f"  {len(cached_results)} URLs already cached, skipping")

        start = time.time()
        new_results = asyncio.run(self._index_urls_async(urls_to_process))
        elapsed = time.time() - start

        success_count = sum(1 for r in new_results if r.get("success"))
        duplicate_count = sum(1 for r in new_results if r.get("duplicate"))

        print(f"  Processed {len(urls_to_process)} URLs in {elapsed:.2f}s")
        print(f"  Success: {success_count} | Duplicates: {duplicate_count} | Cached: {len(cached_results)}")

        for result in new_results:
            if result.get("success") and not result.get("duplicate") and self.url_cache:
                self.url_cache.set(result["url"], {
                    "doc_id": result.get("doc_id", ""),
                    "chunks": result.get("chunks_count", 0),
                    "title": result.get("title", ""),
                })

        self.stats_data["urls_processed"] += len(urls_to_process)
        return cached_results + new_results

    # ------------------------------------------------------------------
    # Search & Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5,
               filter: Optional[Dict] = None,
               min_score: float = 0.0) -> List[Dict]:
        """
        Raw semantic search. Returns ranked chunk dicts.

        For LLM-ready output, use retrieve() instead.

        Args:
            query: Natural language query.
            limit: Max results.
            filter: ChromaDB where-filter (e.g. {"url": "https://..."}).
            min_score: Minimum similarity (0-1).

        Returns:
            List of dicts with text, score, metadata.
        """
        query_vector = self.embedder.embed(query)
        return self.vector_store.search(
            query_vector, limit=limit, filter=filter, min_score=min_score
        )

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        format: Literal["plain", "numbered", "structured"] = "numbered",
        max_context_words: Optional[int] = None,
        filter: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> RetrievedContext:
        """
        Search and return LLM-ready context.

        This is the main method users should call. It searches the indexed
        content and formats the results so they can be directly fed to any LLM.

        Args:
            query: Natural language question.
            limit: Number of chunks to retrieve.
            format: Output format — "plain", "numbered", or "structured".
            max_context_words: Truncate context to this many words.
            filter: ChromaDB where-filter dict.
            min_score: Minimum relevance score (0-1).

        Returns:
            RetrievedContext with .context_text, .sources, .as_messages(), .to_dict()

        Example:
            >>> ctx = pipeline.retrieve("what is python")
            >>> print(ctx.context_text)    # paste into any LLM prompt
            >>> messages = ctx.as_messages()  # OpenAI-compatible format
            >>> # openai.chat.completions.create(model="gpt-4o", messages=messages)
        """
        limit = limit or self.config.search.default_limit
        max_words = max_context_words or self.config.search.max_context_words

        chunks = self.search(query, limit=limit, filter=filter, min_score=min_score)

        formatters = {
            "plain": format_context_plain,
            "numbered": format_context_numbered,
            "structured": format_context_structured,
        }
        formatter = formatters.get(format, format_context_numbered)
        return formatter(chunks, query, max_words=max_words)

    def get_context_for_llm(
        self,
        query: str,
        limit: int = 5,
        format: Literal["plain", "numbered", "structured"] = "plain",
    ) -> str:
        """
        Shorthand: returns just the context string for direct prompt injection.

        Example:
            >>> context = pipeline.get_context_for_llm("how does auth work")
            >>> answer = my_llm(f"Context: {context}\\nQuestion: how does auth work")
        """
        ctx = self.retrieve(query, limit=limit, format=format)
        return ctx.context_text

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stats & maintenance
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return pipeline statistics."""
        stats = {
            "total_chunks_in_database": self.vector_store.count(),
            "total_documents": len(self.vector_store.list_documents()),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "device": self.embedder.device,
            "chunk_size": self.config.chunker.chunk_size,
            "chunk_overlap": self.config.chunker.chunk_overlap,
            "session_stats": self.stats_data,
            "embedding_cache": self.embedder.get_cache_stats(),
        }

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
        if self.embedder:
            self.embedder.clear_cache()
        self.stats_data = {
            "urls_processed": 0,
            "urls_cached": 0,
            "duplicates_skipped": 0,
            "chunks_created": 0,
        }
        print("All data cleared.")

    def clear_caches(self):
        """Clear only caches (URL, content, embedding) — keeps indexed data."""
        if self.url_cache:
            self.url_cache.clear()
        if self.content_cache:
            self.content_cache.clear()
        if self.embedder:
            self.embedder.clear_cache()
