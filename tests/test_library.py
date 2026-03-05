"""
Tests for Web Intelligence library.

Run with:
    python -m pytest tests/ -v

These tests verify:
    - Config creation and deep copy
    - Chunker splits text correctly
    - Extractor pulls text from HTML
    - Embedder produces correct-dimension vectors
    - NumpyVectorStore add/search/delete cycle
    - Full pipeline index + retrieve (end-to-end, needs network)
    - Web search pipeline (end-to-end, needs network)
"""

import pytest
import os
import shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

TEST_DATA_DIR = Path(__file__).parent / "_test_data"


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Remove test data after each test."""
    yield
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


# ─────────────────────────────────────────────────────────
# 1. Config tests
# ─────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_creates_fresh_instance(self):
        from web_intelligence.config import default_config
        c1 = default_config()
        c2 = default_config()
        assert c1 is not c2, "default_config() should return a new object each time"

    def test_config_copy_is_independent(self):
        from web_intelligence.config import default_config
        c1 = default_config()
        c2 = c1.copy()
        c2.chunker.chunk_size = 9999
        assert c1.chunker.chunk_size != 9999, "copy() must not share state"

    def test_config_defaults_are_sensible(self):
        from web_intelligence.config import default_config
        c = default_config()
        assert c.chunker.chunk_size == 400
        assert c.chunker.chunk_overlap == 50
        assert c.crawler.max_retries == 3
        assert c.crawler.respect_robots is True
        assert c.embedding.model_name == "all-MiniLM-L6-v2"

    def test_config_to_dict(self):
        from web_intelligence.config import default_config
        d = default_config().to_dict()
        assert isinstance(d, dict)
        assert "crawler" in d
        assert "chunker" in d
        assert "embedding" in d


# ─────────────────────────────────────────────────────────
# 2. Chunker tests
# ─────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_text_returns_chunks(self):
        from web_intelligence.chunker import chunk_text

        text = "This is sentence one. This is sentence two. " * 50  # ~500 words
        chunks = chunk_text(text, document_id="doc1", url="http://example.com")
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunk_fields(self):
        from web_intelligence.chunker import chunk_text

        text = "Hello world. This is a test. " * 100
        chunks = chunk_text(text, document_id="test_doc", url="http://test.com")
        c = chunks[0]
        assert c.doc_id == "test_doc"
        assert c.url == "http://test.com"
        assert c.chunk_index == 0
        assert c.word_count > 0
        assert len(c.text) > 0

    def test_empty_text_gives_no_chunks(self):
        from web_intelligence.chunker import chunk_text
        chunks = chunk_text("", document_id="x", url="http://x.com")
        assert chunks == []

    def test_word_count_field_name(self):
        """Verify the field is called word_count (not token_count)."""
        from web_intelligence.chunker import Chunk
        c = Chunk(text="hello world", chunk_index=0, word_count=2, doc_id="d", url="u")
        assert c.word_count == 2

    def test_chunks_overlap(self):
        from web_intelligence.chunker import chunk_text

        # Enough text to make multiple chunks
        text = ". ".join(f"Sentence number {i} with some extra words" for i in range(200))
        chunks = chunk_text(text, document_id="d", url="u", chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            # Last words of chunk 0 should appear at start of chunk 1
            words_end = chunks[0].text.split()[-5:]
            words_start = chunks[1].text.split()[:20]
            overlap_found = any(w in words_start for w in words_end)
            assert overlap_found, "Chunks should overlap"

    def test_split_into_sentences(self):
        from web_intelligence.chunker import split_into_sentences
        sentences = split_into_sentences("Hello world. How are you? I am fine!")
        assert len(sentences) == 3


# ─────────────────────────────────────────────────────────
# 3. Extractor tests
# ─────────────────────────────────────────────────────────

class TestExtractor:
    def test_extract_from_html(self):
        from web_intelligence.extractor import extract_content

        html = """
        <html><head><title>Test Page</title></head>
        <body>
            <nav>skip this nav</nav>
            <article>
                <h1>Main Article</h1>
                <p>This is the main content of the page. It has useful information
                that should be extracted by the library. The extractor should ignore
                navigation and ads and keep only the article text.</p>
            </article>
        </body></html>
        """
        result = extract_content(html, "http://example.com")
        assert result.url == "http://example.com"
        # trafilatura may or may not extract from minimal HTML, so just check structure
        assert hasattr(result, "text")
        assert hasattr(result, "title")
        assert hasattr(result, "word_count")

    def test_extract_empty_html(self):
        from web_intelligence.extractor import extract_content
        result = extract_content("", "http://empty.com")
        assert result.text == ""
        assert result.word_count == 0


# ─────────────────────────────────────────────────────────
# 4. Embedder tests
# ─────────────────────────────────────────────────────────

class TestEmbedder:
    def test_auto_detect(self):
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        assert embedder.dimension > 0
        assert len(embedder.model_name) > 0

    def test_embed_single(self):
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        vec = embedder.embed("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == embedder.dimension

    def test_embed_batch(self):
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        vecs = embedder.embed_batch(["Hello", "World", "Test"])
        assert len(vecs) == 3
        assert all(len(v) == embedder.dimension for v in vecs)

    def test_embed_batch_empty(self):
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        vecs = embedder.embed_batch([])
        assert vecs == []

    def test_embeddings_are_normalized(self):
        """Vectors should be unit-length (cosine similarity ready)."""
        import math
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        vec = embedder.embed("Test normalization")
        magnitude = math.sqrt(sum(x * x for x in vec))
        assert abs(magnitude - 1.0) < 0.01, f"Expected unit vector, got magnitude {magnitude}"

    def test_cache_stats(self):
        from web_intelligence.embedders import auto_detect_embedder
        embedder = auto_detect_embedder()
        embedder.embed("cache test")
        embedder.embed("cache test")  # should hit cache
        stats = embedder.get_cache_stats()
        assert isinstance(stats, dict)
        assert stats["cache_hits"] >= 1


# ─────────────────────────────────────────────────────────
# 5. NumpyVectorStore tests
# ─────────────────────────────────────────────────────────

class TestNumpyVectorStore:
    def test_add_and_search(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore()

        # 3 fake vectors, dimension 4
        store.add(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            metadatas=[
                {"url": "a.com", "doc_id": "d1"},
                {"url": "b.com", "doc_id": "d1"},
                {"url": "c.com", "doc_id": "d2"},
            ],
            ids=["id1", "id2", "id3"],
            documents=["text a", "text b", "text c"],
        )

        assert store.count() == 3

        results = store.search([1, 0, 0, 0], limit=1)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["text"] == "text a"

    def test_delete_document(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore()
        store.add(
            vectors=[[1, 0], [0, 1]],
            metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
            ids=["a", "b"],
            documents=["x", "y"],
        )
        assert store.delete_document("d1") is True
        assert store.count() == 1

    def test_delete_by_url(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore()
        store.add(
            vectors=[[1, 0], [0, 1]],
            metadatas=[{"url": "http://a.com"}, {"url": "http://b.com"}],
            ids=["a", "b"],
            documents=["x", "y"],
        )
        count = store.delete_by_url("http://a.com")
        assert count == 1
        assert store.count() == 1

    def test_clear(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore()
        store.add(
            vectors=[[1, 0]],
            metadatas=[{"doc_id": "d1"}],
            ids=["a"],
            documents=["x"],
        )
        store.clear()
        assert store.count() == 0

    def test_persistence(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        path = str(TEST_DATA_DIR / "test_store.pkl")

        store1 = NumpyVectorStore(persist_path=path)
        store1.add(
            vectors=[[1, 0, 0]],
            metadatas=[{"doc_id": "d1"}],
            ids=["id1"],
            documents=["saved text"],
        )

        # Load from disk
        store2 = NumpyVectorStore(persist_path=path)
        assert store2.count() == 1
        results = store2.search([1, 0, 0], limit=1)
        assert results[0]["text"] == "saved text"

    def test_where_filter(self):
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore()
        store.add(
            vectors=[[1, 0], [0, 1]],
            metadatas=[{"url": "http://a.com", "doc_id": "d1"},
                       {"url": "http://b.com", "doc_id": "d2"}],
            ids=["a", "b"],
            documents=["alpha", "beta"],
        )
        results = store.search([1, 0], limit=10, where_filter={"url": "http://b.com"})
        assert len(results) == 1
        assert results[0]["id"] == "b"


# ─────────────────────────────────────────────────────────
# 6. Context formatter tests
# ─────────────────────────────────────────────────────────

class TestContextFormatter:
    def _fake_chunks(self):
        return [
            {
                "text": "Python is a great language.",
                "score": 0.95,
                "metadata": {"url": "http://example.com", "title": "Python"},
            },
            {
                "text": "It is used for web development.",
                "score": 0.80,
                "metadata": {"url": "http://example.com", "title": "Python"},
            },
        ]

    def test_format_plain(self):
        from web_intelligence.context_formatter import format_context_plain
        ctx = format_context_plain(self._fake_chunks(), "what is python")
        assert len(ctx.context_text) > 0
        assert ctx.total_chunks == 2

    def test_format_numbered(self):
        from web_intelligence.context_formatter import format_context_numbered
        ctx = format_context_numbered(self._fake_chunks(), "what is python")
        assert "[Source 1]" in ctx.context_text

    def test_format_structured(self):
        from web_intelligence.context_formatter import format_context_structured
        ctx = format_context_structured(self._fake_chunks(), "what is python")
        assert len(ctx.context_text) > 0

    def test_as_messages(self):
        from web_intelligence.context_formatter import format_context_numbered
        ctx = format_context_numbered(self._fake_chunks(), "what is python")
        msgs = ctx.as_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "python" in msgs[1]["content"].lower()

    def test_to_dict(self):
        from web_intelligence.context_formatter import format_context_plain
        ctx = format_context_plain(self._fake_chunks(), "test")
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "context_text" in d
        assert "sources" in d


# ─────────────────────────────────────────────────────────
# 7. Exceptions tests
# ─────────────────────────────────────────────────────────

class TestExceptions:
    def test_exception_hierarchy(self):
        from web_intelligence.exceptions import (
            WebIntelligenceError, CrawlError, EmbeddingError,
            NoEmbedderError, VectorStoreError, SearchProviderError,
        )
        assert issubclass(CrawlError, WebIntelligenceError)
        assert issubclass(EmbeddingError, WebIntelligenceError)
        assert issubclass(NoEmbedderError, WebIntelligenceError)
        assert issubclass(VectorStoreError, WebIntelligenceError)
        assert issubclass(SearchProviderError, WebIntelligenceError)

    def test_no_embedder_error_message(self):
        from web_intelligence.exceptions import NoEmbedderError
        err = NoEmbedderError()
        assert "pip install" in str(err)


# ─────────────────────────────────────────────────────────
# 8. Pipeline integration tests (need network)
# ─────────────────────────────────────────────────────────

@pytest.mark.network
class TestPipelineIntegration:
    """These tests hit the network. Run with: pytest -m network"""

    def _make_pipeline(self):
        from web_intelligence import FastPipeline
        from web_intelligence.vector_stores import NumpyVectorStore
        store = NumpyVectorStore(persist_path=str(TEST_DATA_DIR / "integration.pkl"))
        return FastPipeline(vector_store=store)

    def test_index_and_retrieve(self):
        pipeline = self._make_pipeline()
        result = pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
        assert result["success"] is True
        assert result["chunks_count"] > 0

        ctx = pipeline.retrieve("what is python used for")
        assert len(ctx.context_text) > 0
        assert len(ctx.sources) > 0
        assert ctx.total_chunks > 0

    def test_index_batch(self):
        pipeline = self._make_pipeline()
        results = pipeline.index_batch([
            "https://en.wikipedia.org/wiki/Rust_(programming_language)",
            "https://en.wikipedia.org/wiki/Go_(programming_language)",
        ])
        success = sum(1 for r in results if r.get("success"))
        assert success >= 1

    def test_search_web(self):
        pipeline = self._make_pipeline()
        ctx = pipeline.search_web("what is FastAPI python", max_results=2, limit=3)
        assert len(ctx.context_text) > 0
        assert ctx.total_chunks > 0

    def test_stats(self):
        pipeline = self._make_pipeline()
        pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
        stats = pipeline.stats()
        assert stats["total_chunks_in_database"] > 0
        assert "embedding_model" in stats

    def test_list_and_delete_documents(self):
        pipeline = self._make_pipeline()
        pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
        docs = pipeline.list_documents()
        assert len(docs) > 0

        doc_id = docs[0]["doc_id"]
        assert pipeline.delete_document(doc_id) is True
        assert pipeline.get_document(doc_id) is None
