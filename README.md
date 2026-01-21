# Web Intelligence

Fast, production-ready web crawling, extraction, and semantic search library with GPU acceleration, async crawling, and intelligent caching.

## Features

- **Async Crawling**: Concurrent HTTP/2 requests with connection pooling
- **GPU Acceleration**: 10-50x faster embeddings with CUDA support
- **Smart Caching**: Three-tier caching (URL, content, embedding)
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Deduplication**: Content-based duplicate detection
- **100% Free**: No API keys, runs entirely on your machine

## Installation

```bash
pip install web-intelligence
```

### Requirements

- Python 3.12+
- Optional: CUDA-compatible GPU for acceleration

## Quick Start

```python
from web_intelligence import FastPipeline

# Initialize pipeline (auto-detects GPU)
pipeline = FastPipeline(
    cache_enabled=True,
    use_gpu=None  # Auto-detect
)

# Index a single URL
result = pipeline.index_url("https://example.com")
print(f"Indexed: {result['title']}, Chunks: {result['chunks_count']}")

# Batch indexing (10 URLs concurrently)
urls = [
    "https://python.org",
    "https://github.com",
    "https://stackoverflow.com"
]
results = pipeline.index_batch(urls)

# Semantic search
search_results = pipeline.search("python programming", limit=5)
for result in search_results:
    print(f"{result['metadata']['title']}: {result['score']:.3f}")
```

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Batch crawling | ~0.5s/URL | 10 concurrent connections |
| GPU embeddings | ~50ms/100 texts | With CUDA |
| CPU embeddings | ~500ms/100 texts | Fallback mode |
| Cached retrieval | <10ms | Instant lookup |

## Advanced Usage

### Custom Configuration

```python
pipeline = FastPipeline(
    storage_path="./my_data",
    cache_enabled=True,
    use_gpu=True,  # Force GPU
    embedding_model="all-MiniLM-L12-v2",  # Larger model
    use_embedding_cache=True
)
```

### Search with Filters

```python
results = pipeline.search("machine learning", limit=10)
for r in results:
    print(f"Title: {r['metadata']['title']}")
    print(f"URL: {r['metadata']['url']}")
    print(f"Score: {r['score']:.3f}")
    print(f"Text: {r['text'][:200]}...")
```

### Cache Management

```python
# View statistics
stats = pipeline.stats()
print(stats)

# Clear caches
pipeline.clear_all_caches()
```

## API Reference

### FastPipeline

Main interface for indexing and searching web content.

#### Methods

- `index_url(url: str, skip_cache: bool = False) -> Dict`
  - Index a single URL into the vector database
  
- `index_batch(urls: List[str], skip_cached: bool = True) -> List[Dict]`
  - Index multiple URLs concurrently
  
- `search(query: str, limit: int = 5) -> List[Dict]`
  - Semantic search across indexed content
  
- `stats() -> Dict`
  - Get pipeline statistics and metrics
  
- `clear_all_caches()`
  - Clear all caches (URL, content, embedding)

## Architecture

```
Web Pages → Async Crawler → HTML Extractor → Smart Chunker → Embedder → Vector DB
                ↓               ↓                ↓             ↓
            URL Cache    Content Cache    Sentence-aware  Embedding Cache
```

## Caching System

1. **URL Cache**: Tracks processed URLs to avoid re-crawling
2. **Content Cache**: Detects duplicate content across different URLs
3. **Embedding Cache**: Stores computed embeddings for instant retrieval

## Development

### Install from Source

```bash
git clone https://github.com/yourusername/web-intelligence.git
cd web-intelligence
pip install -e .
```

### Run Tests

```bash
python benchmark.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v0.2.0
- Added async HTTP/2 crawling
- Implemented three-tier caching system
- Added GPU acceleration support
- Sentence-aware text chunking
- Content-based deduplication

### v0.1.0
- Initial release
