# Web Intelligence

A 100% free, fully local library that crawls the web and serves clean, searchable content to **any LLM** — no API keys, no cloud, no cost.

You bring the LLM (Ollama, OpenAI, Anthropic, local Llama, whatever). This library handles everything else: crawl → extract → chunk → embed → store → retrieve.

---

## What Does This Do?

You give it URLs. It reads them, understands them, and gives you **LLM-ready context** you can feed directly to any model.

1. **Crawls the URL** — Async HTTP/2 requests, 10 pages at once.
2. **Extracts clean text** — Strips HTML, ads, nav bars, footers. Only keeps readable content.
3. **Splits into chunks** — ~400 word overlapping sentence-aware chunks. Context preserved.
4. **Embeds locally** — `all-MiniLM-L6-v2` runs on your machine. GPU auto-detected for 10–50x speed.
5. **Stores in ChromaDB** — Persisted on disk. Survives restarts.
6. **Retrieves by meaning** — Query in plain English → get the most relevant chunks, pre-formatted for your LLM.
7. **Three-tier caching** — Never re-crawls, re-embeds, or re-processes duplicates.

---

## Quick Start

```bash
pip install web-intelligence
```

```python
from web_intelligence import FastPipeline

pipeline = FastPipeline()

# Index a webpage
pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")

# Get LLM-ready context
ctx = pipeline.retrieve("what is python used for")

# Feed to ANY LLM — the library doesn't care which one you use
print(ctx.context_text)     # formatted text you paste into any prompt
print(ctx.sources)          # source URLs for citations
messages = ctx.as_messages()  # OpenAI-compatible messages format
```

---

## Use With Any LLM

### With Ollama (free, local)
```python
from web_intelligence import FastPipeline
import requests

pipeline = FastPipeline()
pipeline.index_url("https://docs.python.org/3/tutorial/")

ctx = pipeline.retrieve("how do I handle errors in python")

# Ollama API
response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2",
    "messages": ctx.as_messages(),  # ready-made messages
    "stream": False
})
print(response.json()["message"]["content"])
```

### With OpenAI
```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var
response = client.chat.completions.create(
    model="gpt-4o",
    messages=ctx.as_messages()
)
print(response.choices[0].message.content)
```

### With Anthropic
```python
import anthropic

client = anthropic.Anthropic()
msgs = ctx.as_messages()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=msgs[0]["content"],
    messages=[{"role": "user", "content": msgs[1]["content"]}]
)
print(response.content[0].text)
```

### With LiteLLM (any provider)
```python
from litellm import completion

response = completion(
    model="ollama/llama3.2",  # or "gpt-4o", "claude-sonnet-4-20250514", etc.
    messages=ctx.as_messages()
)
print(response.choices[0].message.content)
```

### Manual prompt injection (any LLM)
```python
context = pipeline.get_context_for_llm("how does auth work")
prompt = f"Based on this context:\n{context}\n\nQuestion: how does auth work?"
# send `prompt` to literally any LLM
```

---

## Batch Indexing

```python
urls = [
    "https://python.org/about/",
    "https://docs.python.org/3/library/asyncio.html",
    "https://realpython.com/async-io-python/",
]

results = pipeline.index_batch(urls)  # all crawled concurrently

# Search across everything
ctx = pipeline.retrieve("how does async work in python", limit=5)
```

---

## Document Management

```python
# List everything indexed
docs = pipeline.list_documents()
for d in docs:
    print(f"{d['title']} — {d['chunk_count']} chunks — {d['url']}")

# Get full text of a document
doc = pipeline.get_document("doc-id-here")
print(doc["full_text"])

# Delete a document
pipeline.delete_document("doc-id-here")

# Delete by URL
pipeline.delete_url("https://example.com/old-page")
```

---

## REST API

Start the server so any app or LLM agent can query your indexed content over HTTP:

```bash
pip install web-intelligence[server]
python -m web_intelligence.server
# or
python main.py serve
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/index` | Index a single URL |
| POST | `/index/batch` | Index multiple URLs |
| POST | `/search` | Raw semantic search |
| POST | `/retrieve` | **Get LLM-ready context** |
| GET | `/documents` | List indexed documents |
| GET | `/documents/{id}` | Get a specific document |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/stats` | Pipeline statistics |
| GET | `/health` | Health check |

### Example: retrieve context via API

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "how does python handle memory", "format": "numbered"}'
```

Response includes `context_text` (string to inject into prompts) and `messages` (OpenAI-compatible chat format).

---

## Context Formats

Choose the output format that works best for your LLM:

```python
# Plain text — simple concatenation
ctx = pipeline.retrieve("query", format="plain")

# Numbered sources — citations like [Source 1], [Source 2]
ctx = pipeline.retrieve("query", format="numbered")

# Structured XML — best for Claude, GPT-4
ctx = pipeline.retrieve("query", format="structured")
```

Each returns a `RetrievedContext` with:
- `.context_text` — formatted string
- `.sources` — list of source URLs + titles
- `.as_messages()` — OpenAI-compatible messages list
- `.to_dict()` — JSON-serializable dict
- `.total_words`, `.total_chunks` — metadata

---

## CLI

```bash
python main.py index https://example.com
python main.py index https://site1.com https://site2.com
python main.py search "what is python"
python main.py retrieve "how does async work"
python main.py documents
python main.py delete <doc_id>
python main.py stats
python main.py serve
python main.py clear
```

---

## Configuration

```python
from web_intelligence import FastPipeline, Config

# Use defaults (all configurable via env vars)
pipeline = FastPipeline()

# Or override programmatically
config = Config()
config.chunker.chunk_size = 500
config.chunker.chunk_overlap = 75
config.embedding.model_name = "all-mpnet-base-v2"

pipeline = FastPipeline(config=config)

# Quick overrides
pipeline = FastPipeline(
    storage_path="./my_data",
    use_gpu=True,
    embedding_model="all-MiniLM-L6-v2",
    cache_enabled=True,
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WI_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `WI_EMBEDDING_DEVICE` | auto | `cuda` or `cpu` |
| `WI_CHUNK_SIZE` | `400` | Words per chunk |
| `WI_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `WI_VECTOR_STORE_PATH` | `./data/chroma` | ChromaDB storage |
| `WI_CRAWLER_MAX_CONCURRENT` | `10` | Max parallel requests |
| `WI_CRAWLER_TIMEOUT` | `15` | Request timeout (seconds) |
| `WI_CACHE_ENABLED` | `true` | Enable/disable caching |
| `WI_SERVER_PORT` | `8000` | API server port |

---

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Crawling 10 URLs | ~5s total | Async + concurrent |
| Embedding (GPU) | ~50ms per 100 chunks | CUDA auto-detected |
| Embedding (CPU) | ~500ms per 100 chunks | Fallback |
| Search | ~10ms | Semantic similarity |
| Cached URL lookup | <1ms | Instant |

---

## API Reference

### `FastPipeline`

| Method | Description |
|--------|-------------|
| `index_url(url)` | Crawl and index a single URL |
| `index_batch(urls)` | Crawl and index multiple URLs concurrently |
| `search(query, limit, filter, min_score)` | Raw semantic search → list of chunk dicts |
| `retrieve(query, limit, format, max_context_words)` | **LLM-ready context** → `RetrievedContext` |
| `get_context_for_llm(query)` | Shorthand → returns context string directly |
| `list_documents()` | List all indexed documents |
| `get_document(doc_id)` | Get full text + chunks for a document |
| `delete_document(doc_id)` | Delete a document |
| `delete_url(url)` | Delete all content from a URL |
| `stats()` | Pipeline statistics |
| `clear_all()` | Delete everything |
| `clear_caches()` | Clear caches only, keep indexed data |

---

## Architecture

```
You give URLs
      ↓
Async Crawler (httpx, HTTP/2, 10 concurrent)
      ↓
HTML Extractor (trafilatura — strips ads, nav, noise)
      ↓
Smart Chunker (400-word overlapping sentence-aware chunks)
      ↓
FastEmbedder (sentence-transformers, GPU auto-detected, cached)
      ↓
ChromaDB Vector Store (persistent, local)
      ↓
Retrieve → LLM-ready context (plain / numbered / structured)
      ↓
YOU feed it to YOUR LLM (Ollama, OpenAI, Anthropic, anything)
```

---

## How Caching Works

- **URL Cache** — Already indexed `python.org`? Calling `index_url("python.org")` again returns instantly.
- **Content Cache** — Two URLs with identical text? Only indexed once. No duplicates.
- **Embedding Cache** — Same text chunk? Reuses the saved vector. No GPU/CPU work repeated.

---

## Installation from Source

```bash
git clone https://github.com/yourusername/web-intelligence.git
cd web-intelligence
pip install -e ".[server,dev]"
```

## Requirements

- Python 3.10+
- Optional: CUDA GPU for 10–50x faster embedding
- Optional: `fastapi` + `uvicorn` for REST API (`pip install web-intelligence[server]`)

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

## Changelog

### v0.3.0
- **LLM-agnostic retrieval** — `retrieve()` returns context for any LLM
- **Context formatters** — plain, numbered, structured (XML) output
- **OpenAI-compatible messages** — `ctx.as_messages()` works with any chat API
- **Document management** — list, get, delete documents
- **REST API** — full FastAPI server for HTTP access
- **CLI** — index, search, retrieve, serve from terminal
- **Configuration system** — env vars + programmatic config
- **Metadata filtering** — filter search by URL, domain, etc.
- **Cleaned dependencies** — no forced LLM libraries

### v0.2.0
- Async HTTP/2 crawling
- Three-tier caching
- GPU acceleration
- Sentence-aware chunking
- Content deduplication

### v0.1.0
- Initial release
