# Web Intelligence

**Fetch the internet, serve it to any AI.**

Web Intelligence is a Python library that crawls websites (or searches the web for you), extracts the useful text, and turns it into clean, formatted context that any LLM can read — OpenAI, Ollama, Groq, Anthropic, LangChain, or your own code. No API key needed for the core library.

---

## What does it do?

1. **You give it a URL** → it crawls the page, extracts clean text, splits it into chunks, embeds them, and stores them in a vector database.
2. **You ask a question** → it finds the most relevant chunks and gives you LLM-ready context.
3. **Or you give it just a question (no URL)** → it searches the web via DuckDuckGo, crawls the top results, indexes them, and gives you the answer context. One line of code.

---

## Install

```bash
pip install web-intelligence
```

The core library is lightweight (~50 MB). Pick optional extras based on what you need:

```bash
# Lightweight embeddings (recommended to start)
pip install web-intelligence[fastembed]

# GPU-accelerated embeddings (heavier, ~2 GB)
pip install web-intelligence[gpu]

# Web search (DuckDuckGo, no API key)
pip install web-intelligence[search]

# ChromaDB vector store (production-grade persistence)
pip install web-intelligence[chromadb]

# REST API server
pip install web-intelligence[server]

# Everything at once
pip install web-intelligence[all]
```

---

## Quick Start

### Index a website and ask questions

```python
from web_intelligence import FastPipeline

pipeline = FastPipeline()

# Crawl and index a page
pipeline.index_url("https://en.wikipedia.org/wiki/Python_(programming_language)")

# Ask a question — get formatted context for any LLM
ctx = pipeline.retrieve("what is python used for?")
print(ctx.context_text)     # clean text, ready for any LLM
print(ctx.sources)          # source URLs
messages = ctx.as_messages() # OpenAI-compatible message format
```

### Search the web (no URL needed)

```python
from web_intelligence import FastPipeline

pipeline = FastPipeline()

# One line: searches DuckDuckGo → crawls top results → indexes → retrieves
ctx = pipeline.search_web("latest features in Python 3.12")
print(ctx.context_text)
```

### Use with any LLM (Groq + LangChain example)

```python
from web_intelligence import FastPipeline
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

pipeline = FastPipeline()
ctx = pipeline.search_web("what is FastAPI framework")

llm = ChatGroq(model="llama-3.3-70b-versatile")
response = llm.invoke(ctx.as_messages())
print(response.content)
```

### Index multiple pages at once

```python
urls = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://fastapi.tiangolo.com/",
    "https://docs.pydantic.dev/latest/",
]
results = pipeline.index_batch(urls)
```

---

## How It Works (Step by Step)

### When you give it a URL (`index_url`):
1. **Crawl** — Fetches the page using HTTP/2 (with retry, rate limiting, robots.txt).
2. **Extract** — Strips HTML, ads, navbars. Keeps only the useful article text.
3. **Chunk** — Splits the text into overlapping pieces (~400 words each).
4. **Embed** — Converts each chunk into a vector (a list of numbers) that captures its meaning.
5. **Store** — Saves the vectors in a vector database for fast search.

### When you ask a question (`retrieve`):
1. **Embed the question** — Converts your question into a vector.
2. **Search** — Finds the stored chunks most similar to your question.
3. **Format** — Packages the top results into clean, formatted text with source URLs.
4. **Return** — Gives you a `RetrievedContext` object with `.context_text`, `.sources`, and `.as_messages()`.

### When you search the web (`search_web`):
1. **Search DuckDuckGo** — Finds the top web pages for your question.
2. **Crawl** those pages.
3. **Extract + Chunk + Embed + Store** (same as above).
4. **Retrieve** — Searches the freshly indexed content and returns context.

All in one line of code.

---

## Configuration

```python
from web_intelligence import FastPipeline, Config

config = Config()
config.chunker.chunk_size = 500           # words per chunk
config.chunker.chunk_overlap = 75         # overlap between chunks
config.embedding.model_name = "all-mpnet-base-v2"  # better accuracy
config.crawler.max_retries = 5            # more retries

pipeline = FastPipeline(config=config)
```

Or use environment variables (in `.env` file):

```env
WI_CHUNK_SIZE=500
WI_EMBEDDING_MODEL=all-mpnet-base-v2
WI_CRAWLER_MAX_RETRIES=5
WI_SERVER_PORT=9000
```

---

## Pluggable Components

Swap out any component:

```python
from web_intelligence import FastPipeline
from web_intelligence.embedders import FastEmbedEmbedder
from web_intelligence.vector_stores import NumpyVectorStore

pipeline = FastPipeline(
    embedder=FastEmbedEmbedder(),             # lightweight, no GPU
    vector_store=NumpyVectorStore(),          # no ChromaDB needed
)
```

Available embedders: `SentenceTransformerEmbedder`, `FastEmbedEmbedder`, `OpenAIEmbedder`, `OllamaEmbedder`

Available vector stores: `ChromaVectorStore`, `NumpyVectorStore`

Available search providers: `DuckDuckGoSearchProvider`

---

## REST API

```bash
python main.py serve
```

Endpoints:
- `POST /index` — Index a URL
- `POST /index/batch` — Index multiple URLs
- `POST /retrieve` — Get LLM-ready context
- `POST /search-web` — Web search → context
- `GET /documents` — List indexed documents
- `GET /stats` — Pipeline statistics
- `GET /health` — Health check

---

## CLI

```bash
python main.py index https://example.com
python main.py search "what is python"
python main.py retrieve "explain decorators"
python main.py documents
python main.py stats
python main.py serve
```

---

## Project Structure

```
web_intelligence/
├── __init__.py              # Public API exports
├── optimized_pipeline.py    # Core pipeline (crawl → embed → store → retrieve)
├── config.py                # Configuration with env var support
├── async_crawler.py         # HTTP/2 crawler with retry + rate limiting
├── crawler.py               # Crawl result data class
├── extractor.py             # HTML → clean text extraction
├── chunker.py               # Text splitting into overlapping chunks
├── context_formatter.py     # Formats search results for LLMs
├── cache.py                 # URL + content + embedding caches
├── server.py                # FastAPI REST server
├── exceptions.py            # Custom exception types
├── _logging.py              # Logging setup
├── embedders/               # Pluggable embedding backends
│   ├── sentence_transformer.py
│   ├── fastembed_embedder.py
│   ├── openai_embedder.py
│   └── ollama_embedder.py
├── vector_stores/           # Pluggable vector store backends
│   ├── chroma_store.py
│   └── numpy_store.py
└── search_providers/        # Pluggable web search backends
    └── duckduckgo_provider.py
```

---

## License

MIT
