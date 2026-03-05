"""
Centralized configuration for Web Intelligence.

Loads settings from environment variables and .env files with sensible defaults.
All settings can be overridden programmatically.

Environment variables:
    WI_CRAWLER_MAX_CONCURRENT  - Max parallel HTTP requests (default: 10)
    WI_CRAWLER_TIMEOUT         - Request timeout in seconds (default: 15)
    WI_CRAWLER_MAX_RETRIES     - Retry failed requests (default: 3)
    WI_CRAWLER_REQUESTS_PER_SEC - Rate limit (default: 0 = unlimited)
    WI_CRAWLER_RESPECT_ROBOTS  - Respect robots.txt (default: true)
    WI_CHUNK_SIZE              - Words per chunk (default: 400)
    WI_CHUNK_OVERLAP           - Overlap words between chunks (default: 50)
    WI_EMBEDDING_MODEL         - Sentence-transformers model (default: all-MiniLM-L6-v2)
    WI_EMBEDDING_DEVICE        - Force 'cuda' or 'cpu' (default: auto-detect)
    WI_VECTOR_STORE_PATH       - Storage path (default: ./data)
    WI_COLLECTION_NAME         - Collection name (default: web_content)
    WI_CACHE_ENABLED           - Enable caching (default: true)
    WI_SERVER_HOST             - API server host (default: 0.0.0.0)
    WI_SERVER_PORT             - API server port (default: 8000)
"""

import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


def _load_dotenv():
    """Attempt to load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        for candidate in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
            if candidate.exists():
                load_dotenv(candidate)
                return True
    except ImportError:
        pass
    return False


_load_dotenv()


def _env(key: str, default: Any = None, cast: type = str) -> Any:
    """Read an environment variable with type casting."""
    val = os.environ.get(key, default)
    if val is None:
        return None
    if cast == bool:
        return str(val).lower() in ("1", "true", "yes", "on")
    return cast(val)


@dataclass
class CrawlerConfig:
    """Settings for the web crawler."""
    max_concurrent: int = _env("WI_CRAWLER_MAX_CONCURRENT", 10, int)
    timeout: int = _env("WI_CRAWLER_TIMEOUT", 15, int)
    max_retries: int = _env("WI_CRAWLER_MAX_RETRIES", 3, int)
    requests_per_second: float = _env("WI_CRAWLER_REQUESTS_PER_SEC", 0.0, float)
    respect_robots: bool = _env("WI_CRAWLER_RESPECT_ROBOTS", True, bool)
    user_agent: str = _env(
        "WI_CRAWLER_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class ChunkerConfig:
    """Settings for text chunking."""
    chunk_size: int = _env("WI_CHUNK_SIZE", 400, int)
    chunk_overlap: int = _env("WI_CHUNK_OVERLAP", 50, int)


@dataclass
class EmbeddingConfig:
    """Settings for the embedding model."""
    model_name: str = _env("WI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    device: Optional[str] = _env("WI_EMBEDDING_DEVICE", None)
    use_cache: bool = _env("WI_EMBEDDING_CACHE", True, bool)
    batch_size: int = _env("WI_EMBEDDING_BATCH_SIZE", 64, int)


@dataclass
class VectorStoreConfig:
    """Settings for the vector database."""
    persist_directory: str = _env("WI_VECTOR_STORE_PATH", "./data")
    collection_name: str = _env("WI_COLLECTION_NAME", "web_content")


@dataclass
class SearchConfig:
    """Settings for search and context retrieval."""
    default_limit: int = _env("WI_SEARCH_LIMIT", 5, int)
    max_context_words: int = _env("WI_SEARCH_MAX_CONTEXT_WORDS", 3000, int)
    include_sources: bool = _env("WI_SEARCH_INCLUDE_SOURCES", True, bool)
    min_relevance_score: float = _env("WI_SEARCH_MIN_SCORE", 0.0, float)


@dataclass
class ServerConfig:
    """Settings for the REST API server."""
    host: str = _env("WI_SERVER_HOST", "0.0.0.0")
    port: int = _env("WI_SERVER_PORT", 8000, int)
    reload: bool = _env("WI_SERVER_RELOAD", False, bool)


@dataclass
class Config:
    """
    Master configuration object for Web Intelligence.

    All settings are loaded from environment variables with sensible defaults.
    Override any setting programmatically:

        config = Config()
        config.embedding.model_name = "all-mpnet-base-v2"
        config.chunker.chunk_size = 500
    """
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cache_enabled: bool = _env("WI_CACHE_ENABLED", True, bool)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def copy(self) -> "Config":
        """Return a deep copy so mutations don't affect the original."""
        return copy.deepcopy(self)


def default_config() -> Config:
    """Return a fresh Config instance (no shared mutable state)."""
    return Config()
